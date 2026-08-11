"""Run orchestration: sampling, cost gate, execution, JSONL persistence, summaries."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tqdm import tqdm

from vlmeval.cache import ResponseCache
from vlmeval.config import AppConfig, ModelConfig
from vlmeval.cost import CostMeter, estimate_row, format_estimate_table, gate
from vlmeval.models import build_model
from vlmeval.models.base import BaseModel
from vlmeval.tasks import build_task
from vlmeval.tasks.base import BaseTask, Sample

EXIT_OK = 0
EXIT_GATE = 2
EXIT_CAP = 3


def _cache_params_for(cfg: AppConfig, params) -> dict:
    """Same dict BaseModel.cache_params builds — used for pre-run cache counting
    without instantiating providers."""
    d = params.cache_dict()
    d["image_max_side"] = cfg.run.image_max_side
    d["jpeg_quality"] = cfg.run.jpeg_quality
    return d


def _predictions_path(cfg: AppConfig, model_id: str, task_name: str) -> Path:
    return cfg.run.output_dir / "predictions" / f"{model_id}__{task_name}.jsonl"


def _load_rows(path: Path) -> dict[str, dict]:
    """Latest JSONL row per sample_id (later lines win)."""
    rows: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    rows[row["sample_id"]] = row
    return rows


async def _run_model_task(
    model: BaseModel,
    task: BaseTask,
    samples: list[Sample],
    out_path: Path,
    meter: CostMeter,
    scale: str,
) -> None:
    existing = _load_rows(out_path)
    done = {sid for sid, row in existing.items() if not row.get("error")}
    todo = [s for s in samples if s.sample_id not in done]
    if not todo:
        print(f"  {model.cfg.id}/{task.name}: all {len(samples)} samples already done")
        return

    params = task.gen_params()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stop = False
    pbar = tqdm(total=len(todo), desc=f"{model.cfg.id}/{task.name}", unit="q")

    with out_path.open("a", encoding="utf-8") as f:

        async def one(sample: Sample) -> None:
            nonlocal stop
            if stop:
                pbar.update(1)
                return
            resp = await model.generate(
                sample.image_jpeg, sample.prompt, params, task=task.name, sample_id=sample.sample_id
            )
            scored = {"score": 0.0} if resp.error else task.score_one(resp.text, sample.reference)
            aux = {k: v for k, v in scored.items() if k not in ("score", "pred_clean")}
            row = {
                "sample_id": sample.sample_id,
                "model": model.cfg.id,
                "task": task.name,
                "scale": scale,
                "prompt": sample.prompt,
                "pred_raw": resp.text,
                "pred_clean": scored.get("pred_clean"),
                "reference": sample.reference,
                "meta": sample.meta,
                "score": scored["score"],
                "aux": aux,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "usage_source": resp.usage.source,
                "cost_usd": resp.cost_usd,
                "latency_s": resp.latency_s,
                "cached": resp.cached,
                "error": resp.error,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            billed_cost = None if model.cfg.is_local else resp.cost_usd
            if not meter.add(billed_cost, cached=resp.cached):
                stop = True
            pbar.update(1)

        pending = iter(todo)

        async def worker() -> None:
            while not stop:
                try:
                    sample = next(pending)
                except StopIteration:
                    return
                await one(sample)

        worker_count = min(model.cfg.rate_limit.concurrency, len(todo))
        await asyncio.gather(*(worker() for _ in range(worker_count)))
    pbar.close()

    if stop:
        raise CostCapExceeded(
            f"cost cap ${meter.cap_usd:.2f} exceeded (spent ${meter.spent_usd:.4f}); progress persisted"
        )


class CostCapExceeded(RuntimeError):
    pass


def _summarize(
    cfg: AppConfig,
    tasks: list[BaseTask],
    model_cfgs: list[ModelConfig],
    task_samples: dict[str, list[Sample]],
    scale: str,
) -> None:
    summaries_dir = cfg.run.output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        wanted = [s.sample_id for s in task_samples[task.name]]
        summary: dict[str, dict] = {}
        for mc in model_cfgs:
            rows_by_id = _load_rows(_predictions_path(cfg, mc.id, task.name))
            rows = [rows_by_id[sid] for sid in wanted if sid in rows_by_id]
            if rows:
                summary[mc.id] = task.aggregate(rows)
        out = summaries_dir / f"{scale}_{task.name}.json"
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n[{task.name}] ({scale}, n per model varies with errors)")
        for mid, agg in summary.items():
            ci = agg.get("ci95", [0, 0])
            print(
                f"  {mid:<28} {agg['metric']}={agg['score']:.4f} [{ci[0]:.3f}, {ci[1]:.3f}]"
                f"  err={agg['error_rate']:.1%}"
            )


def _print_extrapolation(
    cfg: AppConfig,
    tasks: list[BaseTask],
    model_cfgs: list[ModelConfig],
    task_samples: dict[str, list[Sample]],
) -> None:
    print("\nExtrapolated FULL-scale cost from actual mini spend (uncached rows only):")
    print(f"{'model':<28}{'task':<10}{'$/question':>12}{'n_full':>8}{'est full $':>12}")
    grand = 0.0
    for mc in model_cfgs:
        if mc.is_local:
            continue
        for task in tasks:
            rows_by_id = _load_rows(_predictions_path(cfg, mc.id, task.name))
            wanted = [s.sample_id for s in task_samples[task.name]]
            costs = [
                rows_by_id[sid]["cost_usd"]
                for sid in wanted
                if sid in rows_by_id and rows_by_id[sid].get("cost_usd") is not None
            ]
            if not costs:
                continue
            per_q = sum(costs) / len(costs)
            est = per_q * task.cfg.n_full
            grand += est
            print(f"{mc.id:<28}{task.name:<10}{per_q:>12.6f}{task.cfg.n_full:>8}{est:>12.4f}")
    print(f"{'TOTAL (all API models, full scale, before cache credit)':<58}{grand:>12.4f}")


def run(
    cfg: AppConfig,
    scale: str,
    task_names: list[str],
    model_ids: list[str],
    dry_run: bool = False,
    yes: bool = False,
    no_cache: bool = False,
) -> int:
    cache = ResponseCache(cfg.run.cache_db)
    tasks = [build_task(cfg.task(n), cfg.run) for n in task_names]
    model_cfgs = [cfg.model(mid) for mid in model_ids]

    task_samples: dict[str, list[Sample]] = {}
    for t in tasks:
        print(f"[{t.name}] loading samples ({scale}) ...")
        task_samples[t.name] = t.load_samples(scale)
        print(f"[{t.name}] {len(task_samples[t.name])} samples ready")

    # pre-run estimate + gate
    est_rows = []
    for mc in model_cfgs:
        for t in tasks:
            samples = task_samples[t.name]
            params = t.gen_params()
            gp = _cache_params_for(cfg, params)
            n_cached = (
                0
                if no_cache
                else sum(
                    cache.has(
                        ResponseCache.make_key(
                            mc.id,
                            t.name,
                            s.sample_id,
                            s.prompt,
                            s.image_jpeg,
                            gp,
                        )
                    )
                    for s in samples
                )
            )
            avg_chars = sum(len(s.prompt) for s in samples) / len(samples) if samples else 0
            est_rows.append(
                estimate_row(mc, t.name, len(samples), n_cached, avg_chars, t.cfg.max_output_tokens)
            )
    print("\n" + format_estimate_table(est_rows, cfg.run.cost_cap_usd))

    if dry_run:
        print("\n--dry-run: no API calls made.")
        return EXIT_OK
    if not gate(est_rows, cfg.run.cost_cap_usd, yes):
        print("\nAborted by cost-cap gate. Re-run with --yes to confirm interactively.")
        return EXIT_GATE

    meter = CostMeter(cfg.run.cost_cap_usd)
    api_models = [mc for mc in model_cfgs if not mc.is_local]
    local_models = [mc for mc in model_cfgs if mc.is_local]

    async def _execute() -> None:
        # API models first; local models last, strictly sequential with unload between.
        # One event loop for the whole run: rate-limiter primitives bind to it once.
        for mc in api_models + local_models:
            model = build_model(mc, cfg.run, cache)
            model.use_cache = not no_cache
            try:
                for t in tasks:
                    await _run_model_task(
                        model,
                        t,
                        task_samples[t.name],
                        _predictions_path(cfg, mc.id, t.name),
                        meter,
                        scale,
                    )
            finally:
                model.unload()

    try:
        asyncio.run(_execute())
    except CostCapExceeded as e:
        print(f"\n!! {e}")
        _summarize(cfg, tasks, model_cfgs, task_samples, scale)
        return EXIT_CAP

    _summarize(cfg, tasks, model_cfgs, task_samples, scale)
    print(
        f"\nActual new spend this run: ${meter.spent_usd:.4f}"
        f" (cache-credited: ${meter.cached_usd:.4f})"
    )
    if scale == "mini":
        _print_extrapolation(cfg, tasks, model_cfgs, task_samples)
        print("\nScale up with: python run.py --scale full")
    return EXIT_OK
