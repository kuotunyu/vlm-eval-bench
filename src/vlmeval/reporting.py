"""Leaderboard + charts, regenerated deterministically from prediction JSONL files.

No wall-clock timestamps go into leaderboard.md, so re-running report.py over
the same JSONL produces byte-identical output (fixed bootstrap seed included).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlmeval.audit import read_latest_rows
from vlmeval.config import AppConfig
from vlmeval.tasks import build_task

LOCAL_FLAG = "†"


# -- data loading ------------------------------------------------------------


def _dedup_rows(path: Path) -> list[dict]:
    """Latest row per sample_id, in canonical sample-id order."""
    rows = read_latest_rows(path) if path.exists() else []
    return sorted(rows, key=lambda row: str(row["sample_id"]))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[idx]


def _fmt_ci(agg: dict) -> str:
    lo, hi = agg["ci95"]
    return f"{agg['score']:.3f} [{lo:.3f}, {hi:.3f}]"


def _complete_model_ids(
    cfg: AppConfig,
    rows_by: dict[tuple[str, str], list[dict]],
) -> tuple[list[str], dict[str, dict[str, tuple[int, int]]]]:
    """Return full-run model ids and progress for partial runs.

    A model belongs on the comparison leaderboard only after every configured
    task has its full sample count.  This prevents an interrupted run from
    looking competitive because it happened to stop on an easy prefix.
    """
    complete: list[str] = []
    incomplete: dict[str, dict[str, tuple[int, int]]] = {}
    for model in cfg.models:
        if not model.enabled:
            continue
        progress = {
            task.name: (len(rows_by.get((model.id, task.name), [])), task.n_full)
            for task in cfg.tasks
        }
        if progress and all(have >= expected for have, expected in progress.values()):
            complete.append(model.id)
        elif any(have for have, _ in progress.values()):
            incomplete[model.id] = progress
    return complete, incomplete


def _average_score(model_id: str, task_names: list[str], aggs: dict) -> float:
    return sum(aggs[(model_id, name)]["score"] for name in task_names) / len(task_names)


def _cost_per_100(
    model_id: str,
    task_names: list[str],
    rows_by: dict[tuple[str, str], list[dict]],
) -> float:
    rows = [row for name in task_names for row in rows_by.get((model_id, name), [])]
    return (sum(row.get("cost_usd") or 0.0 for row in rows) / len(rows) * 100) if rows else 0.0


def _analysis_lines(
    task_names: list[str],
    model_ids: list[str],
    local_ids: set[str],
    aggs: dict[tuple[str, str], dict],
    rows_by: dict[tuple[str, str], list[dict]],
) -> list[str]:
    """Build deterministic, evidence-backed interpretation from full results."""
    if not model_ids or not task_names:
        return ["No complete full-scale runs are available for analysis."]

    averages = {mid: _average_score(mid, task_names, aggs) for mid in model_ids}
    best = max(model_ids, key=lambda mid: (averages[mid], mid))
    lines = [
        f"**Cross-task summary.** `{best}` has the highest unweighted {len(task_names)}-task "
        f"arithmetic mean (**{averages[best]:.3f}**) among these runs. This is a navigation aid, not a "
        "universal model ranking: each benchmark measures a different behavior and the "
        "per-task confidence intervals remain the primary evidence."
    ]

    base_id = "qwen3vl-8b-base"
    tuned_id = "qwen3vl-8b-receipt-qlora"
    if base_id in model_ids and tuned_id in model_ids:
        deltas = {
            name: aggs[(tuned_id, name)]["score"] - aggs[(base_id, name)]["score"]
            for name in task_names
        }
        cord_delta = deltas.get("cord")
        ood = [(name, delta) for name, delta in deltas.items() if name != "cord"]
        ood_text = " and ".join(f"{name} {delta:+.3f}" for name, delta in ood)
        cord_text = f"CORD {cord_delta:+.3f}" if cord_delta is not None else "the in-domain task"
        lines.append(
            f"**What the QLoRA changed.** Against the same 8B base model, the adapter moves "
            f"{cord_text}; outside its receipt-training domain the changes are {ood_text}. "
            "That pattern supports a narrow, useful adaptation rather than a blanket claim "
            "that fine-tuning improves every vision-language task."
        )

    api_ids = [mid for mid in model_ids if mid not in local_ids]
    if api_ids:
        best_api = max(api_ids, key=lambda mid: (averages[mid], mid))
        api_costs = {mid: _cost_per_100(mid, task_names, rows_by) for mid in api_ids}
        cheapest_api = min(api_ids, key=lambda mid: (api_costs[mid], mid))
        if best_api == cheapest_api:
            tradeoff = (
                f"`{best_api}` is both the strongest API average (**{averages[best_api]:.3f}**) "
                f"and the least expensive measured API run (**${api_costs[best_api]:.4f}/100 questions**)."
            )
        else:
            tradeoff = (
                f"`{best_api}` has the strongest API average (**{averages[best_api]:.3f}**), while "
                f"`{cheapest_api}` is least expensive at **${api_costs[cheapest_api]:.4f}/100 questions** "
                f"(average **{averages[cheapest_api]:.3f}**)."
            )
        lines.append(
            "**API trade-off.** "
            + tradeoff
            + " These are configured-price estimates reconstructed from provider-returned usage, "
            "not invoice-reconciled charges. Local-model dollars are separately marked as imputed "
            "GPU rental and must not be read as API prices."
        )

    all_rows = [row for mid in model_ids for name in task_names for row in rows_by[(mid, name)]]
    error_rate = sum(bool(row.get("error")) for row in all_rows) / len(all_rows)
    sample_scope = ", ".join(
        f"{name} n={len(rows_by[(model_ids[0], name)])}" for name in task_names
    )
    lines.append(
        f"**Reliability and scope.** Final deduplicated rows have a combined terminal error rate of "
        f"**{error_rate:.2%}**; transient retry failures are not retained in that figure. Results "
        f"still describe fixed, seeded samples ({sample_scope}), "
        "not production traffic; local batch-1 latency excludes "
        "network time and therefore is not directly comparable with API round trips."
    )
    return lines


# -- report ------------------------------------------------------------------


def generate_report(cfg: AppConfig, results_dir: Path) -> None:
    tasks = [build_task(t, cfg.run) for t in cfg.tasks]
    pred_dir = results_dir / "predictions"
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    rows_by: dict[tuple[str, str], list[dict]] = {}
    for m in cfg.models:
        for t in tasks:
            rows = _dedup_rows(pred_dir / f"{m.id}__{t.name}.jsonl")
            if rows:
                rows_by[(m.id, t.name)] = rows

    model_ids, incomplete = _complete_model_ids(cfg, rows_by)
    local_ids = {m.id for m in cfg.models if m.is_local}
    aggs = {
        (mid, t.name): t.aggregate(rows_by[(mid, t.name)])
        for mid in model_ids
        for t in tasks
        if (mid, t.name) in rows_by
    }

    lines: list[str] = []
    w = lines.append

    w("# vlm-eval-bench — Leaderboard")
    w("")
    counts = ", ".join(
        f"{t.name} n={max((aggs[(m, t.name)]['n'] for m in model_ids if (m, t.name) in aggs), default=0)}"
        for t in tasks
    )
    w(f"Seed {cfg.run.seed} · {counts} · greedy decoding · identical prompts/images for all models")
    w(f"· 95% CI via percentile bootstrap (n={cfg.run.bootstrap_iters})")
    w("")

    if incomplete:
        w("## Incomplete runs excluded")
        w("")
        w("Interrupted runs are kept on disk for resume, but excluded from every comparison below.")
        w("")
        w("| Model | " + " | ".join(t.name for t in cfg.tasks) + " |")
        w("|" + "---|" * (len(cfg.tasks) + 1))
        for mid, progress in incomplete.items():
            cells = [f"{progress[t.name][0]}/{progress[t.name][1]}" for t in cfg.tasks]
            w(f"| {mid} | " + " | ".join(cells) + " |")
        w("")

    # 1. main score table -----------------------------------------------------
    w("## Scores")
    w("")
    header = (
        "| Model | "
        + " | ".join(
            {"docvqa": "DocVQA ANLS", "chartqa": "ChartQA RelaxedAcc", "cord": "CORD-v2 F1"}.get(
                t.name, t.name
            )
            for t in tasks
        )
        + " | Avg |"
    )
    w(header)
    w("|" + "---|" * (len(tasks) + 2))
    for mid in model_ids:
        cells, scores = [], []
        for t in tasks:
            agg = aggs.get((mid, t.name))
            if agg:
                cells.append(_fmt_ci(agg))
                scores.append(agg["score"])
            else:
                cells.append("—")
        avg = f"{sum(scores) / len(scores):.3f}" if len(scores) == len(tasks) else "—"
        name = mid + (LOCAL_FLAG if mid in local_ids else "")
        w(f"| {name} | " + " | ".join(cells) + f" | {avg} |")
    w("")
    chartqa_aggs = {m: aggs[(m, "chartqa")] for m in model_ids if (m, "chartqa") in aggs}
    if chartqa_aggs:
        w(
            "ChartQA split: "
            + " · ".join(
                f"{m} human={a['human_accuracy']} machine={a['machine_accuracy']}"
                for m, a in chartqa_aggs.items()
            )
        )
        w("")
    w(f"{LOCAL_FLAG} local model on RTX 4090 (WSL2), 4-bit, batch=1.")
    w("")

    # 2. cost table -------------------------------------------------------------
    w("## Cost")
    w("")
    w("| Model | Total $ | $/100 questions | Input tokens | Output tokens |")
    w("|---|---|---|---|---|")
    for mid in model_ids:
        rows = [r for t in tasks for r in rows_by.get((mid, t.name), [])]
        cost = sum(r["cost_usd"] or 0.0 for r in rows)
        n = len(rows)
        tin = sum(r["input_tokens"] or 0 for r in rows)
        tout = sum(r["output_tokens"] or 0 for r in rows)
        per100 = cost / n * 100 if n else 0.0
        suffix = LOCAL_FLAG if mid in local_ids else ""
        w(f"| {mid}{suffix} | {cost:.4f}{suffix} | {per100:.4f}{suffix} | {tin:,} | {tout:,} |")
    w("")
    w(
        f"{LOCAL_FLAG} local cost is *imputed*: RTX 4090 cloud-rental rate "
        f"(${cfg.run.gpu_rent_usd_per_hour:.2f}/hr) × measured inference wall time — not an API bill. "
        "Local token counts come from the tokenizer, not billed usage. "
        "OpenAI image/text input-token split is approximated with the documented patch formula."
    )
    w("")

    # 3. latency table -----------------------------------------------------------
    w("## Latency (seconds per question, uncached calls only)")
    w("")
    w("| Model | Task | Mean | p50 | p95 |")
    w("|---|---|---|---|---|")
    for mid in model_ids:
        for t in tasks:
            rows = [
                r
                for r in rows_by.get((mid, t.name), [])
                if not r.get("cached") and not r.get("error")
            ]
            lats = [r["latency_s"] for r in rows if r.get("latency_s")]
            if not lats:
                continue
            mean = sum(lats) / len(lats)
            suffix = LOCAL_FLAG if mid in local_ids else ""
            w(
                f"| {mid}{suffix} | {t.name} | {mean:.2f} | {_percentile(lats, 50):.2f} | {_percentile(lats, 95):.2f} |"
            )
    w("")
    w(
        f"{LOCAL_FLAG} local latency (batch=1, no network) is **not comparable** to API round-trip latency."
    )
    w("")

    # 4. reliability ---------------------------------------------------------------
    w("## Reliability")
    w("")
    w("| Model | Error rate | CORD valid-JSON rate |")
    w("|---|---|---|")
    for mid in model_ids:
        rows = [r for t in tasks for r in rows_by.get((mid, t.name), [])]
        err = sum(1 for r in rows if r.get("error")) / len(rows) if rows else 0.0
        cord_agg = aggs.get((mid, "cord"))
        vjr = f"{cord_agg['valid_json_rate']:.2%}" if cord_agg else "—"
        w(f"| {mid} | {err:.2%} | {vjr} |")
    w("")

    # 5. analysis --------------------------------------------------------------------
    w("## Analysis")
    w("")
    for paragraph in _analysis_lines([t.name for t in tasks], model_ids, local_ids, aggs, rows_by):
        w(paragraph)
        w("")

    # charts ----------------------------------------------------------------------
    charts = _make_charts(cfg, tasks, model_ids, local_ids, aggs, rows_by, charts_dir)
    if charts:
        w("## Charts")
        w("")
        for c in charts:
            w(f"![{c.stem}](charts/{c.name})")
        w("")

    out = results_dir / "leaderboard.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} and {len(charts)} charts")


# -- charts ---------------------------------------------------------------------


def _make_charts(cfg, tasks, model_ids, local_ids, aggs, rows_by, charts_dir: Path) -> list[Path]:
    made: list[Path] = []
    task_names = [t.name for t in tasks]

    # 1. grouped bar: scores by task with CI error bars
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / max(1, len(model_ids))
    for i, mid in enumerate(model_ids):
        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for j, tn in enumerate(task_names):
            agg = aggs.get((mid, tn))
            if not agg:
                continue
            xs.append(j + i * width)
            ys.append(agg["score"])
            yerr_lo.append(max(0.0, agg["score"] - agg["ci95"][0]))
            yerr_hi.append(max(0.0, agg["ci95"][1] - agg["score"]))
        if xs:
            ax.bar(
                xs,
                ys,
                width=width * 0.95,
                label=mid,
                yerr=[yerr_lo, yerr_hi],
                capsize=2,
                error_kw={"lw": 0.8},
            )
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(task_names))])
    ax.set_xticklabels(task_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Scores by task (95% bootstrap CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = charts_dir / "scores_by_task.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    # 2. latency p50/p95 horizontal bars (mean across tasks)
    fig, ax = plt.subplots(figsize=(8, 0.6 * max(4, len(model_ids)) + 1))
    labels, p50s, p95s = [], [], []
    for mid in model_ids:
        lats = [
            r["latency_s"]
            for tn in task_names
            for r in rows_by.get((mid, tn), [])
            if not r.get("cached") and not r.get("error") and r.get("latency_s")
        ]
        if not lats:
            continue
        labels.append(mid + (LOCAL_FLAG if mid in local_ids else ""))
        p50s.append(_percentile(lats, 50))
        p95s.append(_percentile(lats, 95))
    ys = range(len(labels))
    ax.barh([y + 0.2 for y in ys], p95s, height=0.35, label="p95", alpha=0.5)
    ax.barh([y - 0.2 for y in ys], p50s, height=0.35, label="p50")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"seconds per question ({LOCAL_FLAG} local batch=1, not comparable to API)")
    ax.set_title("Latency")
    ax.legend()
    fig.tight_layout()
    p = charts_dir / "latency_p50_p95.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    # 3. CORD per-field F1: base vs fine-tuned local model
    pair = [m for m in model_ids if m in local_ids]
    if len(pair) >= 2 and all((m, "cord") in aggs for m in pair[:2]):
        base_f, ft_f = (aggs[(m, "cord")]["fields"] for m in pair[:2])
        field_names = list(base_f)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        xs = range(len(field_names))
        ax.bar(
            [x - 0.2 for x in xs], [base_f[f]["f1"] for f in field_names], width=0.38, label=pair[0]
        )
        ax.bar(
            [x + 0.2 for x in xs], [ft_f[f]["f1"] for f in field_names], width=0.38, label=pair[1]
        )
        ax.set_xticks(list(xs))
        ax.set_xticklabels(field_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1.05)
        ax.set_title("CORD-v2 per-field F1: base vs QLoRA fine-tune")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = charts_dir / "cord_f1_breakdown.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    return made
