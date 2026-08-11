"""Smoke test: run a few samples through selected models and print everything.

Usage:
    python scripts/smoke_test.py --models gemini-3.1-flash-lite [--tasks cord,docvqa] [--n 2]

Prints raw response text, usage, cost, latency, cached flag, and score per sample.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


async def _run(args: argparse.Namespace) -> int:
    from vlmeval.cache import ResponseCache
    from vlmeval.config import load_config
    from vlmeval.models import build_model
    from vlmeval.tasks import build_task

    cfg = load_config(args.config)
    cache = ResponseCache(cfg.run.cache_db)

    for task_name in args.tasks.split(","):
        task = build_task(cfg.task(task_name), cfg.run)
        samples = task.load_samples("mini")[: args.n]
        params = task.gen_params()
        for model_id in args.models.split(","):
            model = build_model(cfg.model(model_id), cfg.run, cache)
            try:
                for s in samples:
                    resp = await model.generate(
                        s.image_jpeg,
                        s.prompt,
                        params,
                        task=task.name,
                        sample_id=s.sample_id,
                    )
                    print(f"\n=== {model_id} / {task.name} / {s.sample_id} ===")
                    print(
                        f"cached={resp.cached}  latency={resp.latency_s:.2f}s  "
                        f"in={resp.usage.input_tokens}  out={resp.usage.output_tokens}  "
                        f"cost=${resp.cost_usd if resp.cost_usd is not None else float('nan'):.6f}"
                    )
                    if resp.error:
                        print(f"ERROR: {resp.error}")
                        continue
                    scored = task.score_one(resp.text, s.reference)
                    print(f"score={scored['score']:.4f}")
                    print(f"raw: {resp.text[:400]}")
                    print(f"ref: {str(s.reference)[:200]}")
            finally:
                model.unload()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", required=True, help="comma-separated model ids")
    p.add_argument("--tasks", default="cord", help="comma-separated task names")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args(argv)
    # One loop for the whole smoke run: the model's semaphore and RPM lock
    # must not be carried across a new event loop for each sample.
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
