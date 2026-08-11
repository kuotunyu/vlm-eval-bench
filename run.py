"""Run VLM evaluations.

Usage:
    python run.py --scale mini|full [--tasks docvqa,chartqa,cord] [--models id1,id2]
                  [--config config.yaml] [--dry-run] [--yes] [--no-cache]

Exit codes: 0 ok / 2 cost-cap gate aborted / 3 mid-run cost cap hit.
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--scale",
        choices=["mini", "full"],
        required=True,
        help="mini = prefix subset; full = complete sample",
    )
    p.add_argument(
        "--tasks", default=None, help="comma-separated task names (default: all configured)"
    )
    p.add_argument(
        "--models", default=None, help="comma-separated model ids (default: all enabled)"
    )
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    p.add_argument(
        "--dry-run", action="store_true", help="sampling + cost estimate only, zero API calls"
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="allow proceeding past the cost-cap gate (still asks interactively)",
    )
    p.add_argument(
        "--no-cache", action="store_true", help="bypass cache reads (responses are still written)"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from vlmeval.config import load_config
    from vlmeval.runner import run

    cfg = load_config(args.config)
    task_names = args.tasks.split(",") if args.tasks else [t.name for t in cfg.tasks]
    model_ids = args.models.split(",") if args.models else [m.id for m in cfg.enabled_models()]
    return run(
        cfg,
        scale=args.scale,
        task_names=task_names,
        model_ids=model_ids,
        dry_run=args.dry_run,
        yes=args.yes,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    sys.exit(main())
