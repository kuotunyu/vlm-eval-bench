"""Regenerate results/leaderboard.md and charts from the prediction JSONL files.

Usage:
    python report.py [--config config.yaml] [--results-dir results]

Deterministic: running twice over the same JSONL produces byte-identical output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--results-dir", default=None, help="override run.output_dir")
    args = p.parse_args(argv)

    from vlmeval.config import load_config
    from vlmeval.reporting import generate_report

    cfg = load_config(args.config)
    results_dir = Path(args.results_dir) if args.results_dir else cfg.run.output_dir
    generate_report(cfg, results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
