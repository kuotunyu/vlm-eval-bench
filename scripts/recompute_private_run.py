"""Recompute corrected metrics from a complete private prediction matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("results/audit/evaluation_config.yaml"))
    parser.add_argument(
        "--archived-manifest", type=Path, default=Path("results/audit/run_manifest.json")
    )
    args = parser.parse_args(argv)

    from vlmeval.config import load_config
    from vlmeval.private_rescore import recompute_private_run

    manifest = recompute_private_run(
        load_config(args.config, load_environment=False),
        args.input_dir,
        args.output_dir,
        args.archived_manifest,
    )
    print(
        "corrected rescore written: "
        f"{len(manifest['scope']['models'])} models, "
        f"{len(manifest['scope']['tasks'])} tasks, "
        f"{manifest['scope']['total_rows']} rows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
