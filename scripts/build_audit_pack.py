"""Build the committed, dataset-content-free audit pack from local predictions.

Usage:
    uv run python scripts/build_audit_pack.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--audit-dir", type=Path, default=Path("results/audit"))
    args = parser.parse_args(argv)

    from vlmeval.audit import build_audit_pack

    manifest = build_audit_pack(args.config, args.predictions_dir, args.audit_dir)
    print(
        f"wrote {len(manifest['files'])} files / {manifest['scope']['total_rows']} rows "
        f"to {args.audit_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
