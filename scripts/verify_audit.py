"""Verify the committed audit pack and its published leaderboard values.

Usage:
    uv run python scripts/verify_audit.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("results/audit"))
    parser.add_argument("--leaderboard", type=Path, default=Path("results/archived_leaderboard.md"))
    parser.add_argument("--readme", type=Path)
    args = parser.parse_args(argv)

    from vlmeval.audit import verify_audit_pack

    summary = verify_audit_pack(args.audit_dir, args.leaderboard, args.readme)
    print(
        "audit verified: "
        f"{summary['models']} models, {summary['tasks']} tasks, "
        f"{summary['files']} files, {summary['rows']} rows; published claims consistent"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
