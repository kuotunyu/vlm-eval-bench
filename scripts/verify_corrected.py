"""Verify corrected rows against archived evidence and public claims."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected-dir", type=Path, default=Path("results/corrected"))
    parser.add_argument("--archived-audit-dir", type=Path, default=Path("results/audit"))
    parser.add_argument("--leaderboard", type=Path, default=Path("results/leaderboard.md"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args(argv)

    from vlmeval.private_rescore import verify_corrected_pack

    summary = verify_corrected_pack(
        args.corrected_dir,
        args.archived_audit_dir,
        args.leaderboard,
        args.readme,
    )
    print(
        "corrected evidence verified: "
        f"{summary['models']} models, {summary['tasks']} tasks, "
        f"{summary['files']} files, {summary['rows']} rows; public claims consistent"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
