"""Inspect built wheel and source-distribution contents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args(argv)

    from vlmeval.release import ReleaseError, verify_distribution

    archives = args.archives or sorted(Path("dist").glob("vlm_eval_bench-*"))
    if not archives:
        raise ReleaseError("no wheel or source distribution found")
    for archive in archives:
        summary = verify_distribution(archive)
        print(f"verified {archive}: {summary['files']} files, {summary['bytes']} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
