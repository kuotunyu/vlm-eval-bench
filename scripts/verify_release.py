"""Verify that the repository is safe to export as a public release."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    from vlmeval.release import verify_release_tree

    summary = verify_release_tree(args.root)
    print(f"release tree verified: {summary['files']} public files, {summary['bytes']} total bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
