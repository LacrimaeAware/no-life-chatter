"""Rotate known local logs before the bot appends to them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.log_rotation import rotate_file


DEFAULT_LOGS = (
    ("data/bot.log", 10 * 1024 * 1024, 5),
    ("data/unsynced/persona_logs.jsonl", 25 * 1024 * 1024, 5),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rotate NoLifeChatter local logs by size.")
    parser.add_argument("--quiet", action="store_true", help="only print errors")
    parser.add_argument("--max-mb", type=float, default=None,
                        help="override max size for every configured log")
    parser.add_argument("--keep", type=int, default=None,
                        help="override rollover count for every configured log")
    args = parser.parse_args()

    for path, default_bytes, default_keep in DEFAULT_LOGS:
        max_bytes = int(args.max_mb * 1024 * 1024) if args.max_mb else default_bytes
        keep = args.keep if args.keep is not None else default_keep
        rotated = rotate_file(path, max_bytes=max_bytes, keep=keep)
        if rotated and not args.quiet:
            print(f"rotated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
