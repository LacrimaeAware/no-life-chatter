"""Evidence-backed archive/lore query CLI.

Examples:

    python scripts/ask_chat.py --author someuser twin primes
    python scripts/ask_chat.py --channel somechannel "what did people say about Claude?"
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import archive_qa  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*")
    parser.add_argument("--author", "--user", default="")
    parser.add_argument("--channel", "--chat", default="")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    if not query:
        parser.error("query is required")
    report = archive_qa.build_report(
        query,
        author=args.author or None,
        channel=args.channel or None,
        limit=args.limit,
    )
    print(archive_qa.format_cli(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
