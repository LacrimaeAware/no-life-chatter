"""Query the private fact/claim bank built by scripts/build_fact_bank.py."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import fact_bank  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="terms to search in claim text")
    parser.add_argument("--author", default="")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--path", default=str(fact_bank.DEFAULT_OUT))
    args = parser.parse_args()

    rows = fact_bank.load_jsonl(Path(args.path))
    if not rows:
        print(f"No fact bank at {args.path}. Run scripts/build_fact_bank.py first.")
        return 1

    query = " ".join(args.query)
    for row in fact_bank.search(rows, author=args.author or None,
                                query=query, limit=args.limit):
        ev = row.get("evidence") or []
        first = ev[0] if ev else {}
        print(
            f"{row['author']} | {row['kind']} | support={row['support_count']} "
            f"conf={row['confidence']}: {row['claim']}"
        )
        if first:
            text = first.get("clean_text") or first.get("text")
            print(f"  {first.get('sent_at')} #{first.get('channel')}: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
