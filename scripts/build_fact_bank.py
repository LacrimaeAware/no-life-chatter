"""Build a private evidence-backed fact/claim bank from the chat archive.

The output is gitignored JSONL under data/unsynced. Rows are candidate claims
with evidence, not verified truths.

Examples:

    python scripts/build_fact_bank.py --max-authors 40
    python scripts/build_fact_bank.py --authors user1,user2 --max-utterances 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import fact_bank  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authors", default="",
                        help="comma-separated authors; default = top archive authors")
    parser.add_argument("--max-authors", type=int, default=40)
    parser.add_argument("--min-messages", type=int, default=500)
    parser.add_argument(
        "--max-utterances", type=int, default=20000,
        help="message-local rows scanned per author; facts intentionally use deeper history",
    )
    parser.add_argument("--evidence-limit", type=int, default=5)
    parser.add_argument("--out", default=str(fact_bank.DEFAULT_OUT))
    args = parser.parse_args()

    authors = [a.strip() for a in args.authors.split(",") if a.strip()] or None
    rows = fact_bank.build_fact_bank(
        authors,
        max_authors=args.max_authors,
        min_messages=args.min_messages,
        max_utterances=args.max_utterances,
        evidence_limit=args.evidence_limit,
    )
    out = Path(args.out)
    fact_bank.write_jsonl(rows, out, metadata={
        "max_authors": args.max_authors,
        "min_messages": args.min_messages,
        "max_utterances": args.max_utterances,
        "evidence_limit": args.evidence_limit,
    })
    by_author = {}
    for row in rows:
        by_author[row["author"]] = by_author.get(row["author"], 0) + 1
    print(json.dumps({
        "out": str(out),
        "claims": len(rows),
        "authors": len(by_author),
        "top_authors": sorted(by_author.items(), key=lambda kv: -kv[1])[:10],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
