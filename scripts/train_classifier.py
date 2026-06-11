"""Train the authorship classifier (powers ~whosaid and persona eval).

    python scripts/train_classifier.py [--authors a,b,c] [--max-authors 24]
                                       [--per-author 4000]

Reads the local archive, trains a char-ngram + word Naive Bayes, saves it to
config.CLASSIFIER_FILE (gitignored), and prints held-out top-1 accuracy so you
can see how separable the chatters actually are. Nothing is sent anywhere.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import persona_classifier  # noqa: E402


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--authors", default="", help="comma-separated; default = top non-bot authors")
    ap.add_argument("--max-authors", type=int, default=24)
    ap.add_argument("--per-author", type=int, default=4000)
    ap.add_argument("--min-messages", type=int, default=300)
    ap.add_argument("--vocab-size", type=int, default=20000)
    args = ap.parse_args()

    authors = [a.strip() for a in args.authors.split(",") if a.strip()] or None
    report = persona_classifier.train(
        authors=authors, max_authors=args.max_authors, per_author=args.per_author,
        min_messages=args.min_messages, vocab_size=args.vocab_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
