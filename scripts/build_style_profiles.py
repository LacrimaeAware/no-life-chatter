"""Rebuild lexical voice profiles for ~markers, ~like, and ~twin.

This updates the profile/prevalence sections inside the classifier pickle after
the authorship classifier has been trained.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-messages", type=int, default=2000,
                        help="minimum archive messages for a roster candidate")
    parser.add_argument("--max-roster", type=int, default=80,
                        help="maximum number of non-bot roster authors")
    parser.add_argument("--words-top", type=int, default=300,
                        help="words kept per author")
    parser.add_argument("--phrases-top", type=int, default=150,
                        help="word-pairs kept per author")
    parser.add_argument("--bg-cap", type=int, default=120000,
                        help="background messages sampled for log-odds")
    args = parser.parse_args()

    from utils import persona_classifier  # noqa: WPS433

    count = persona_classifier.build_style_profiles(
        words_top=args.words_top,
        phrases_top=args.phrases_top,
        bg_cap=args.bg_cap,
        min_messages=args.min_messages,
        max_roster=args.max_roster,
    )
    print(json.dumps({
        "profiles": count,
        "min_messages": args.min_messages,
        "max_roster": args.max_roster,
        "words_top": args.words_top,
        "phrases_top": args.phrases_top,
        "bg_cap": args.bg_cap,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
