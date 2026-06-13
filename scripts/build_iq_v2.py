"""Build the v2 `~iq` cache.

This is intentionally offline. The chat command only reads the cache when it is
fresh; this script is where embedding and optional judge work should happen.

    python scripts/build_iq_v2.py --force
    python scripts/build_iq_v2.py --force --judge
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import persona_iq  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if a v2 cache already exists")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="skip embedding axes and build lexical/syntax only")
    parser.add_argument("--judge", action="store_true",
                        help="use the local chat model to judge top utterances")
    parser.add_argument("--max-utterances", type=int,
                        default=persona_iq.DEFAULT_MAX_UTTERANCES,
                        help="max utterances embedded per author")
    parser.add_argument("--min-utterances", type=int,
                        default=persona_iq.DEFAULT_MIN_UTTERANCES)
    parser.add_argument("--author-cap", type=int,
                        default=persona_iq.DEFAULT_AUTHOR_CAP,
                        help="max merged utterances used for lexical features")
    parser.add_argument("--sample-word-freq", type=int,
                        default=persona_iq.DEFAULT_WORD_FREQ_SAMPLE)
    parser.add_argument("--max-authors", type=int, default=None)
    parser.add_argument("--judge-items", type=int, default=8)
    parser.add_argument("--show", type=int, default=5,
                        help="print top and bottom rows after building")
    args = parser.parse_args()

    scores = persona_iq.compute_all(
        force=args.force,
        use_embeddings=not args.no_embeddings,
        use_llm=args.judge,
        max_utterances=args.max_utterances,
        min_utterances=args.min_utterances,
        author_cap=args.author_cap,
        sample_word_freq=args.sample_word_freq,
        max_authors=args.max_authors,
        judge_items=args.judge_items,
    )
    meta = persona_iq.cache_info()
    print(
        f"built {len(scores)} v2 text-IQ rows -> {persona_iq.CACHE}\n"
        f"meta: {meta}",
        flush=True,
    )
    if args.show and scores:
        print("\ntop:")
        for author, row in persona_iq.leaderboard(args.show):
            print(f"  {author:24} {row['iq']:3d} pct={row.get('percentile')} conf={row.get('confidence')}")
        print("\nbottom:")
        for author, row in persona_iq.leaderboard(args.show, reverse=True):
            print(f"  {author:24} {row['iq']:3d} pct={row.get('percentile')} conf={row.get('confidence')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
