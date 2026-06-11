"""Preview a user's Markov persona in the terminal (nothing is sent to chat).

    python scripts/persona_preview.py <user> [n] [--order 2]

Generates n fake messages from the user's archived history so you can see how
recognizable the persona is before any LLM money or chat-posting is involved.
This script ONLY prints — it never connects to Twitch.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import persona_markov  # noqa: E402


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("user")
    ap.add_argument("n", nargs="?", type=int, default=8)
    ap.add_argument("--order", type=int, default=2,
                    help="context words (2 = looser/funnier, 3 = closer to real)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    info, lines = persona_markov.sample(
        args.user, n=args.n, order=args.order, seed=args.seed
    )
    if not lines:
        print(f"Not enough archived messages for '{args.user}' to build a persona.")
        return
    print(f"--- {args.user}-bot (Markov order-{args.order}, "
          f"from {info['source_messages']:,} real messages) ---")
    for line in lines:
        print(f"  {line}")
    print("\n(These are recombined from real messages — preview only, not posted.)")


if __name__ == "__main__":
    main()
