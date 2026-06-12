"""Smoke-check semantic trait axes after rebuilding person vectors."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3,
                        help="number of leaderboard rows to print per pole")
    args = parser.parse_args()

    from utils import persona_embeddings  # noqa: WPS433
    from utils.persona_traits import leaderboard, pole_map  # noqa: WPS433

    if not persona_embeddings.available():
        print("No semantic vectors found. Run scripts/build_persona_embeddings.py first.")
        return 1

    ok = True
    for pole in sorted(pole_map()):
        rows = leaderboard(pole, args.n)
        if not rows:
            print(f"{pole}: no rows")
            ok = False
            continue
        rendered = " | ".join(f"{author} {score:+.1f}" for author, score in rows)
        print(f"{pole}: {rendered}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
