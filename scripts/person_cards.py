"""Per-person cards for an interactive labeling/validation pass.

For each roster chatter: the current trait readout + contradiction flag, and a
spread of their REAL messages (most-edgy, most-wholesome, and a random sample)
so a human has context to judge irony / sincerity / performativity instead of
recalling from memory. Local diagnostic — prints real handles/messages, so the
output is for the owner's eyes only and must NOT be pasted into tracked docs.

    python scripts/person_cards.py                 # all, roster order
    python scripts/person_cards.py name1 name2 ...  # just these
    python scripts/person_cards.py --skip a,b,c     # all except these
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from utils import persona_msg_index as pmi  # noqa: E402
from utils import persona_traits as pt  # noqa: E402


def _clean(t: str) -> str:
    return " ".join(str(t).split())[:120]


def card(name, menace_axis, contra, rng):
    try:
        V, texts = pmi._load(name)
    except Exception:
        return f"=== {name} ===\n  (no message index)\n"
    V = np.asarray(V, dtype="float32"); V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    m = V @ menace_axis
    n = len(texts)

    # trait readout
    try:
        readout = pt.traits_for(name)
    except Exception:
        readout = []
    rparts = []
    for axis, z in readout:
        neg, pos = pt.AXES[axis][0], pt.AXES[axis][1]
        rparts.append(f"{(pos if z>=0 else neg)} {abs(z):.1f}")
    flag = f"  [contradiction(menace) z={contra:+.2f}{' ⚡' if contra and contra>1.0 else ''}]" if contra is not None else ""

    # a BIG mixed sample so a human has real context: some of their most
    # negative-leaning, some most positive, and a broad random draw — shuffled
    # together as a flat list (no interpretive labels; just what they say).
    neg = list(m.argsort()[::-1][:5])
    pos = list(m.argsort()[:5])
    used = set(neg) | set(pos)
    pool = [i for i in range(n) if i not in used]
    rnd = rng.sample(pool, min(15, len(pool)))
    idxs = neg + pos + rnd
    rng.shuffle(idxs)

    out = [f"=== {name} ===  ({n} msgs)  readout: " + " · ".join(rparts),
           "  " + (flag.strip() if flag.strip() else "")]
    out += [f"   - {_clean(texts[i])}" for i in idxs]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()

    roster = sorted(a[:-4] for a in os.listdir(pmi.DIR) if a.endswith(".npz"))
    names = args.names or roster
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    names = [n for n in names if n not in skip]

    menace_axis = np.asarray(pt.ortho_axis_vectors()["menace"], dtype="float32")
    contra = pmi.contradiction_scores("menace")
    rng = random.Random(7)

    for nm in names:
        print(card(nm, menace_axis, contra.get(nm), rng))
        print()


if __name__ == "__main__":
    main()
