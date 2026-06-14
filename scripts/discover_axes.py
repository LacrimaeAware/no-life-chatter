"""Discover personality axes from the data, unnamed — then a human names them.

The right direction (user, repeatedly): don't project people onto five
hand-picked axes ("you chose those randomly"). Instead find the orthogonal
directions the data ACTUALLY varies along, show the people and messages at each
extreme, and name them afterwards.

This runs PCA (orthogonal) or ICA (independent, often more interpretable) over
the roster's person vectors — the same centered + ABTT space production uses —
and for each discovered component prints:
  - the people at the + and - extremes,
  - the real messages most aligned with each end (what the axis "is"),
  - the closest hand-picked axis + cosine (does it re-discover menace/doomer, or
    is it something new we have no name for?).

    python scripts/discover_axes.py [--method pca|ica] [--k 6] [--msgs 6]

Read it as: here are N axes the data insists exist; you tell me what they are.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from utils import persona_embeddings as pe  # noqa: E402
from utils import persona_msg_index as pmi  # noqa: E402
from utils import persona_traits as pt  # noqa: E402


def _clean(t):
    return " ".join(str(t).split())[:76]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("pca", "ica"), default="pca")
    ap.add_argument("--k", type=int, default=6, help="how many axes to surface")
    ap.add_argument("--msgs", type=int, default=6, help="example messages per pole")
    ap.add_argument("--per-author", type=int, default=250)
    args = ap.parse_args()

    centered = pe._centered()                 # {author: centered+ABTT vec}
    names = list(centered)
    M = np.vstack([centered[a] for a in names]).astype("float64")  # (P, D)

    if args.method == "ica":
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=args.k, random_state=7, max_iter=1000, whiten="unit-variance")
        S = ica.fit_transform(M)              # (P, k) person loadings
        comps = ica.components_               # (k, D) — not orthonormal
        dirs = comps / (np.linalg.norm(comps, axis=1, keepdims=True) + 1e-9)
        person_load = S
    else:
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        dirs = Vt[:args.k]                     # (k, D) orthonormal
        person_load = (M @ dirs.T)             # (P, k)
        var = (s ** 2) / (s ** 2).sum()

    # message pool for showing what each axis "is"
    msg_vecs, msg_text, msg_auth = [], [], []
    for a in names:
        try:
            V, texts = pmi._load(a)
        except Exception:
            continue
        V = np.asarray(V, dtype="float64"); V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        idx = np.linspace(0, len(V) - 1, min(args.per_author, len(V))).astype(int)
        for i in idx:
            msg_vecs.append(V[i]); msg_text.append(texts[i]); msg_auth.append(a)
    MV = np.vstack(msg_vecs)
    MV -= MV.mean(axis=0)                       # remove the common "is chat" component

    named = pt.ortho_axis_vectors()
    named_names = list(named)
    NA = np.vstack([named[n] for n in named_names])

    print(f"=== {args.method.upper()}: {args.k} discovered axes over {len(names)} people ===\n")
    for k in range(args.k):
        d = dirs[k]
        # nearest hand-picked axis (sign-agnostic)
        cos = NA @ d
        j = int(np.argmax(np.abs(cos)))
        match = f"{named_names[j]} (cos {cos[j]:+.2f})"
        head = f"--- AXIS {k+1} ---"
        if args.method == "pca":
            head += f"   variance {var[k]*100:.0f}%"
        head += f"   closest named: {match}"
        print(head)

        order = np.argsort(person_load[:, k])
        neg_people = [names[i] for i in order[:4]]
        pos_people = [names[i] for i in order[::-1][:4]]
        print(f"  + people: {', '.join(pos_people)}")
        print(f"  - people: {', '.join(neg_people)}")

        mp = MV @ d
        mo = np.argsort(mp)
        print(f"  + messages:")
        for i in mo[::-1][:args.msgs]:
            print(f"      {_clean(msg_text[i])!r}  ({msg_auth[i]})")
        print(f"  - messages:")
        for i in mo[:args.msgs]:
            print(f"      {_clean(msg_text[i])!r}  ({msg_auth[i]})")
        print()


if __name__ == "__main__":
    main()
