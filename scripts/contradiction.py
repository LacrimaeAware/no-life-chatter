"""Self-contradiction as a performativity / irony signal — no oracle needed.

The insight (user, 2026-06-14): you cannot detect irony from a single message,
but a *deeply ironic / performative* person is detectable from their DATA — they
hold contradictory traits. On an axis like menace<->wholesome a sincere person
clusters at one pole; a person who shitposts edgily AND is wholesome appears at
BOTH ends. "High in both feminism and misogyny" is impossible as one axis value
but ordinary in one person's messages.

This is the distributional person model (keep the per-message cloud instead of
mean-pooling it) applied to irony: contradiction = mass at both poles at once.
Mean-pooling destroys exactly this signal — a bimodal person averages to a bland
midpoint, which is precisely why their charged-axis score misleads.

    python scripts/contradiction.py [--axis menace] [--highlight <name>]

Metrics per person on the chosen axis (using global per-message deciles):
  frac_hi  = fraction of their messages above the global 90th percentile pole
  frac_lo  = fraction below the global 10th percentile pole
  contra   = sqrt(frac_hi * frac_lo)  -> high ONLY if they live at both ends
  spread   = within-person std of the projection (a coarser version)
Ranked by contra. A high-contra person's charged-axis mean is unreliable.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from utils import persona_msg_index as pmi  # noqa: E402
from utils import persona_traits as pt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", default="menace")
    ap.add_argument("--highlight", default="")
    ap.add_argument("--show-top", type=int, default=0,
                    help="print both-pole evidence for the top N contradictory people")
    args = ap.parse_args()

    ortho = pt.ortho_axis_vectors()
    if args.axis not in ortho:
        print(f"axis '{args.axis}' not built-in; choose {list(ortho)}")
        return
    av = np.asarray(ortho[args.axis], dtype="float32")

    authors = sorted(a[:-4] for a in os.listdir(pmi.DIR) if a.endswith(".npz"))
    proj = {}
    for a in authors:
        try:
            V, texts = pmi._load(a)
        except Exception:
            continue
        V = np.asarray(V, dtype="float32"); V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        proj[a] = (V @ av, [str(t) for t in texts])
    authors = [a for a in authors if a in proj]

    allp = np.concatenate([proj[a][0] for a in authors])
    hi, lo = np.percentile(allp, 90), np.percentile(allp, 10)

    rows = {}
    for a in authors:
        p = proj[a][0]
        fh = float((p > hi).mean())
        fl = float((p < lo).mean())
        rows[a] = {"hi": fh, "lo": fl, "contra": float(np.sqrt(fh * fl)),
                   "spread": float(p.std()), "mean": float(p.mean())}

    contra = {a: rows[a]["contra"] for a in authors}
    cz_mu = np.mean(list(contra.values())); cz_sd = np.std(list(contra.values())) or 1.0
    order = sorted(authors, key=lambda a: -contra[a])

    print(f"=== self-contradiction on the '{args.axis}' axis "
          f"(both-poles-at-once; {len(authors)} chatters) ===")
    print(f"{'rank':>4}  {'author':24} {'frac_hi':>7} {'frac_lo':>7} {'contra':>7} {'contra_z':>8}")
    for i, a in enumerate(order):
        z = (contra[a] - cz_mu) / cz_sd
        mark = "  <<< HIGHLIGHT" if a == args.highlight else ""
        if i < 12 or a == args.highlight:
            print(f"{i+1:>4}. {a:24} {rows[a]['hi']:7.3f} {rows[a]['lo']:7.3f} "
                  f"{rows[a]['contra']:7.3f} {z:+8.2f}{mark}")

    # evidence for the top-N contradictory people, so a human can judge whether
    # the signal catches the actually-performative chatters
    def _pole_examples(a, n=2):
        p, texts = proj[a]
        hi_i = p.argsort()[::-1][:n]
        lo_i = p.argsort()[:n]
        return ([(float(p[i]), texts[i]) for i in hi_i],
                [(float(p[i]), texts[i]) for i in lo_i])

    if args.show_top:
        print(f"\n--- both-pole evidence for the top {args.show_top} ({args.axis}) ---")
        for a in order[:args.show_top]:
            hiex, loex = _pole_examples(a, 2)
            print(f"\n{a}  (contra_z={(contra[a]-cz_mu)/cz_sd:+.2f})")
            for s, t in hiex:
                print(f"    +{args.axis[:4]} {s:+.2f}  {t[:70]!r}")
            for s, t in loex:
                print(f"    -{args.axis[:4]} {s:+.2f}  {t[:70]!r}")

    # mechanism for the highlighted person: show they live at BOTH ends
    h = args.highlight
    if h in proj:
        p, texts = proj[h]
        print(f"\n{h}: contra rank {order.index(h)+1}/{len(authors)}, "
              f"mean {args.axis} proj={rows[h]['mean']:+.3f} "
              f"(the bland midpoint mean-pooling would keep)")
        hi_i = p.argsort()[::-1][:3]
        lo_i = p.argsort()[:3]
        print(f"  their most-{args.axis} lines:")
        for i in hi_i:
            print(f"    {p[i]:+.2f}  {texts[i][:74]!r}")
        print(f"  their most-opposite lines (same person):")
        for i in lo_i:
            print(f"    {p[i]:+.2f}  {texts[i][:74]!r}")
        print("\n  -> if both lists look like the same person performing both poles,")
        print("     the mean is a lie and 'contra' is the honest read of their charged axis.")


if __name__ == "__main__":
    main()
