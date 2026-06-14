"""Measure the irony confound in charged trait axes.

The problem (observed live): a chatter who says edgy/charged things IRONICALLY
tops a charged axis (menace, and custom racism/misogyny axes), while a chatter
who means it sincerely may not. The embedder sees words, not intent: an ironic
"I hate <group>" embeds next to a sincere one. So a charged-axis score partly
measures "performs edginess as a bit," which is not the trait.

This script tests whether that confound is real and measurable, and whether
down-weighting a person's ironic messages changes the charged-axis ranking
toward what a human would say.

It uses the per-message clouds (data/unsynced/msg_index/*.npz) and the built-in
'menace' and 'ironic' trait axes. No new embeddings; just dot products.

    python scripts/irony_confound.py [--axis menace] [--highlight <name>]

Read it as a diagnostic, not a verdict: irony is ONE driver of the gap (axis
pole quality and the multilingual embedder are others).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from utils import persona_msg_index as pmi  # noqa: E402
from utils import persona_traits as pt  # noqa: E402


def _per_message_projections(axis_vecs, authors):
    """{author: {axis: array of per-message projections}} from the npz clouds."""
    out = {}
    for a in authors:
        try:
            V, _texts = pmi._load(a)
        except Exception:
            continue
        V = np.asarray(V, dtype="float32")
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        out[a] = {name: V @ np.asarray(av, dtype="float32") for name, av in axis_vecs.items()}
    return out


def _z(d):
    vals = np.array(list(d.values()))
    mu, sd = vals.mean(), vals.std() or 1.0
    return {k: (v - mu) / sd for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", default="menace", help="charged axis to test (built-in)")
    ap.add_argument("--highlight", default="",
                    help="roster name to call out in detail (optional)")
    ap.add_argument("--k", type=float, default=1.5, help="irony down-weight strength")
    args = ap.parse_args()

    ortho = pt.ortho_axis_vectors()
    if args.axis not in ortho:
        print(f"axis '{args.axis}' is not a built-in; choose from {list(ortho)}")
        return
    axis_vecs = {"charged": ortho[args.axis], "ironic": ortho["ironic"]}

    authors = sorted(a[:-4] for a in os.listdir(pmi.DIR) if a.endswith(".npz"))
    proj = _per_message_projections(axis_vecs, authors)
    authors = [a for a in authors if a in proj]

    # global irony stats for a comparable per-message down-weight
    all_ir = np.concatenate([proj[a]["ironic"] for a in authors])
    ir_mu, ir_sd = float(all_ir.mean()), float(all_ir.std()) or 1.0

    raw, disc, irony = {}, {}, {}
    for a in authors:
        m = proj[a]["charged"]
        ir = proj[a]["ironic"]
        raw[a] = float(m.mean())
        irony[a] = float(ir.mean())
        # weight: a message counts less toward the charged trait the more ironic
        # it reads. w in (0,1], = sigmoid(-k * irony_z_of_message).
        w = 1.0 / (1.0 + np.exp(args.k * (ir - ir_mu) / ir_sd))
        disc[a] = float((w * m).sum() / (w.sum() + 1e-9))

    rawz, discz, irz = _z(raw), _z(disc), _z(irony)

    # 1. is the confound real? correlation of irony with the charged axis
    A = np.array([irz[a] for a in authors])
    B = np.array([rawz[a] for a in authors])
    r = float(np.corrcoef(A, B)[0, 1])
    print(f"=== irony confound on the '{args.axis}' axis ({len(authors)} chatters) ===")
    print(f"corr(irony_score, {args.axis}_score) across roster = {r:+.3f}  "
          f"({'CONFOUND: ironic people score higher' if r > 0.2 else 'weak/none'})\n")

    # 2. ranking shift: raw vs irony-discounted
    raw_rank = sorted(authors, key=lambda a: -rawz[a])
    disc_rank = sorted(authors, key=lambda a: -discz[a])
    pos = {a: i for i, a in enumerate(disc_rank)}
    print(f"top {args.axis} (raw)         -> irony-discounted rank change")
    for i, a in enumerate(raw_rank[:10]):
        shift = i - pos[a]
        arrow = f"  (moves {'+' if shift>0 else ''}{shift})" if shift else "  (same)"
        mark = "  <<< HIGHLIGHT" if a == args.highlight else ""
        print(f"  {i+1:2}. {a:24} raw={rawz[a]:+.2f} disc={discz[a]:+.2f} iron={irz[a]:+.2f}{arrow}{mark}")

    # 3. the highlighted person's mechanism + the deeper diagnosis: can the
    #    irony axis even fire? If its dynamic range over this person's messages
    #    is tiny, there is nothing to discount and the confound is undetectable.
    h = args.highlight
    if h in proj:
        m = proj[h]["charged"]
        ir = proj[h]["ironic"]
        top_idx = m.argsort()[::-1][:50]
        frac_ironic = float((ir[top_idx] > ir_mu).mean())
        ir_range = float(np.percentile(ir, 95) - np.percentile(ir, 5))
        print(f"\n{h}: raw {args.axis} z={rawz[h]:+.2f} (rank {raw_rank.index(h)+1}) "
              f"-> discounted z={discz[h]:+.2f} (rank {disc_rank.index(h)+1})")
        print(f"  {frac_ironic*100:.0f}% of their top-50 most-{args.axis} messages read "
              f"above-average ironic (not a clear majority).")
        print(f"  ironic-axis dynamic range over their messages (p95-p5) = {ir_range:.3f} "
              f"— if this is near zero, the zero-shot irony axis cannot detect their")
        print(f"  irony at all, so irony-discounting has nothing to act on. The real fix")
        print(f"  is a SUPERVISED irony signal (see the irony oracle queue), not a discount.")


if __name__ == "__main__":
    main()
