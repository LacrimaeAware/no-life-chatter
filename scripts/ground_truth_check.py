"""Re-verify the numbers recorded in docs/GROUND_TRUTH.md against live state.

This is the anti-stale-number mechanism. docs/GROUND_TRUTH.md quotes specific
measurements (person-vector anisotropy, trait-axis entanglement, etc.). Numbers
in prose rot silently — a future reader trusts "menace~doomer = 0.91" long after
it became 0.64. This script recomputes each recorded number from the live
artifacts/code and flags drift, so the ground truth stays falsifiable.

    python scripts/ground_truth_check.py            # human-readable
    python scripts/ground_truth_check.py --json      # machine-readable

Exit code is non-zero if any metric DRIFTED beyond tolerance, so it can gate a
freshness check or a pre-commit hook. Metrics that need the embedder (the axis
geometry) SKIP cleanly when LM Studio is down — they do not fail the run.

IMPORTANT: a DRIFT is not always a bug. If you deliberately rebuilt the person
embeddings on new data, the geometry legitimately moves. The correct response is
to re-verify the new values are sane and then UPDATE both EXPECTED below and the
matching numbers in docs/GROUND_TRUTH.md in the same commit. That update IS the
freshness loop; never silence a drift by widening a tolerance without looking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (value, tolerance, needs_embedder, how-to-reverify). Update these in the SAME
# commit as any change that legitimately moves them, and mirror docs/GROUND_TRUTH.md.
EXPECTED = {
    "abtt_k": (2, 0, False,
               "utils/persona_embeddings.py ABTT_K constant"),
    "person_count": (34, None, False,
                     "count of vectors in data/unsynced/persona_embeddings.pkl (grows with rebuilds; informational)"),
    "person_anisotropy": (0.149, 0.03, False,
                          "mean |off-diagonal cosine| of production persona_embeddings._centered() (centered + ABTT-2)"),
    "axis_entanglement": (0.249, 0.05, True,
                          "mean |off-diagonal| correlation of per-person trait z-scores via persona_traits.traits_for"),
    "menace_doomer_axis_cos": (0.64, 0.06, True,
                               "cosine of the raw menace and doomer axis vectors (the number once mis-stated as 0.91)"),
    "doomer_lowdin_alignment": (0.917, 0.04, True,
                                "cosine of the raw doomer axis with its Löwdin-orthogonalized version (labels stay valid)"),
}


def _embedder_up() -> bool:
    import urllib.request
    import config
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=4):
            return bool(config.LLM_EMBED_MODEL)
    except Exception:
        return False


def measure() -> dict:
    """{metric: value or None}. None means could-not-measure (e.g. embedder down)."""
    import numpy as np
    from utils import persona_embeddings as pe

    out = {}
    out["abtt_k"] = pe.ABTT_K

    # offline person-vector geometry (production transform, includes ABTT)
    try:
        pe._CENTERED = None
        centered = pe._centered()
        names = list(centered)
        out["person_count"] = len(names)
        M = np.vstack([centered[a] for a in names])
        S = M @ M.T
        off = S[~np.eye(len(names), dtype=bool)]
        out["person_anisotropy"] = round(float(np.abs(off).mean()), 3)
    except Exception as e:
        out["person_count"] = None
        out["person_anisotropy"] = None
        out["_person_error"] = str(e)

    # embedder-dependent axis geometry
    if not _embedder_up():
        out["axis_entanglement"] = None
        out["menace_doomer_axis_cos"] = None
        out["doomer_lowdin_alignment"] = None
        out["_axes_skipped"] = "embedder unavailable"
        return out

    try:
        from utils import persona_traits as pt
        pt._AXIS_VECS = None
        pt._ORTHO_VECS = None
        raw = pt._axis_vectors()
        m = np.asarray(raw["menace"], dtype="float64")
        d = np.asarray(raw["doomer"], dtype="float64")
        out["menace_doomer_axis_cos"] = round(float(m @ d), 2)

        ortho = pt.ortho_axis_vectors()
        d_o = np.asarray(ortho["doomer"], dtype="float64")
        out["doomer_lowdin_alignment"] = round(float(d @ d_o), 3)

        centered = pe._centered()
        roster = list(centered)
        axes = list(pt.AXES)
        Z = np.array([[dict(pt.traits_for(a))[ax] for ax in axes] for a in roster])
        C = np.corrcoef(Z.T)
        offd = C[~np.eye(len(axes), dtype=bool)]
        out["axis_entanglement"] = round(float(np.abs(offd).mean()), 3)
    except Exception as e:
        out["axis_entanglement"] = None
        out["menace_doomer_axis_cos"] = None
        out["doomer_lowdin_alignment"] = None
        out["_axes_error"] = str(e)
    return out


def evaluate(values: dict):
    rows = []
    drift = False
    for metric, (exp, tol, _needs, how) in EXPECTED.items():
        got = values.get(metric)
        if got is None:
            rows.append(("SKIP", metric, exp, got, how))
            continue
        if tol is None:                     # informational only (e.g. counts)
            status = "INFO" if got == exp else "CHANGED"
            rows.append((status, metric, exp, got, how))
            continue
        if abs(got - exp) <= tol:
            rows.append(("PASS", metric, exp, got, how))
        else:
            rows.append(("DRIFT", metric, exp, got, how))
            drift = True
    return rows, drift


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    values = measure()
    rows, drift = evaluate(values)

    if args.json:
        print(json.dumps({"values": values,
                          "results": [{"status": s, "metric": m, "expected": e, "got": g}
                                      for s, m, e, g, _ in rows],
                          "drift": drift}, indent=2))
        return 1 if drift else 0

    for status, metric, exp, got, how in rows:
        line = f"[{status:7}] {metric:24} expected={exp}  got={got}"
        print(line)
        if status in ("DRIFT", "CHANGED"):
            print(f"           re-verify: {how}")
    if values.get("_axes_skipped"):
        print(f"\nnote: axis metrics skipped ({values['_axes_skipped']}) — "
              "rerun with LM Studio up to check them.")
    print("\nDRIFT means a recorded number no longer matches live state. If the change "
          "was intentional,\nupdate EXPECTED in this script AND docs/GROUND_TRUTH.md in the same commit.")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
