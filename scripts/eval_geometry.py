"""Geometry dial for the persona/embedding stack — the falsifiability harness.

Everything downstream (person similarity, trait axes, retrieval) reads cosine
geometry off a FROZEN BGE-M3 manifold. That manifold is anisotropic (a few
high-variance directions dominate every cosine) and the trait axes are constant
mean-difference steering vectors on it. Before changing any of that we need a
NUMBER that moves, otherwise "the embeddings feel mushy" stays a feeling.

This script computes three deterministic diagnostics from artifacts already on
disk (no bot, no rebuild — only the embedder for the axis section):

  1. ANISOTROPY of the person-vector matrix under three post-processings:
       raw            -> the embeddings as stored
       centered       -> current production (_centered(): subtract roster mean)
       abtt-k         -> centered THEN remove top-k principal components
                         (Mu & Viswanath, "All But The Top", ICLR 2018)
     Reported as mean off-diagonal cosine (signed + |.|) and the share of total
     variance held by the top-1 / top-5 singular directions. Lower |cos| and a
     flatter spectrum = more usable geometry.

  2. AXIS COLLINEARITY: the pairwise cosine matrix of the five trait axes
     (menace/ironic/unhinged/professor/doomer). If two "different" axes sit at
     cosine ~0.9 they are measuring one thing. This is the steering-vector
     pathology the owner's structured-transform-discovery repo documented
     (Exp 23-26: a constant mean-difference direction is a poor operator on a
     frozen encoder). Needs the embedder; cached to data/unsynced/eval/.

  3. ABTT SAFETY GUARD: how much of each axis direction lives inside the top-k
     person-space principal subspace that ABTT removes. If an axis is nearly
     orthogonal to the removed subspace (retained energy ~1.0), ABTT cannot
     blunt it. This is the "axis separability must not regress" check.

Plus a qualitative neighbor dump (top-3 semantic neighbors per author) under
centered vs abtt so a human can eyeball whether the generic-chatter direction
stopped dominating.

    python scripts/eval_geometry.py [--k 1] [--no-axes] [--neighbors 8]

Deterministic: no RNG, no network except the cached axis embeddings.
"""

import argparse
import json
import os
import pickle
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import config  # noqa: E402
from utils import persona_traits  # noqa: E402

EMB_PATH = os.path.join("data", "unsynced", "persona_embeddings.pkl")
AXIS_CACHE = os.path.join("data", "unsynced", "eval", "axis_vecs_cache.json")


# ---------------------------------------------------------------- loading -----

def load_person_matrix():
    """(names, M[N,1024] float64 L2-normalized) from the production pickle."""
    with open(EMB_PATH, "rb") as fh:
        data = pickle.load(fh)
    vectors = data["vectors"]
    names = list(vectors)
    M = np.vstack([np.asarray(vectors[a], dtype="float64") for a in names])
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return names, M, data.get("model", "?")


# ------------------------------------------------------------ transforms ------

def center(M):
    C = M - M.mean(axis=0)
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return C


def abtt(M, k):
    """Center, then project out the top-k principal components, then renorm.
    Returns (transformed_matrix, removed_PCs[k,1024]). k=0 is plain centering."""
    C = M - M.mean(axis=0)
    if k > 0:
        # principal directions of the centered cloud (rows are observations)
        _u, _s, vt = np.linalg.svd(C, full_matrices=False)
        pcs = vt[:k]                       # (k, d) orthonormal
        C = C - (C @ pcs.T) @ pcs          # remove their contribution
    else:
        pcs = np.zeros((0, M.shape[1]))
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return C, pcs


# ------------------------------------------------------------- metrics --------

def anisotropy(M):
    """mean signed off-diagonal cosine, mean |off-diagonal| cosine, and the
    top-1 / top-5 singular-value variance share of the (already-normalized)
    rows. M rows are assumed L2-normalized."""
    N = M.shape[0]
    S = M @ M.T
    off = S[~np.eye(N, dtype=bool)]
    s = np.linalg.svd(M, compute_uv=False)
    var = s ** 2
    var_share = var / var.sum()
    return {
        "mean_offdiag_cos": float(off.mean()),
        "mean_abs_offdiag_cos": float(np.abs(off).mean()),
        "top1_sv_share": float(var_share[0]),
        "top5_sv_share": float(var_share[:5].sum()),
    }


def _embed(texts):
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    body = json.dumps({"model": config.LLM_EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(base + "/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return [d["embedding"] for d in json.load(r)["data"]]


def axis_vectors(use_cache=True):
    """{axis: unit 1024-vec} for the five built-in axes, embedded via the live
    endpoint and cached so reruns are free. Mirrors persona_traits._axis_vectors
    exactly (mean(pos) - mean(neg), normalized)."""
    if use_cache and os.path.exists(AXIS_CACHE):
        cached = json.load(open(AXIS_CACHE, encoding="utf-8"))
        if cached.get("model") == config.LLM_EMBED_MODEL and \
           set(cached.get("axes", {})) == set(persona_traits.AXES):
            return {a: np.asarray(v, dtype="float64") for a, v in cached["axes"].items()}
    vecs = {}
    for name, (_neg, _pos, neg_s, pos_s) in persona_traits.AXES.items():
        embs = _embed(neg_s + pos_s)
        neg = np.asarray(embs[:len(neg_s)], dtype="float64").mean(axis=0)
        pos = np.asarray(embs[len(neg_s):], dtype="float64").mean(axis=0)
        v = pos - neg
        vecs[name] = v / (np.linalg.norm(v) + 1e-9)
    os.makedirs(os.path.dirname(AXIS_CACHE), exist_ok=True)
    json.dump({"model": config.LLM_EMBED_MODEL,
               "axes": {a: v.tolist() for a, v in vecs.items()}},
              open(AXIS_CACHE, "w", encoding="utf-8"))
    return vecs


def axis_collinearity(axes):
    names = list(axes)
    A = np.vstack([axes[a] for a in names])
    C = A @ A.T
    off = C[~np.eye(len(names), dtype=bool)]
    worst = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if worst is None or abs(C[i, j]) > abs(worst[2]):
                worst = (names[i], names[j], float(C[i, j]))
    return names, C, float(np.abs(off).mean()), worst


def axis_energy_retained(axes, removed_pcs):
    """For each axis a, 1 - ||proj of a onto removed subspace||^2. ~1.0 means
    ABTT does not touch what the axis reads; low means ABTT would blunt it."""
    out = {}
    for a, v in axes.items():
        if removed_pcs.shape[0] == 0:
            out[a] = 1.0
            continue
        proj = (v @ removed_pcs.T) @ removed_pcs
        out[a] = float(1.0 - (proj @ proj))
    return out


# --------------------------------------------------------------- report -------

def lowdin(A):
    """Symmetric (order-independent) orthogonalization of axis rows A[m,d]:
    A_orth = (A A^T)^{-1/2} A. Decorrelates the axes without privileging any
    one (unlike Gram-Schmidt, whose result depends on input order)."""
    G = A @ A.T
    w, U = np.linalg.eigh(G)
    w = np.clip(w, 1e-9, None)
    inv_sqrt = U @ np.diag(w ** -0.5) @ U.T
    R = inv_sqrt @ A
    R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    return R


def gram_schmidt(A):
    """Order-dependent orthogonalization (what persona_axes._ortho_builtin does
    today): each row is stripped of all earlier rows' components, in input
    order. Included only to measure what the production ~top path already gets."""
    basis = []
    out = []
    for v in A:
        v = v.astype("float64").copy()
        for b in basis:
            v -= (v @ b) * b
        n = np.linalg.norm(v)
        if n > 1e-9:
            v /= n
        out.append(v)
        basis.append(v)
    return np.vstack(out)


def axis_score_entanglement(M, axes):
    """The thing the user EXPERIENCES as 'axes feel the same': do the per-person
    axis z-scores correlate across the roster? Compares axis geometries:
      current   - raw _axis_vectors (production traits_for / ~traits path)
      gram      - Gram-Schmidt, fixed order (production axis_scores / ~top path)
      lowdin    - symmetric-orthogonalized axes (order-independent, proposed)
      whitened  - ZCA-whiten the score covariance directly (max decorrelation)
    Returns {label: (names, corr_matrix, mean_abs_offdiag)} on one person space."""
    anames = list(axes)
    A = np.vstack([axes[a] for a in anames])

    def scores_for(axis_mat):
        Z = M @ axis_mat.T                         # (people, axes) projections
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)    # z-score per axis
        return Z

    def corr(Z):
        C = np.corrcoef(Z.T)
        off = C[~np.eye(C.shape[0], dtype=bool)]
        return C, float(np.abs(off).mean())

    out = {}
    Zc = scores_for(A)
    out["current"] = (anames,) + corr(Zc)
    out["gram"] = (anames,) + corr(scores_for(gram_schmidt(A)))
    out["lowdin"] = (anames,) + corr(scores_for(lowdin(A)))
    # whitened: decorrelate the score covariance itself (Mahalanobis on scores)
    Cz = np.cov(Zc.T)
    w, U = np.linalg.eigh(Cz)
    W = U @ np.diag(np.clip(w, 1e-9, None) ** -0.5) @ U.T
    out["whitened"] = (anames,) + corr(Zc @ W)
    return out


def neighbors_table(names, M, n=8, top=3):
    S = M @ M.T
    np.fill_diagonal(S, -1.0)
    lines = []
    for i in range(min(n, len(names))):
        order = S[i].argsort()[::-1][:top]
        nn = " | ".join(f"{names[j]} {S[i, j]:+.2f}" for j in order)
        lines.append(f"  {names[i]:24} {nn}")
    return "\n".join(lines)


def fmt(d):
    return (f"offdiag_cos={d['mean_offdiag_cos']:+.3f}  "
            f"|offdiag|={d['mean_abs_offdiag_cos']:.3f}  "
            f"top1_sv={d['top1_sv_share']:.3f}  top5_sv={d['top5_sv_share']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=2, help="ABTT components to remove for the neighbor dump (production ABTT_K=2)")
    ap.add_argument("--no-axes", action="store_true", help="skip the axis section (no embedder call)")
    ap.add_argument("--neighbors", type=int, default=8, help="how many authors in the neighbor dump")
    args = ap.parse_args()

    names, M, model = load_person_matrix()
    print(f"person vectors: {len(names)}  dim={M.shape[1]}  model={model}\n")

    print("=== 1. ANISOTROPY (person-vector matrix) ===")
    print(f"  raw        {fmt(anisotropy(M))}")
    Cc = center(M)
    print(f"  centered   {fmt(anisotropy(Cc))}   <- current production")
    removed_for_k = None
    for k in (1, 2, 3):
        Ck, pcs = abtt(M, k)
        if k == args.k:
            removed_for_k = pcs
        print(f"  abtt-{k}     {fmt(anisotropy(Ck))}")
    print()

    if not args.no_axes:
        print("=== 2. AXIS COLLINEARITY (trait axes; verifies the steering-vector pathology) ===")
        try:
            axes = axis_vectors()
        except Exception as e:
            print(f"  (embedder unavailable: {type(e).__name__} {e}) — rerun with embedder up\n")
            axes = None
        if axes:
            anames, C, mean_abs, worst = axis_collinearity(axes)
            hdr = "        " + " ".join(f"{a[:6]:>7}" for a in anames)
            print(hdr)
            for i, a in enumerate(anames):
                row = " ".join(f"{C[i, j]:+.2f}".rjust(7) for j in range(len(anames)))
                print(f"  {a[:6]:>6} {row}")
            print(f"  mean |off-diagonal| axis cosine = {mean_abs:.3f}")
            print(f"  most collinear pair: {worst[0]} ~ {worst[1]} = {worst[2]:+.2f}")
            print()

            print("=== 3. ABTT SAFETY GUARD (axis energy retained after removing top-"
                  f"{args.k}) ===")
            retained = axis_energy_retained(axes, removed_for_k)
            for a, r in sorted(retained.items(), key=lambda kv: kv[1]):
                flag = "  <-- WATCH" if r < 0.85 else ""
                print(f"  {a:12} retained={r:.3f}{flag}")
            print()

            print("=== 3b. AXIS-SCORE ENTANGLEMENT (do per-person scores correlate "
                  "across the roster?) ===")
            ent = axis_score_entanglement(Cc, axes)
            for label in ("current", "gram", "lowdin", "whitened"):
                _an, _C, mean_abs = ent[label]
                print(f"  {label:9} mean |off-diagonal| score correlation = {mean_abs:.3f}")
            print("  current=~traits (raw)  gram=~top (Gram-Schmidt, order-dependent)  "
                  "lowdin=proposed (order-independent)")
            print()

    print(f"=== 4. NEIGHBOR SANITY (centered vs abtt-{args.k}) ===")
    print("  -- centered (current) --")
    print(neighbors_table(names, Cc, n=args.neighbors))
    Ck, _ = abtt(M, args.k)
    print(f"  -- abtt-{args.k} --")
    print(neighbors_table(names, Ck, n=args.neighbors))


if __name__ == "__main__":
    main()
