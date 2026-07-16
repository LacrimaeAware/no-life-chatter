"""Semantic person-vectors (built offline by scripts/build_persona_embeddings.py).

The lexical layer (~markers/~like) sees which exact words someone overuses;
this layer sees what they talk ABOUT — two chatters with zero shared
catchphrases can still be semantic twins. Vectors are mean-pooled local
embeddings of each person's messages, stored in a gitignored pickle.

Raw mean-pooled chat embeddings are anisotropic: every chatter lands ~0.99
cosine from every other because "is twitch chat" dominates the direction. So
similarity is computed on CENTERED vectors — subtract the roster's mean
person-vector and renormalize, leaving only how each person deviates from the
generic chatter. (Same cure as the lexical layer's everyone-overlaps problem.)
"""

import os
import pickle

import config
from utils import chat_archive

_FILE = os.path.join("data", "unsynced", "persona_embeddings.pkl")
_DATA = None
_CENTERED = None
_STAMP = None

# All-but-the-top-k isotropy correction (Mu & Viswanath, ICLR 2018) applied
# AFTER mean-centering. Centering already does the heavy lifting (mean
# |off-diagonal cosine| 0.983 raw -> 0.165 centered); k chosen EMPIRICALLY by
# scripts/eval_geometry.py, not by theory, because the effect is non-monotonic:
#   k=0  person-cos 0.165 / axis-score-corr 0.302
#   k=1  person-cos 0.157 / axis-score-corr 0.335  (k=1 tangles the trait axes!)
#   k=2  person-cos 0.149 / axis-score-corr 0.249  (improves BOTH)
# k=1 removes a direction that carried axis-discriminative signal; the 2nd
# component is a shared nuisance whose removal helps person similarity AND trait
# decorrelation. At k=2 every trait axis still retains >=0.94 of its energy
# (eval_geometry guard: doomer 0.943 lowest, unhinged 0.995 highest), so the
# dials are not blunted. Set to 0 to disable.
ABTT_K = 2


def _metadata_current(data) -> bool:
    return bool(
        isinstance(data, dict)
        and data.get("unit") == "utterance"
        and data.get("model") == config.LLM_EMBED_MODEL
        and data.get("alias_signature") == chat_archive.alias_signature()
        and data.get("utterance_version") == chat_archive.UTTERANCE_VERSION
    )


def load():
    """Cached pickle, hot-reloaded when the file changes (an offline rebuild
    should reach the running bot without a restart)."""
    global _DATA, _CENTERED, _STAMP
    stat = os.stat(_FILE)
    stamp = (stat.st_mtime_ns, stat.st_size)
    if _DATA is None or stamp != _STAMP:
        with open(_FILE, "rb") as fh:
            candidate = pickle.load(fh)
        if not _metadata_current(candidate):
            _DATA = None
            _CENTERED = None
            _STAMP = None
            raise ValueError("stale or incompatible person semantic vectors")
        _DATA = candidate
        _CENTERED = None
        _STAMP = stamp
    return _DATA


def available() -> bool:
    if not os.path.exists(_FILE):
        return False
    try:
        return bool(load().get("vectors"))
    except Exception:
        return False


def _centered():
    global _CENTERED
    load()
    if _CENTERED is None:
        import numpy as np
        vectors = load()["vectors"]
        names = list(vectors)
        M = np.vstack([vectors[a] for a in names]).astype("float64")
        M = M - M.mean(axis=0)
        if ABTT_K > 0 and M.shape[0] > ABTT_K + 1:
            # remove the top-k principal directions of the centered cloud
            _u, _s, vt = np.linalg.svd(M, full_matrices=False)
            pcs = vt[:ABTT_K]
            M = M - (M @ pcs.T) @ pcs
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        _CENTERED = {a: M[i].astype("float32") for i, a in enumerate(names)}
    return _CENTERED


def similarities(author):
    """{other: centered cosine} for every roster member, or {}."""
    if not available():
        return {}
    vectors = _centered()
    canon = chat_archive.normalize_author(author)
    key = _vector_key(vectors, canon)
    v = vectors.get(key) if key else None
    if v is None:
        return {}
    sims = {}
    for other, w in vectors.items():
        other_canon = chat_archive.normalize_author(other)
        if other_canon == canon or chat_archive._is_noise_author(other_canon):
            continue
        score = float(v @ w)
        if other_canon not in sims or score > sims[other_canon]:
            sims[other_canon] = score
    return sims


def neighbors(author, n=5):
    """[(author, cosine), ...] most semantically similar first, or []."""
    if not available():
        return []
    vectors = _centered()
    canon = chat_archive.normalize_author(author)
    key = _vector_key(vectors, canon)
    v = vectors.get(key) if key else None
    if v is None:
        return []
    best = {}
    for other, w in vectors.items():
        other_canon = chat_archive.normalize_author(other)
        if other_canon == canon or chat_archive._is_noise_author(other_canon):
            continue
        score = float(v @ w)
        if other_canon not in best or score > best[other_canon]:
            best[other_canon] = score
    sims = list(best.items())
    sims.sort(key=lambda kv: -kv[1])
    return sims[:n]


def _vector_key(vectors: dict, canon: str) -> str | None:
    if canon in vectors:
        return canon
    for key in chat_archive.author_keys(canon):
        if key in vectors:
            return key
    return None
