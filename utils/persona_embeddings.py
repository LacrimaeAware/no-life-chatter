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

from utils import chat_archive

_FILE = os.path.join("data", "unsynced", "persona_embeddings.pkl")
_DATA = None
_CENTERED = None
_MTIME = None


def load():
    """Cached pickle, hot-reloaded when the file changes (an offline rebuild
    should reach the running bot without a restart)."""
    global _DATA, _CENTERED, _MTIME
    mtime = os.path.getmtime(_FILE)
    if _DATA is None or mtime != _MTIME:
        with open(_FILE, "rb") as fh:
            _DATA = pickle.load(fh)
        _CENTERED = None
        _MTIME = mtime
    return _DATA


def available() -> bool:
    return os.path.exists(_FILE)


def _centered():
    global _CENTERED
    load()
    if _CENTERED is None:
        import numpy as np
        vectors = load()["vectors"]
        names = list(vectors)
        M = np.vstack([vectors[a] for a in names])
        M = M - M.mean(axis=0)
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        _CENTERED = {a: M[i] for i, a in enumerate(names)}
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
