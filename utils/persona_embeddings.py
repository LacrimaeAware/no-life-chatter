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


def load():
    global _DATA
    if _DATA is None:
        with open(_FILE, "rb") as fh:
            _DATA = pickle.load(fh)
    return _DATA


def available() -> bool:
    return os.path.exists(_FILE)


def _centered():
    global _CENTERED
    if _CENTERED is None:
        import numpy as np
        vectors = load()["vectors"]
        names = list(vectors)
        M = np.vstack([vectors[a] for a in names])
        M = M - M.mean(axis=0)
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        _CENTERED = {a: M[i] for i, a in enumerate(names)}
    return _CENTERED


def neighbors(author, n=5):
    """[(author, cosine), ...] most semantically similar first, or []."""
    if not available():
        return []
    vectors = _centered()
    canon = chat_archive.normalize_author(author)
    v = vectors.get(canon)
    if v is None:
        return []
    sims = [(c, float(v @ w)) for c, w in vectors.items() if c != canon]
    sims.sort(key=lambda kv: -kv[1])
    return sims[:n]
