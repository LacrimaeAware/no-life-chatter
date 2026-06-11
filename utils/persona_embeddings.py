"""Semantic person-vectors (built offline by scripts/build_persona_embeddings.py).

The lexical layer (~markers/~like) sees which exact words someone overuses;
this layer sees what they talk ABOUT — two chatters with zero shared
catchphrases can still be semantic twins. Vectors are mean-pooled local
embeddings of each person's messages, stored in a gitignored pickle.
"""

import os
import pickle

from utils import chat_archive

_FILE = os.path.join("data", "unsynced", "persona_embeddings.pkl")
_DATA = None


def load():
    global _DATA
    if _DATA is None:
        with open(_FILE, "rb") as fh:
            _DATA = pickle.load(fh)
    return _DATA


def available() -> bool:
    return os.path.exists(_FILE)


def neighbors(author, n=5):
    """[(author, cosine), ...] most semantically similar first, or []."""
    if not available():
        return []
    vectors = load()["vectors"]
    canon = chat_archive.normalize_author(author)
    v = vectors.get(canon)
    if v is None:
        return []
    sims = [(c, float(v @ w)) for c, w in vectors.items() if c != canon]
    sims.sort(key=lambda kv: -kv[1])
    return sims[:n]
