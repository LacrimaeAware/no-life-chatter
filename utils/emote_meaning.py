"""Read-side helpers for emote meaning (registry + usage-context vectors)."""

import json
import os
import pickle

REG_PATH = os.path.join("data", "unsynced", "emote_registry.json")
SEM_PATH = os.path.join("data", "unsynced", "emote_semantics.pkl")
TAG_PATH = os.path.join("data", "unsynced", "emote_tag_vecs.pkl")

_reg = None
_sem = None
_centered = None
_names = None


def registry():
    global _reg
    if _reg is None:
        _reg = json.load(open(REG_PATH, encoding="utf-8")) if os.path.exists(REG_PATH) else {}
    return _reg


def semantics():
    global _sem
    if _sem is None:
        _sem = pickle.load(open(SEM_PATH, "rb")) if os.path.exists(SEM_PATH) else {}
    return _sem


def _center(d):
    import numpy as np
    names = list(d)
    if not names:
        return {}
    M = np.vstack([np.asarray(d[e], dtype="float32") for e in names])
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    M -= M.mean(axis=0)
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return {e: M[i] for i, e in enumerate(names)}


def _centered_space():
    """Blended emote-MEANING space: usage-context (0.45) + cleaned 7TV tags
    (0.55), each centered (raw chat embeddings are anisotropic). Usage alone
    confounds STIMULUS with STANCE (KEKW and SEETHE react to the same
    situations); tags carry the stance for ~1700 emotes and pull the whole
    geometry apart, which even fixes untagged emotes' neighborhoods. Raw
    emote-NAME embeddings are deliberately excluded (string-similarity
    pollution: KEKW->KEKWait, DansGame->'Dance')."""
    global _centered, _names
    if _centered is None:
        import numpy as np
        sem = semantics()
        U = _center({e: d["vector"] for e, d in sem.items()})
        T = {}
        if os.path.exists(TAG_PATH):
            T = _center(pickle.load(open(TAG_PATH, "rb")))
        _names = sorted(set(U) | set(T))
        out = {}
        for e in _names:
            parts = []
            if e in U:
                parts.append(0.45 * U[e])
            if e in T:
                parts.append(0.55 * T[e])
            v = np.sum(parts, axis=0)
            n = np.linalg.norm(v)
            if n > 0:
                out[e] = v / n
        _centered = out
    return _centered


def vector(token):
    """Centered usage vector for an emote (case-insensitive), or None."""
    c = _centered_space()
    if token in c:
        return c[token]
    return next((c[k] for k in c if k.lower() == token.lower()), None)


def semantic_key(token):
    """Exact stored usage-vector key for an emote token, case-insensitive."""
    sem = semantics()
    if token in sem:
        return token
    low = (token or "").lower()
    return next((k for k in sem if k.lower() == low), None)


def usage_count(token):
    """How many cleaned context snippets built this emote's usage vector."""
    key = semantic_key(token)
    if not key:
        return 0
    data = semantics().get(key) or {}
    try:
        return int(data.get("n") or 0)
    except Exception:
        return 0


def lookup(token):
    """Registry facts for an emote token (case-insensitive), or None."""
    reg = registry()
    if token in reg:
        return token, reg[token]
    low = token.lower()
    for k, v in reg.items():
        if k.lower() == low:
            return k, v
    return None, None


def nearest_emotes(token, n=6):
    """Emotes whose usage-context is closest (same meaning by usage), or []."""
    v = vector(token)
    if v is None:
        return []
    c = _centered_space()
    sims = [(k, float(v @ w)) for k, w in c.items()
            if not (k == token or k.lower() == token.lower())]
    sims.sort(key=lambda kv: -kv[1])
    return sims[:n]


def meaning_tags(token, n=6):
    """Plain-meaning words for an emote: the 7TV tags of its nearest-usage
    emotes (more reliable than cross-space probe words)."""
    from collections import Counter
    reg = registry()
    c = Counter()
    for e, sim in nearest_emotes(token, 12):
        for t in (reg.get(e, {}).get("tags") or [])[:4]:
            c[t.lower()] += sim
    return [t for t, _ in c.most_common(n)]


def meaning_words(token, probes=None, n=3):
    """Which plain-meaning probe words the emote's usage sits closest to."""
    import numpy as np
    from utils.persona_traits import _embed
    sem = semantics()
    key = token if token in sem else next((k for k in sem if k.lower() == token.lower()), None)
    if not key:
        return []
    probes = probes or ["funny laughing", "sad crying", "disgust gross", "angry mad",
                        "happy wholesome", "scared anxious", "horny lust", "confused",
                        "hype excited", "sarcastic mocking", "love affection", "dance party"]
    v = np.asarray(sem[key]["vector"], dtype="float32")
    v /= (np.linalg.norm(v) + 1e-9)
    P = _embed(probes)
    scored = []
    for p, e in zip(probes, P):
        e = np.asarray(e, dtype="float32")
        e /= (np.linalg.norm(e) + 1e-9)
        scored.append((p, float(v @ e)))
    scored.sort(key=lambda kv: -kv[1])
    return scored[:n]
