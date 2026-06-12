"""Read-side helpers for emote meaning (registry + usage-context vectors)."""

import json
import os
import pickle

REG_PATH = os.path.join("data", "unsynced", "emote_registry.json")
SEM_PATH = os.path.join("data", "unsynced", "emote_semantics.pkl")

_reg = None
_sem = None


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
    import numpy as np
    sem = semantics()
    key = token if token in sem else next((k for k in sem if k.lower() == token.lower()), None)
    if not key:
        return []
    v = np.asarray(sem[key]["vector"], dtype="float32")
    v /= (np.linalg.norm(v) + 1e-9)
    sims = []
    for k, d in sem.items():
        if k == key:
            continue
        w = np.asarray(d["vector"], dtype="float32")
        sims.append((k, float(v @ w / (np.linalg.norm(w) + 1e-9))))
    sims.sort(key=lambda kv: -kv[1])
    return sims[:n]


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
