"""Dynamic + emote-aware trait axes (Bucket C #17, first slices).

Two upgrades over the static persona_traits axes:

1. **Dynamic axes**: ~top <anything> — if no axis exists, the local LLM writes
   pole example sentences for the term and its opposite, they're embedded
   into an axis, and the axis is SAVED (data/unsynced/custom_axes.pkl). If a
   later request is semantically a duplicate (axis-direction cosine >= 0.80,
   e.g. racism vs bigot), the new name merges as an alias of the existing
   axis instead of fragmenting.

2. **Emote-name semantics**: emote names are self-describing (ApuDoomer,
   FeelsBadMan, ICANTSTOPFUCKINGDESPAIRING), and for short-form posters the
   trait signal lives THERE, not in their prose. Each person gets an
   emote-vector (their profile's distinctive emotes, log-odds weighted, name
   embeddings mean-pooled); axis projections blend text 0.75 / emotes 0.25.
"""

import json
import logging
import os
import pickle
import re
import urllib.request

import config
from utils import chat_archive, persona_embeddings
from utils.persona_traits import AXES, _axis_vectors, _embed, pole_map

CUSTOM_FILE = os.path.join("data", "unsynced", "custom_axes.pkl")
EMOTE_VEC_FILE = os.path.join("data", "unsynced", "emote_embeddings.pkl")
MERGE_COSINE = 0.80
TEXT_W, EMOTE_W = 0.75, 0.25

_custom = None
_emote_person_vecs = None


# ----------------------------- custom axes -----------------------------

def _load_custom():
    global _custom
    if _custom is None:
        if os.path.exists(CUSTOM_FILE):
            with open(CUSTOM_FILE, "rb") as fh:
                _custom = pickle.load(fh)
        else:
            _custom = {}
    return _custom


def _save_custom():
    with open(CUSTOM_FILE, "wb") as fh:
        pickle.dump(_custom, fh)


def _chat_sync(prompt, max_tokens=400):
    """Blocking chat call (callers run via asyncio.to_thread)."""
    body = json.dumps({
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.7, "stream": False,
    }).encode()
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def _generate_poles(term):
    """LLM-written pole sentences for an arbitrary trait term.
    Returns (opposite_label, pos_sentences, neg_sentences) or None."""
    prompt = (
        f'Trait: "{term}".\n'
        'Reply with ONLY a JSON object, no other text:\n'
        '{"opposite": "<one-word opposite of the trait>",\n'
        ' "trait_examples": ["5 short chat messages that maximally express the trait"],\n'
        ' "opposite_examples": ["5 short chat messages that maximally express the opposite"]}\n'
        "Write the examples in casual Twitch-chat register (lowercase ok, "
        "slang ok). Make them strongly, unambiguously expressive of each pole."
    )
    try:
        raw = _chat_sync(prompt)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        d = json.loads(m.group(0))
        pos = [s for s in d.get("trait_examples", []) if isinstance(s, str)][:6]
        neg = [s for s in d.get("opposite_examples", []) if isinstance(s, str)][:6]
        opp = re.sub(r"[^a-z0-9_-]", "", str(d.get("opposite", "")).lower()) or f"non-{term}"
        if len(pos) >= 3 and len(neg) >= 3:
            return opp, pos, neg
    except Exception as e:
        logging.warning(f"dynamic axis generation failed for {term!r}: {e}")
    return None


def _all_axis_vectors():
    """{name: (vector, pos_label, neg_label)} for builtin + custom axes."""
    import numpy as np
    out = {}
    for name, v in _axis_vectors().items():
        out[name] = (np.asarray(v), AXES[name][1], AXES[name][0])
    for name, d in _load_custom().items():
        out[name] = (np.asarray(d["vector"]), d["pos_label"], d["neg_label"])
    return out


NAME_MERGE_COSINE = 0.60  # word-embedding: synonyms >=0.66, distinct <=0.49


def _name_merge_candidate(term):
    """Does this term's WORD embedding already match an existing axis pole?
    (LLM-written pole sentences vary run to run, so finished-axis cosine
    misses synonyms; the term names themselves separate cleanly — measured:
    same concept >=0.66, different concepts <=0.49.)"""
    import numpy as np
    labels = []   # (label, axis_name, sign)
    for name, (_v, pos, neg) in _all_axis_vectors().items():
        labels.append((pos, name, +1))
        labels.append((neg, name, -1))
        for al in _load_custom().get(name, {}).get("aliases", []):
            labels.append((al, name, +1))
    embs = _embed([term] + [l for l, _, _ in labels])
    t = np.asarray(embs[0]); t /= (np.linalg.norm(t) + 1e-9)
    best, best_cos, best_sign = None, 0.0, 1
    for (label, name, sign), e in zip(labels, embs[1:]):
        e = np.asarray(e); e /= (np.linalg.norm(e) + 1e-9)
        c = float(t @ e)
        if c > best_cos:
            best, best_cos, best_sign = name, c, sign
    if best and best_cos >= NAME_MERGE_COSINE:
        return best, best_sign
    return None


def resolve_axis(term):
    """(axis_name, sign, note) for any term — builtin pole, custom axis or
    alias, or a freshly built+saved dynamic axis. None if it can't be built.
    note explains what happened ('new axis ...', 'merged into ...', None)."""
    import numpy as np
    term = (term or "").lower().strip()
    if not term:
        return None
    builtin = pole_map()
    if term in builtin:
        return (*builtin[term], None)
    custom = _load_custom()
    for name, d in custom.items():
        if term == name or term in d.get("aliases", []):
            return name, +1, None
        if term == d["neg_label"]:
            return name, -1, None

    # synonym of an existing axis? alias it instead of building a duplicate
    hit = _name_merge_candidate(term)
    if hit:
        name, sign = hit
        if name in custom:
            custom[name].setdefault("aliases", []).append(term)
            _save_custom()
        return name, sign, f"'{term}' ≈ existing axis '{name}' — merged"

    made = _generate_poles(term)
    if not made:
        return None
    opp, pos_s, neg_s = made
    embs = _embed(neg_s + pos_s)
    neg = np.asarray(embs[:len(neg_s)], dtype="float32").mean(axis=0)
    pos = np.asarray(embs[len(neg_s):], dtype="float32").mean(axis=0)
    v = pos - neg
    v = v / (np.linalg.norm(v) + 1e-9)

    # near-duplicate of an existing axis? merge as alias instead of forking
    best, best_cos = None, 0.0
    for name, (av, _p, _n) in _all_axis_vectors().items():
        c = float(v @ av)
        if abs(c) > best_cos:
            best, best_cos, best_sign = name, abs(c), (1 if c >= 0 else -1)
    if best and best_cos >= MERGE_COSINE:
        if best in custom:
            custom[best].setdefault("aliases", []).append(term)
            _save_custom()
        return best, best_sign, f"'{term}' ≈ existing axis '{best}' — merged"

    custom[term] = {"pos_label": term, "neg_label": opp, "vector": v,
                    "pos_sentences": pos_s, "neg_sentences": neg_s, "aliases": []}
    _save_custom()
    return term, +1, f"new axis '{term}' (opposite: {opp}) built and saved"


# ----------------------- emote-aware projections -----------------------

def _emote_vectors():
    """Per-person emote-name semantic vector (log-odds-weighted mean of their
    distinctive emotes' name embeddings), centered across the roster."""
    global _emote_person_vecs
    if _emote_person_vecs is not None:
        return _emote_person_vecs
    import numpy as np
    from utils import persona_classifier as pc
    profiles = pc.load().get("profiles") or {}

    cache = {}
    if os.path.exists(EMOTE_VEC_FILE):
        with open(EMOTE_VEC_FILE, "rb") as fh:
            cache = pickle.load(fh)
    all_emotes = sorted({e for p in profiles.values() for e in p.get("emotes", {})})
    missing = [e for e in all_emotes if e not in cache]
    for i in range(0, len(missing), 64):
        batch = missing[i:i + 64]
        # split camel-case so the embedder reads words: ApuDoomer -> Apu Doomer
        readable = [re.sub(r"(?<=[a-z])(?=[A-Z])", " ", e) for e in batch]
        for e, v in zip(batch, _embed(readable)):
            cache[e] = np.asarray(v, dtype="float32")
    if missing:
        with open(EMOTE_VEC_FILE, "wb") as fh:
            pickle.dump(cache, fh)

    vecs = {}
    for author, prof in profiles.items():
        em = prof.get("emotes", {})
        if not em:
            continue
        acc = np.zeros(next(iter(cache.values())).shape, dtype="float32")
        for e, w in em.items():
            if e in cache:
                acc += w * cache[e]
        n = np.linalg.norm(acc)
        if n > 0:
            vecs[author] = acc / n
    if vecs:
        names = list(vecs)
        M = np.vstack([vecs[a] for a in names])
        M = M - M.mean(axis=0)
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        vecs = {a: M[i] for i, a in enumerate(names)}
    _emote_person_vecs = vecs
    return vecs


def axis_scores(axis_name):
    """{author: z} on an axis, blending text and emote-name semantics.
    Short-form posters whose traits live in their emotes get read correctly."""
    import numpy as np
    av, _pos, _neg = _all_axis_vectors()[axis_name]
    text = persona_embeddings._centered()
    emote = _emote_vectors()
    names = list(text)
    raw = []
    for a in names:
        s = TEXT_W * float(text[a] @ av)
        if a in emote:
            s += EMOTE_W * float(emote[a] @ av)
        raw.append(s)
    raw = np.array(raw)
    z = (raw - raw.mean()) / (raw.std() or 1.0)
    return dict(zip(names, z))


def top(term, n=5):
    """(rows, note): leaderboard toward any term — builtin, saved, or freshly
    built. rows=None if the axis couldn't be made."""
    resolved = resolve_axis(term)
    if not resolved:
        return None, None
    axis, sign, note = resolved
    scores = axis_scores(axis)
    ranked = sorted(scores.items(), key=lambda kv: -sign * kv[1])[:n]
    return [(a, sign * z) for a, z in ranked], note
