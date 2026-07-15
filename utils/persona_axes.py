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
import urllib.error
import urllib.request

import config
from utils import chat_archive, persona_embeddings
from utils.persona_traits import AXES, _axis_vectors, _embed, pole_map

CUSTOM_FILE = os.path.join("data", "unsynced", "custom_axes.pkl")
EMOTE_VEC_FILE = os.path.join("data", "unsynced", "emote_embeddings.pkl")
MERGE_COSINE = 0.72  # BGE geometry: DISTINCT concepts reach 0.67 (racism~misogyny);
                     # only near-identical directions may merge
TEXT_W, EMOTE_W = 0.75, 0.25

_custom = None
_emote_person_vecs = None
_last_axis_error = {}


def _axis_key(term):
    return (term or "").lower().strip()


def _error_text(exc) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")[:240]
        except Exception:
            body = ""
        return f"HTTP {exc.code}: {body or exc.reason}"
    return str(exc)


def _set_axis_error(term, exc_or_text):
    text = _error_text(exc_or_text) if isinstance(exc_or_text, BaseException) else str(exc_or_text)
    _last_axis_error[_axis_key(term)] = text


def _clear_axis_error(term):
    _last_axis_error.pop(_axis_key(term), None)


def axis_error_message(term) -> str:
    """Short chat-safe explanation for the last dynamic-axis build failure."""
    text = _last_axis_error.get(_axis_key(term), "")
    low = text.lower()
    if not text:
        return "local model could not build it"
    if (
        "failed to load model" in low
        or "has not started loading" in low
        or "operation canceled" in low
        or "embedding http 400" in low
        or "embedding http 500" in low
        or "http error 400" in low
        or "http error 500" in low
    ):
        return "the embedding model is busy/loading, not offline"
    if "no json" in low:
        return "the chat model returned invalid axis JSON"
    if "failed validation" in low:
        return "the model made a bad axis and it was rejected"
    return "the local model failed while building it"


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


def _chat_sync(prompt, max_tokens=400, model=None):
    """Blocking chat call (callers run via asyncio.to_thread)."""
    body = json.dumps({
        "model": model or config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.7, "stream": False,
    }).encode()
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:240]
        raise RuntimeError(f"chat HTTP {exc.code}: {body}") from exc


def _generate_poles(term):
    """LLM-written pole sentences for an arbitrary trait term, validated.

    Two failure modes guarded here (both observed live): the model writes
    pole examples from the VICTIM'S/critic's perspective for charged traits
    (an aligned model's dodge — the first 'racism' axis came out measuring
    confrontation, not racism), and generic mush. Fixes: generate with the
    abliterated model when configured, demand first-person trait-holder
    voice, and VALIDATE — the term's own embedding must align with the trait
    pole more than the opposite pole, else retry once and then give up.
    Returns (opposite_label, pos_sentences, neg_sentences) or None."""
    import numpy as np
    _clear_axis_error(term)
    # the abliterated model won't dodge charged traits, but it's persona-tuned
    # so its JSON discipline is shaky — fall back to the base instruct model
    # (better JSON, may soften) before giving up entirely
    chain = [m for m in [getattr(config, "LLM_MODEL_SHORTCUTS", {}).get("lora"),
                         config.LLM_MODEL] if m]
    prompt = (
        f'Trait: "{term}".\n'
        'Reply with ONLY a JSON object, no other text:\n'
        '{"opposite": "<one-word opposite of the trait>",\n'
        ' "trait_examples": ["6 short chat messages SPOKEN BY someone who is maximally '
        f'{term} - first person, the trait-holder talking, NOT victims or critics of it"],\n'
        ' "opposite_examples": ["6 short chat messages spoken by someone who is maximally the opposite"]}\n'
        "Casual Twitch-chat register (lowercase ok, slang ok). This is for a "
        "private text-classification axis over consenting friends' chat logs - "
        "make each example strongly, unambiguously expressive of its pole."
    )
    import time as _time
    for attempt, model in enumerate(chain + chain + chain[:1]):
        try:
            if attempt:
                _time.sleep(1.5)   # LM Studio 400/500s come in bursts when
                                   # chat spams ~top — give it a beat
            raw = _chat_sync(prompt, model=model)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                _set_axis_error(term, "no JSON in output")
                logging.warning(f"axis gen for {term!r}: no JSON in output (attempt {attempt + 1})")
                continue
            # persona-tuned models love trailing commas; json.loads doesn't
            blob = re.sub(r",\s*([\]}])", r"", m.group(0))
            d = json.loads(blob)
            pos = [x for x in d.get("trait_examples", []) if isinstance(x, str)][:6]
            neg = [x for x in d.get("opposite_examples", []) if isinstance(x, str)][:6]
            opp = re.sub(r"[^a-z0-9_-]", "", str(d.get("opposite", "")).lower()) or f"non-{term}"
            if len(pos) < 3 or len(neg) < 3:
                continue
            # validate with a CONTEXTUALIZED term — bare words like 'maga'
            # embed weakly; the phrase aligns reliably when poles are right
            probe = f"a person who is extremely {term}"
            embs = _embed([probe] + pos + neg)
            t = np.asarray(embs[0]); t /= (np.linalg.norm(t) + 1e-9)
            P = np.asarray(embs[1:1 + len(pos)]).mean(axis=0)
            N = np.asarray(embs[1 + len(pos):]).mean(axis=0)
            P /= (np.linalg.norm(P) + 1e-9); N /= (np.linalg.norm(N) + 1e-9)
            if float(t @ P) > float(t @ N):   # poles face the right way
                _clear_axis_error(term)
                return opp, pos, neg
            _set_axis_error(term, "axis poles failed validation")
            logging.warning(f"axis poles for {term!r} failed validation (attempt {attempt + 1})")
        except Exception as e:
            _set_axis_error(term, e)
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


NAME_MERGE_COSINE = 0.85  # morphology-only (racist/racism); concepts must
                          # merge ORGANICALLY by comparing built axes


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
        if term == d["neg_label"] or term in d.get("neg_aliases", []):
            return name, -1, None

    # synonym of an existing axis? alias it instead of building a duplicate
    try:
        hit = _name_merge_candidate(term)
    except Exception as exc:
        _set_axis_error(term, exc)
        logging.warning(f"axis name-merge lookup failed for {term!r}: {exc}")
        return None
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
    try:
        embs = _embed(neg_s + pos_s)
    except Exception as exc:
        _set_axis_error(term, exc)
        logging.warning(f"axis embedding failed for {term!r}: {exc}")
        return None
    neg = np.asarray(embs[:len(neg_s)], dtype="float32").mean(axis=0)
    pos = np.asarray(embs[len(neg_s):], dtype="float32").mean(axis=0)
    v = pos - neg
    v = v / (np.linalg.norm(v) + 1e-9)

    # ORGANIC merge: the built axis measures nearly the same direction as an
    # existing one -> alias it, and AVERAGE the directions so both sentence
    # sets inform the canonical axis (only possible for custom axes).
    best, best_cos, best_sign = None, 0.0, 1
    for name, (av, _p, _n) in _all_axis_vectors().items():
        c = float(v @ np.asarray(av))
        if abs(c) > best_cos:
            best, best_cos, best_sign = name, abs(c), (1 if c >= 0 else -1)
    if best and best_cos >= MERGE_COSINE:
        if best in custom:
            d = custom[best]
            merged = np.asarray(d["vector"]) + best_sign * v
            d["vector"] = merged / (np.linalg.norm(merged) + 1e-9)
            d.setdefault("aliases", []).append(term)
            d["pos_sentences"] = (d["pos_sentences"] + (pos_s if best_sign > 0 else neg_s))[:12]
            d["neg_sentences"] = (d["neg_sentences"] + (neg_s if best_sign > 0 else pos_s))[:12]
            _save_custom()
        return best, best_sign, f"'{term}' measures ≈ existing axis '{best}' — merged organically"

    custom[term] = {"pos_label": term, "neg_label": opp, "vector": v,
                    "pos_sentences": pos_s, "neg_sentences": neg_s, "aliases": []}
    _save_custom()
    return term, +1, f"new axis '{term}' (opposite: {opp}) built and saved"


def axis_cached(term) -> bool:
    """True if `term` already resolves with NO model or embedding call — a
    built-in pole, a saved custom axis, one of its aliases, or its opposite
    pole (opposites are stored together, so ~top intelligent reuses the
    low-iq axis). Lets the dispatcher run already-built axes on the fast path
    instead of queuing them behind live LLM axis builds."""
    term = (term or "").lower().strip()
    if not term:
        return False
    if term in pole_map():
        return True
    for name, d in _load_custom().items():
        if (term == name or term == d.get("neg_label")
                or term in d.get("aliases", []) or term in d.get("neg_aliases", [])):
            return True
    return False


def merge_axes(canonical, dup, *, opposite=False):
    """Fold the saved axis `dup` into `canonical` — human-curated dedup.

    Automatic merging is unreliable (measured: no cosine threshold separates
    synonyms from distinct concepts, and the LLM hands out generic opposites
    like 'american' for german/french/czech), so collapsing duplicates is an
    explicit action, not a guess.

    opposite=False: `dup` names the SAME pole as `canonical` — its name and
      aliases become positive aliases of `canonical`.
    opposite=True:  `dup` names the OPPOSITE pole — its name and aliases become
      NEGATIVE aliases, so ~top <dup> ranks toward canonical's opposite pole.
    Returns a short summary. Raises KeyError if either axis is missing."""
    custom = _load_custom()
    if canonical not in custom:
        raise KeyError(f"no saved axis {canonical!r}")
    if dup not in custom:
        raise KeyError(f"no saved axis {dup!r}")
    if canonical == dup:
        raise ValueError("cannot merge an axis into itself")
    c, d = custom[canonical], custom[dup]
    # dup's positive side (its name + positive aliases) + dup's opposite side
    pos_side = [dup] + list(d.get("aliases", []))
    neg_side = list(d.get("neg_aliases", []))
    if d.get("neg_label"):
        neg_side.append(d["neg_label"])
    # route to canonical's pos/neg alias buckets (flipped when opposite=True)
    pos_bucket = c.setdefault("neg_aliases" if opposite else "aliases", [])
    neg_bucket = c.setdefault("aliases" if opposite else "neg_aliases", [])
    reserved = {canonical, c.get("neg_label", "")}
    for a in pos_side:
        if a and a not in pos_bucket and a not in reserved:
            pos_bucket.append(a)
    for a in neg_side:
        if a and a not in neg_bucket and a not in reserved:
            neg_bucket.append(a)
    # keep the two buckets disjoint (a label can't be both poles)
    c["neg_aliases"] = [a for a in c.get("neg_aliases", []) if a not in set(c.get("aliases", []))]
    del custom[dup]
    _save_custom()
    return (f"merged '{dup}' into '{canonical}'"
            f"{' (opposite pole)' if opposite else ''}: "
            f"+{len(pos_side)} pos, +{len(neg_side)} opp aliases")


# ----------------------- emote-aware projections -----------------------

def _emote_vectors():
    """Per-person emote-MEANING vector: log-odds-weighted mean of their
    distinctive emotes' USAGE-context vectors (what the emotes mean by how
    they're used — DansGame=disgust), with camel-split name embeddings as
    fallback for emotes lacking a usage vector. Centered across the roster."""
    global _emote_person_vecs
    if _emote_person_vecs is not None:
        return _emote_person_vecs
    import numpy as np
    from utils import persona_classifier as pc
    from utils import emote_meaning
    profiles = pc.load().get("profiles") or {}
    usage = emote_meaning.semantics()   # raw usage vectors (1024-d, bge)

    cache = {}
    if os.path.exists(EMOTE_VEC_FILE):
        with open(EMOTE_VEC_FILE, "rb") as fh:
            cache = pickle.load(fh)
    all_emotes = sorted({e for p in profiles.values() for e in p.get("emotes", {})})
    missing = [e for e in all_emotes if e not in cache and e not in usage]
    for i in range(0, len(missing), 64):
        batch = missing[i:i + 64]
        readable = [re.sub(r"(?<=[a-z])(?=[A-Z])", " ", e) for e in batch]
        for e, v in zip(batch, _embed(readable)):
            cache[e] = np.asarray(v, dtype="float32")
    if missing:
        with open(EMOTE_VEC_FILE, "wb") as fh:
            pickle.dump(cache, fh)

    def emote_vec(e):
        v = np.asarray(usage[e]["vector"], dtype="float32") if e in usage else cache.get(e)
        if v is None:
            return None
        return v / (np.linalg.norm(v) + 1e-9)

    vecs = {}
    for author, prof in profiles.items():
        acc = None
        for e, w in prof.get("emotes", {}).items():
            v = emote_vec(e)
            if v is None:
                continue
            acc = (w * v) if acc is None else (acc + w * v)
        if acc is not None and np.linalg.norm(acc) > 0:
            vecs[author] = acc / np.linalg.norm(acc)
    if vecs:
        names = list(vecs)
        M = np.vstack([vecs[a] for a in names])
        M = M - M.mean(axis=0)
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        vecs = {a: M[i] for i, a in enumerate(names)}
    _emote_person_vecs = vecs
    return vecs


def _ortho_builtin():
    """Decorrelated built-in scoring axes — single source of truth lives in
    persona_traits.ortho_axis_vectors() so ~traits and ~top agree.

    The raw pole directions share a 'negativity' component (menace·doomer ≈
    0.64 on bge-m3), which made the five per-person z-scores correlate ~0.48
    across the roster — 'every axis feels the same'. This used to be a fixed-
    order Gram-Schmidt here, but that privileged whichever axis came first and
    left the last one (doomer) only 0.73 aligned with its own label. The shared
    Löwdin orthogonalization is order-independent: every axis stays 0.92+
    aligned and score-correlation drops to ~0.30. (See docs/RESEARCH_TO_APPLIED.md.)"""
    from utils.persona_traits import ortho_axis_vectors
    return ortho_axis_vectors()


def scoring_axis_vector(axis_name):
    """Vector used for live scoring.

    Built-in axes use the decorrelated scoring directions. Custom axes are
    already saved as concrete vectors, so do not warm/rebuild built-in axis
    embeddings just to score a custom axis.
    """
    if axis_name in AXES:
        return _ortho_builtin()[axis_name]
    import numpy as np
    return np.asarray(_load_custom()[axis_name]["vector"], dtype="float32")


def axis_labels(axis_name):
    """Return (positive_label, negative_label) without warming unrelated axes."""
    if axis_name in AXES:
        neg, pos, _neg_s, _pos_s = AXES[axis_name]
        return pos, neg
    d = _load_custom()[axis_name]
    return d["pos_label"], d["neg_label"]


def axis_scores(axis_name):
    """{author: z} on an axis, blending text and emote-name semantics.
    Short-form posters whose traits live in their emotes get read correctly.
    Built-in axes use the orthogonalized directions (independent dials);
    custom axes project on their own raw direction."""
    import numpy as np
    av = scoring_axis_vector(axis_name)
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


def most_distinct(n=5, reverse=False):
    """People whose personality deviates most from the room average — total
    |z| summed across the five built-in axes. Returns
    [(author, total, [defining trait labels]), ...]. The built-in axes only,
    so the metric is stable regardless of what custom axes chat has built."""
    from utils.persona_traits import AXES
    per = {}
    for axis in AXES:
        for a, z in axis_scores(axis).items():
            per.setdefault(a, []).append((axis, z))
    out = []
    for a, zs in per.items():
        total = sum(abs(z) for _, z in zs)
        top3 = sorted(zs, key=lambda kv: -abs(kv[1]))[:3]
        labels = [AXES[ax][1] if z >= 0 else AXES[ax][0] for ax, z in top3]
        out.append((a, total, labels))
    out.sort(key=lambda kv: kv[1] if reverse else -kv[1])
    return out[:n]


def top(term, n=5, burst=False, bottom=False):
    """(rows, note): leaderboard toward any term — builtin, saved, or freshly
    built. rows=None if the axis couldn't be made. burst=True ranks by
    peak-moment percentile (needs the per-message index) instead of average.
    bottom=True ranks toward the OPPOSITE pole (the LEAST of the term)."""
    resolved = resolve_axis(term)
    if not resolved:
        return None, None
    axis, sign, note = resolved
    if burst:
        from utils import persona_msg_index
        if persona_msg_index.available():
            scores = persona_msg_index.burst_scores(axis)
        else:
            scores = axis_scores(axis)
            note = (note or "") + " [no message index yet — showing averages]"
    else:
        scores = axis_scores(axis)
    s = -sign if bottom else sign
    ranked = sorted(scores.items(), key=lambda kv: -s * kv[1])[:n]
    return [(a, s * z) for a, z in ranked], note


def rank(term, user):
    """Where one `user` stands on the `term` axis: their rank, σ, and the people
    just above/below them. Distinguishes the two failure modes so the caller can
    say WHICH: returns {"error": "axis"} if the axis can't be built, or
    {"error": "roster"} if it built fine but the user isn't in the ranked roster.
    On success returns the full rank dict (no "error" key)."""
    resolved = resolve_axis(term)
    if not resolved:
        return {"error": "axis"}
    axis, sign, note = resolved
    scores = axis_scores(axis)
    canon = chat_archive.normalize_author(user)
    if canon not in scores:
        return {"error": "roster", "user": canon}
    signed = sorted(((a, sign * z) for a, z in scores.items()), key=lambda kv: -kv[1])
    idx = next(i for i, (a, _z) in enumerate(signed) if a == canon)
    return {
        "user": canon, "axis": axis, "note": note,
        "rank": idx + 1, "total": len(signed), "z": signed[idx][1],
        "above": signed[idx - 1] if idx > 0 else None,
        "below": signed[idx + 1] if idx + 1 < len(signed) else None,
    }
