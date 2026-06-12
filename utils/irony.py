"""Live irony reading for an arbitrary message (the ~irony command).

Same features as scripts/irony_probe.py: surface-sarcasm axis + a
proposition-level harm axis, calibrated against the archive. Emotes are
RESOLVED to their usage-meaning words before scoring (so 'DansGame' adds
disgust instead of being stripped to nothing) — a first taste of the
emote-operator fix. context= can prepend extra context that shifts the read.
"""

import numpy as np

from utils import chat_archive, emote_meaning
from utils.persona_traits import _axis_vectors, _embed

_harm = None
_cal = {}


def _harm_axis():
    global _harm
    if _harm is None:
        pos = ['children deserve to suffer', 'i would enjoy hurting someone vulnerable',
               'causing pain to innocent people is good', 'i hope something terrible happens to them',
               'making others miserable brings me joy']
        neg = ['children deserve to be safe', 'i would enjoy helping someone vulnerable',
               'protecting innocent people is good', 'i hope something wonderful happens to them',
               'making others happy brings me joy']
        e = _embed(pos + neg)
        P = np.asarray(e[:5], dtype="float32").mean(axis=0)
        N = np.asarray(e[5:], dtype="float32").mean(axis=0)
        v = P - N
        _harm = v / (np.linalg.norm(v) + 1e-9)
    return _harm


def _calib(name, av):
    if name not in _cal:
        conn = chat_archive.connect()
        msgs = [r[0] for r in conn.execute(
            "SELECT content FROM messages WHERE LENGTH(content) > 12 "
            "ORDER BY RANDOM() LIMIT 300")]
        E = np.asarray(_embed(msgs), dtype="float32")
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        s = E @ av
        _cal[name] = (float(s.mean()), float(s.std()) or 1.0)
    return _cal[name]


def _resolve_emotes(text):
    """Replace recognized emotes with their usage-meaning words so the
    embedder sees the operator (DansGame -> 'disgust gross')."""
    out = []
    for tok in text.split():
        words = emote_meaning.meaning_words(tok, n=1)
        if words:
            out.append(tok + " (" + words[0][0].split()[0] + ")")
        else:
            out.append(tok)
    return " ".join(out)


def read(message, context=""):
    iron = np.asarray(_axis_vectors()["ironic"], dtype="float32")
    harm = _harm_axis()
    text = _resolve_emotes(message)
    if context:
        text = f"{context}. {text}"
    e = np.asarray(_embed([text])[0], dtype="float32")
    e /= (np.linalg.norm(e) + 1e-9)
    mi, si = _calib("ironic", iron)
    mh, sh = _calib("harm", harm)
    zi = (float(e @ iron) - mi) / si
    zh = (float(e @ harm) - mh) / sh
    if zi > 0.6:
        verdict = "reads IRONIC (marked sarcasm)"
    elif zh > 1.2:
        verdict = "reads DEADPAN-IRONIC (calm surface, extreme content)"
    elif zi < -0.6:
        verdict = "reads sincere"
    else:
        verdict = "unclear / mild"
    return verdict, zi, zh
