"""Behavioral (structural) personality profile — HOW a chatter types.

This is the half of personality that IS measurable from data (see
docs/PERSONALITY_SYSTEM_DESIGN.md): topic-free, intent-free counts like emote
rate, verbosity, caps, @mentions, profanity, vocabulary richness. Each chatter's
features are z-scored against the roster, so a profile is "what stands out about
how this person types." Used by the ~style command.

Intent/disposition traits (irony, hostility, sincerity) are deliberately NOT
here — surface counts cannot recover them.
"""

from __future__ import annotations

import os
import re

from utils import chat_archive, persona_msg_index as pmi

try:
    from utils.persona_classifier import _is_emote_token as _emote_tok
except Exception:
    def _emote_tok(t):
        return bool(re.match(r"[A-Z][a-z]+[A-Z]", t)) or (t.isalpha() and t.isupper() and len(t) >= 3)

LAUGH = re.compile(r"\b(l+u+l+w?|ke+kw?|lma+o+|lo+l|xd+|kekw|omegalul|icant|pepelaugh)\b", re.I)
PROFAN = re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|cunt\w*|nigg\w*|retard\w*|ass\w*|dick\w*|pussy\w*)\b", re.I)
ELONG = re.compile(r"(.)\1\1")
WORD = re.compile(r"[a-z']+", re.I)

# plain-language labels: feature -> (high-pole phrase, low-pole phrase).
# NOTE: whole-message doubling ("X X") is NOT included as a trait — it is mostly
# a logging/import artifact (duplicated rows), not a behavior. We still collapse
# it per-message before counting everything else, so it doesn't inflate the
# other features.
LABELS = {
    "words":    ("writes long messages", "writes short messages"),
    "caps":     ("TYPES IN CAPS A LOT", "types in all-lowercase"),
    "emote":    ("uses lots of emotes", "rarely uses emotes"),
    "exclaim":  ("lots of exclamation marks", None),
    "question": ("asks lots of questions", None),
    "repeat":   ("repeats words/emotes in a message", None),
    "elong":    ("stretches letters (soooo / loool)", None),
    "mention":  ("replies @ people directly", "posts to the room, not at people"),
    "laugh":    ("laughs a lot (LUL/KEK/lmao)", None),
    "profan":   ("swears a lot", "rarely swears"),
    "vocab":    ("varied vocabulary", "simple/repetitive vocabulary"),
}

_PROFILES = None   # {author: feature dict}
_STATS = None      # {feature: (mean, std)}


def _dedouble(t):
    w = t.split(); m = len(w)
    return " ".join(w[:m // 2]) if (m >= 2 and m % 2 == 0 and w[:m // 2] == w[m // 2:]) else t


def features(texts):
    n = len(texts)
    if n == 0:
        return None
    tot_words = caps_up = caps_alpha = emote_n = 0
    excl = ment = ques = rep = elong = laugh = prof = 0
    vocab = set()
    for t in texts:
        t = _dedouble(str(t))   # collapse logging/import doubling before counting
        toks = t.split()
        ws = WORD.findall(t)
        tot_words += len(ws)
        vocab.update(w.lower() for w in ws)
        for ch in t:
            if ch.isalpha():
                caps_alpha += 1
                if ch.isupper():
                    caps_up += 1
        emote_n += sum(1 for tk in toks if _emote_tok(tk.strip(",.!?")))
        excl += t.count("!")
        ment += t.count("@")
        if t.strip().endswith("?") or re.match(r"\s*(what|why|how|who|when|where|is|are|do|does|did|can|should)\b", t, re.I):
            ques += 1
        low = [w.lower() for w in toks]
        if any(low[i] == low[i + 1] and len(low[i]) >= 2 for i in range(len(low) - 1)):
            rep += 1
        if ELONG.search(t):
            elong += 1
        if LAUGH.search(t):
            laugh += 1
        if PROFAN.search(t):
            prof += 1
    return {
        "words": tot_words / n, "caps": caps_up / (caps_alpha + 1e-9), "emote": emote_n / n,
        "exclaim": excl / n, "question": ques / n, "repeat": rep / n, "elong": elong / n,
        "mention": ment / n, "laugh": laugh / n, "profan": prof / n,
        "vocab": len(vocab) / (tot_words + 1e-9),
    }


def available() -> bool:
    return pmi.available()


def _load_roster():
    global _PROFILES, _STATS
    if _PROFILES is None:
        import numpy as np
        profs = {}
        for f in sorted(os.listdir(pmi.DIR)) if os.path.isdir(pmi.DIR) else []:
            if not f.endswith(".npz"):
                continue
            a = f[:-4]
            try:
                _v, texts = pmi._load(a)
            except Exception:
                continue
            fe = features([str(t) for t in texts])
            if fe:
                profs[a] = fe
        stats = {}
        if profs:
            for k in next(iter(profs.values())):
                vals = np.array([p[k] for p in profs.values()])
                stats[k] = (float(vals.mean()), float(vals.std()) or 1.0)
        _PROFILES, _STATS = profs, stats
    return _PROFILES, _STATS


def profile(author):
    """[(feature, value, z, label)] sorted by |z| desc, or [] if no data.
    label is the high/low friendly phrase for the direction they lean."""
    profs, stats = _load_roster()
    canon = chat_archive.normalize_author(author)
    fe = profs.get(canon)
    if fe is None:
        fe = next((profs[k] for k in profs if chat_archive.normalize_author(k) == canon), None)
    if fe is None:
        return []
    out = []
    for k, v in fe.items():
        mu, sd = stats.get(k, (0.0, 1.0))
        z = (v - mu) / sd
        hi, lo = LABELS.get(k, (k, None))
        label = hi if z >= 0 else (lo or None)
        out.append((k, v, z, label))
    out.sort(key=lambda r: -abs(r[2]))
    return out
