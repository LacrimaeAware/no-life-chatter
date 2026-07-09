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

from utils import chat_archive, emote_meaning, persona_msg_index as pmi

_EMOTES = None
# "trailing-caps" emote shape: a lowercase base then 1-3 trailing capitals and at
# most one more lowercase — monkaS, forsenE, forsenCD, monkaW. Deliberately does
# NOT match concatenated words like imGonna / iPhone / isThis / youTube (a capital
# starting a multi-letter word), which are NOT emotes.
_EMOTE_SHAPE = re.compile(r"[a-z]{2,}[A-Z]{1,3}[a-z]?$")


def _emote_set():
    """Registered emote names, EXACT case. Twitch/7TV emotes are case-sensitive
    ('Pog' renders, 'pog' does not), so matching exact-case avoids flagging
    common lowercase words ('there', 'this') that collide with an emote name."""
    global _EMOTES
    if _EMOTES is None:
        reg = emote_meaning.registry() or {}
        s = set(reg.keys())
        for info in reg.values():
            o = (info or {}).get("original")
            if o:
                s.add(str(o))
        _EMOTES = s
    return _EMOTES


def _is_emote(tok):
    """Best-effort emote detection from text. Requires MIXED case (an upper AND a
    lower letter) — Pog/Sadge/Lemon/FeelsDankMan/chatterAnalysis qualify, while
    the common-word emotes that cause false positives do not: lowercase words
    ('there', 'omg') and all-caps shouting/words ('HELP', 'YOU') are excluded.
    Within mixed-case, accept camelCase shape OR an exact registry match.

    Capitalization disambiguates most word/emote collisions ('Pain' the emote vs
    'in pain' the word), which is the main signal. Known residual errors:
    sentence-initial capitalized words ('Pain is temporary' counts 'Pain'), and
    all-caps emotes (KEKW/OMEGALUL) are dropped to avoid flagging SHOUTING. Could
    be improved with per-channel active-emote sets and token position. Treat the
    rate as a good approximation / lower bound, not exact."""
    t = tok.strip(",.!?:;\"'")
    if t.startswith("@") or len(t) < 2:   # @mentions are not emotes (don't double-count)
        return False
    has_upper = any(c.isupper() for c in t)
    has_lower = any(c.islower() for c in t)
    if not (has_upper and has_lower):     # exclude lowercase words AND all-caps shouting
        return False
    return t in _emote_set() or bool(_EMOTE_SHAPE.fullmatch(t))

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
    "words":    ("wordy", "terse"),
    "caps":     ("lots of CAPS", "all-lowercase"),
    "emote":    ("emote-heavy", "low on emotes (vs this chat)"),
    "exclaim":  ("exclamation-heavy", None),
    "question": ("asks lots of questions", None),
    "mention":  ("@s people directly", "posts to the room"),
    "laugh":    ("laughs a lot", None),
    "profan":   ("swears a lot", "barely swears"),
    "vocab":    ("rich vocabulary", "simple vocabulary"),
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
    excl = ment = ques = laugh = prof = 0
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
        emote_n += sum(1 for tk in toks if _is_emote(tk))
        excl += t.count("!")
        ment += t.count("@")
        if t.rstrip().endswith("?"):   # a trailing ? is the reliable signal; the
            ques += 1                  # leading-keyword heuristic flagged statements ("what a day")
        if LAUGH.search(t):
            laugh += 1
        if PROFAN.search(t):
            prof += 1
    return {
        "words": tot_words / n, "caps": caps_up / (caps_alpha + 1e-9), "emote": emote_n / n,
        "exclaim": excl / n, "question": ques / n,
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
