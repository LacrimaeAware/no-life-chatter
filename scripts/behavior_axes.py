"""Discover personality axes from BEHAVIOR, not topic.

The meaning-embedder encodes WHAT people say (topic/language), so PCA over it
finds topic axes, not personality (see scripts/discover_axes.py). This instead
measures HOW each person types — countable, topic-free style features — then runs
PCA over those and shows which features and which people define each axis, so a
human can name them. This is where "intense / attention-seeking / emote-reliant /
deadpan / verbose" actually live.

Features per author (rates over their archived messages, all topic-free):
  words        mean words per message (verbose <-> terse)
  caps         fraction of letters typed UPPERCASE (shouting/intensity)
  emote        emotes per message (emote-reliance)
  exclaim      ! per message
  question     fraction of messages that are questions
  repeat       fraction with a repeated span ("X X X" / copypasta)
  elong        fraction with elongation ("LMAOOOO", "soooo")
  mention      @mentions per message (directed <-> broadcasting)
  laugh        laughter markers per message (LUL/KEK/LMAO/xd)
  profan       profanity per message
  vocab        type-token ratio (vocabulary richness)

    python scripts/behavior_axes.py [--k 6]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from utils import persona_msg_index as pmi  # noqa: E402

try:
    from utils.persona_classifier import _is_emote_token as _emote_tok
except Exception:
    def _emote_tok(t):
        return bool(re.match(r"[A-Z][a-z]+[A-Z]", t)) or (t.isalpha() and t.isupper() and len(t) >= 3)

LAUGH = re.compile(r"\b(l+u+l+w?|ke+kw?|lma+o+|lo+l|xd+|kekw|omegalul|icant|pepelaugh)\b", re.I)
PROFAN = re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|cunt\w*|nigg\w*|retard\w*|ass\w*|dick\w*|pussy\w*)\b", re.I)
ELONG = re.compile(r"(.)\1\1")
WORD = re.compile(r"[a-z']+", re.I)


def _dedouble(t):
    """Collapse a whole-message exact doubling ("X Y X Y" -> "X Y"). This is a
    real Twitch duplicate-filter-bypass behavior AND/OR an import artifact, and
    it varies 0-31% by person, so it inflates every per-message count. We strip
    it for the OTHER features and count its RATE as its own behavior."""
    w = t.split()
    m = len(w)
    if m >= 2 and m % 2 == 0 and w[:m // 2] == w[m // 2:]:
        return " ".join(w[:m // 2]), True
    return t, False


def features(texts):
    n = len(texts)
    if n == 0:
        return None
    tot_words = caps_up = caps_alpha = emote_n = 0
    excl = ment = ques = rep = elong = laugh = prof = doubled_n = 0
    vocab = set()
    for t in texts:
        t, was_doubled = _dedouble(str(t))
        doubled_n += was_doubled
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
        # repeated span: any token (len>=2) immediately repeated
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
        "words": tot_words / n,
        "caps": caps_up / (caps_alpha + 1e-9),
        "emote": emote_n / n,
        "exclaim": excl / n,
        "question": ques / n,
        "repeat": rep / n,
        "elong": elong / n,
        "mention": ment / n,
        "laugh": laugh / n,
        "profan": prof / n,
        "vocab": len(vocab) / (tot_words + 1e-9),
        "doubles": doubled_n / n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    authors, rows = [], []
    for f in sorted(os.listdir(pmi.DIR)):
        if not f.endswith(".npz"):
            continue
        a = f[:-4]
        try:
            _V, texts = pmi._load(a)
        except Exception:
            continue
        fe = features(list(texts))
        if fe:
            authors.append(a); rows.append(fe)

    feat = list(rows[0])
    X = np.array([[r[f] for f in feat] for r in rows], dtype="float64")
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)            # z-score features
    U, s, Vt = np.linalg.svd(Xz - Xz.mean(0), full_matrices=False)
    var = (s ** 2) / (s ** 2).sum()
    load = Xz @ Vt[:args.k].T

    print(f"=== behavioral PCA: {args.k} axes over {len(authors)} people, "
          f"{len(feat)} topic-free features ===\n")
    for k in range(args.k):
        comp = Vt[k]
        top = np.argsort(np.abs(comp))[::-1][:5]
        desc = ", ".join(f"{'+' if comp[i]>0 else '-'}{feat[i]}" for i in top)
        order = np.argsort(load[:, k])
        pos = ", ".join(authors[i] for i in order[::-1][:5])
        neg = ", ".join(authors[i] for i in order[:5])
        print(f"--- BEHAVIOR AXIS {k+1} ---  variance {var[k]*100:.0f}%")
        print(f"  defined by: {desc}")
        print(f"  + people: {pos}")
        print(f"  - people: {neg}\n")

    # also print the raw feature leaderboard extremes for context
    print("=== feature extremes (who is highest on each raw behavior) ===")
    for fi, fname in enumerate(feat):
        col = X[:, fi]
        hi = authors[int(col.argmax())]
        lo = authors[int(col.argmin())]
        print(f"  {fname:9} high: {hi:22} ({col.max():.2f})   low: {lo} ({col.min():.2f})")


if __name__ == "__main__":
    main()
