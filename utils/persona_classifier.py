"""Authorship classifier: given any text, who's most likely to have said it.

A dependency-free character-n-gram + word Naive Bayes over the chat archive.
Character n-grams are the classic strong signal for *style* (casing, spelling,
emotes, punctuation rhythm); word tokens add vocabulary/topic. Works on novel
sentences, not just archived lines, because it scores sub-word features.

Two uses:
  - the `~whosaid` command (fun: "sounds most like X");
  - an objective persona metric — run a generated persona line through it and
    see whether it reads as the target author (and NOT as someone else).

Probabilities are a softmax over NB log-scores: a real ranking, but Naive Bayes
is overconfident (long inputs spike toward ~100%). Treat them as confidence,
not calibrated truth — which is exactly what the user asked for.
"""

import math
import os
import pickle
import random
import re

import config
from utils import chat_archive

_MODEL = None

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def _char_ngrams(text):
    s = " " + (text or "") + " "
    for n in (3, 4):
        for i in range(len(s) - n + 1):
            yield "c:" + s[i:i + n]


def _word_tokens(text):
    for w in _WORD_RE.findall((text or "").lower()):
        if len(w) >= 2:
            yield "w:" + w


def features(text):
    """Counts of style (char-ngram, case-preserving) + vocab (word) features."""
    from collections import Counter
    c = Counter()
    for g in _char_ngrams(text):
        c[g] += 1
    for w in _word_tokens(text):
        c[w] += 1
    return c


def _usable(msg):
    return bool(msg and len(msg.split()) >= 2 and not msg.lstrip().startswith(config.PREFIX))


def train(authors=None, per_author=4000, vocab_size=20000, alpha=0.1,
          test_frac=0.1, seed=1337, min_messages=300, max_authors=24):
    """Train and persist the classifier. Returns an accuracy report.

    authors: explicit list, else top non-bot authors by message count.
    """
    from collections import Counter
    rng = random.Random(seed)
    conn = chat_archive.connect()

    if authors:
        authors = [chat_archive.normalize_author(a) for a in authors]
    else:
        exclude = {u.lower() for u in getattr(config, "EXCLUDE_USERS", set())}
        rows = conn.execute(
            "SELECT author, COUNT(*) c FROM messages GROUP BY author "
            "HAVING c >= ? ORDER BY c DESC", (min_messages,)
        ).fetchall()
        authors = [a for a, _ in rows if a not in exclude][:max_authors]

    train_msgs, test_msgs = {}, {}
    for a in authors:
        msgs = [m for m in chat_archive.messages_for(a) if _usable(m)]
        rng.shuffle(msgs)
        msgs = msgs[: per_author]
        cut = int(len(msgs) * (1 - test_frac))
        train_msgs[a], test_msgs[a] = msgs[:cut], msgs[cut:]

    # Global feature frequency from TRAIN, to cap the vocabulary.
    global_df = Counter()
    per_author_counts = {a: Counter() for a in authors}
    for a in authors:
        for m in train_msgs[a]:
            f = features(m)
            per_author_counts[a].update(f)
            global_df.update(f.keys())
    vocab = {feat for feat, _ in global_df.most_common(vocab_size)}

    # Restrict author counts to vocab; compute totals + priors.
    counts = {a: {f: n for f, n in per_author_counts[a].items() if f in vocab} for a in authors}
    totals = {a: sum(counts[a].values()) for a in authors}
    n_train = sum(len(train_msgs[a]) for a in authors)
    priors = {a: math.log(max(1, len(train_msgs[a])) / n_train) for a in authors}

    _MODEL_LOCAL = {
        "authors": authors, "vocab": vocab, "counts": counts, "totals": totals,
        "priors": priors, "alpha": alpha, "vocab_size": len(vocab),
    }

    # Held-out accuracy: top-1 over the test split.
    correct = total = 0
    per_author_acc = {a: [0, 0] for a in authors}
    for a in authors:
        for m in test_msgs[a]:
            ranked = _classify_with(_MODEL_LOCAL, m, top_k=1)
            total += 1
            per_author_acc[a][1] += 1
            if ranked and ranked[0][0] == a:
                correct += 1
                per_author_acc[a][0] += 1

    os.makedirs(os.path.dirname(config.CLASSIFIER_FILE), exist_ok=True)
    with open(config.CLASSIFIER_FILE, "wb") as fh:
        pickle.dump(_MODEL_LOCAL, fh)
    global _MODEL
    _MODEL = _MODEL_LOCAL

    return {
        "authors": authors,
        "n_authors": len(authors),
        "train_messages": n_train,
        "test_messages": total,
        "vocab_size": len(vocab),
        "top1_accuracy": round(correct / total, 3) if total else None,
        "baseline_random": round(1 / len(authors), 3),
        "per_author_accuracy": {
            a: round(c / n, 2) for a, (c, n) in sorted(
                per_author_acc.items(), key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else 0))
            if n
        },
    }


def load():
    global _MODEL
    if _MODEL is None:
        with open(config.CLASSIFIER_FILE, "rb") as fh:
            _MODEL = pickle.load(fh)
    return _MODEL


def _classify_with(model, text, top_k=5):
    feats = features(text)
    feats = {f: n for f, n in feats.items() if f in model["vocab"]}
    if not feats:
        return []
    V = model["vocab_size"]
    alpha = model["alpha"]
    scores = {}
    for a in model["authors"]:
        ca, ta = model["counts"][a], model["totals"][a]
        denom = math.log(ta + alpha * V)
        s = model["priors"][a]
        for f, n in feats.items():
            s += n * (math.log(ca.get(f, 0) + alpha) - denom)
        scores[a] = s
    # Temperature so probabilities don't saturate to 100%/0%: NB log-scores grow
    # with input length (every n-gram adds a term), so divide by ~sqrt(#features).
    # Monotonic, so the ranking/accuracy is unchanged — only the displayed spread.
    temp = max(1.0, sum(feats.values()) ** 0.5)
    top = max(scores.values())
    exp = {a: math.exp((s - top) / temp) for a, s in scores.items()}
    z = sum(exp.values())
    probs = sorted(((a, exp[a] / z) for a in exp), key=lambda kv: -kv[1])
    return probs[:top_k]


def classify(text, top_k=5):
    """[(author, probability), ...] most-likely first, or [] if untrained."""
    try:
        return _classify_with(load(), text, top_k=top_k)
    except FileNotFoundError:
        return []
