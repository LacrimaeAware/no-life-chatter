"""Authorship classifier: given any text, who's most likely to have said it.

TF-IDF over character n-grams (style: casing, spelling, emotes, punctuation
rhythm) + word n-grams (vocabulary/topic), into a multinomial logistic
regression. This is the standard, strong authorship-attribution pipeline.

Why TF-IDF + LR over the old Naive Bayes: NB had to cap the vocabulary at the
most *common* features, so a rare-but-distinctive word (e.g. "simmons" for
poggerooskii) got dropped entirely and couldn't influence the guess. TF-IDF
keeps rare terms and weights them by how distinctive they are, and LR learns a
per-author weight for each — so a single signature word can carry the call.

Two uses: the `~whosaid` command, and an objective persona metric (run a
generated persona line through `classify()` and see if it reads as the target).
LR `predict_proba` gives reasonable probabilities — a real ranking, treat the
numbers as confidence.
"""

import os
import pickle
import random

import config
from utils import chat_archive

_MODEL = None


def _usable(msg):
    return bool(msg and len(msg.split()) >= 2 and not msg.lstrip().startswith(config.PREFIX))


def _dedupe_canonical(names, max_authors=None):
    """Collapse alias accounts to one canonical label (e.g. fernardo→earnest),
    preserving order. Without this, an aliased account becomes a second class
    trained on the SAME merged messages — splitting its own votes."""
    seen, out = set(), []
    for n in names:
        canon = chat_archive.normalize_author(n)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
        if max_authors and len(out) >= max_authors:
            break
    return out


def _pick_authors(conn, authors, min_messages, max_authors):
    if authors:
        return _dedupe_canonical(authors)
    exclude = {u.lower() for u in getattr(config, "EXCLUDE_USERS", set())}
    rows = conn.execute(
        "SELECT author, COUNT(*) c FROM messages GROUP BY author "
        "HAVING c >= ? ORDER BY c DESC", (min_messages,)
    ).fetchall()
    return _dedupe_canonical((a for a, _ in rows if a not in exclude), max_authors)


def _build_pipeline(char_max=200000, word_max=60000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                           lowercase=False, sublinear_tf=True, max_features=char_max)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True, max_features=word_max)
    feats = FeatureUnion([("char", char), ("word", word)])
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")
    return Pipeline([("feats", feats), ("clf", clf)])


def train(authors=None, per_author=3000, test_frac=0.1, seed=1337,
          min_messages=300, max_authors=24, **_):
    """Train and persist the classifier. Returns a held-out accuracy report."""
    from collections import Counter
    rng = random.Random(seed)
    conn = chat_archive.connect()
    authors = _pick_authors(conn, authors, min_messages, max_authors)

    tr_X, tr_y, te_X, te_y = [], [], [], []
    for a in authors:
        msgs = [m for m in chat_archive.messages_for(a) if _usable(m)]
        rng.shuffle(msgs)
        msgs = msgs[:per_author]
        cut = int(len(msgs) * (1 - test_frac))
        for m in msgs[:cut]:
            tr_X.append(m); tr_y.append(a)
        for m in msgs[cut:]:
            te_X.append(m); te_y.append(a)

    pipe = _build_pipeline()
    pipe.fit(tr_X, tr_y)

    preds = pipe.predict(te_X)
    correct = sum(1 for p, y in zip(preds, te_y) if p == y)
    per_author = {a: [0, 0] for a in authors}
    for p, y in zip(preds, te_y):
        per_author[y][1] += 1
        if p == y:
            per_author[y][0] += 1

    model = {"pipe": pipe, "authors": list(pipe.classes_)}
    os.makedirs(os.path.dirname(config.CLASSIFIER_FILE), exist_ok=True)
    with open(config.CLASSIFIER_FILE, "wb") as fh:
        pickle.dump(model, fh)
    global _MODEL
    _MODEL = model

    return {
        "n_authors": len(authors),
        "train_messages": len(tr_X),
        "test_messages": len(te_X),
        "top1_accuracy": round(correct / len(te_X), 3) if te_X else None,
        "baseline_random": round(1 / len(authors), 3) if authors else None,
        "per_author_accuracy": {
            a: round(c / n, 2) for a, (c, n) in sorted(
                per_author.items(),
                key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else 0)) if n
        },
    }


def load():
    global _MODEL
    if _MODEL is None:
        with open(config.CLASSIFIER_FILE, "rb") as fh:
            _MODEL = pickle.load(fh)
    return _MODEL


def classify(text, top_k=5):
    """[(author, probability), ...] most-likely first, or [] if unusable/untrained."""
    if not text or not text.strip():
        return []
    try:
        model = load()
    except FileNotFoundError:
        return []
    pipe = model["pipe"]
    probs = pipe.predict_proba([text])[0]
    ranked = sorted(zip(pipe.classes_, probs), key=lambda kv: -kv[1])
    return [(a, float(p)) for a, p in ranked[:top_k]]


def top_signature_terms(author, n=15):
    """The word features the LR weights most toward this author — a peek at what
    the model thinks is distinctively them (debugging / fun)."""
    model = load()
    pipe = model["pipe"]
    if author not in list(pipe.classes_):
        return []
    import numpy as np
    clf = pipe.named_steps["clf"]
    feats = pipe.named_steps["feats"]
    names = feats.get_feature_names_out()
    idx = list(pipe.classes_).index(author)
    coefs = clf.coef_[idx]
    word_mask = [i for i, nm in enumerate(names) if nm.startswith("word__")]
    top = sorted(word_mask, key=lambda i: -coefs[i])[:n]
    return [(names[i].replace("word__", ""), round(float(coefs[i]), 2)) for i in top]
