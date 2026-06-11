"""Authorship classifier: given any text, who's most likely to have said it.

TF-IDF over character n-grams (style: casing, spelling, emotes, punctuation
rhythm) + word n-grams (vocabulary/topic), into a multinomial logistic
regression. This is the standard, strong authorship-attribution pipeline.

Why TF-IDF + LR over the old Naive Bayes: NB had to cap the vocabulary at the
most *common* features, so a rare-but-distinctive word (a niche term one chatter
uses constantly) got dropped entirely and couldn't influence the guess. TF-IDF
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
import re

import config
from utils import chat_archive

_MODEL = None
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


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


def _build_pipeline(char_max=200000, word_max=60000, verbose=False):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                           lowercase=False, sublinear_tf=True, max_features=char_max)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True, max_features=word_max)
    feats = FeatureUnion([("char", char), ("word", word)])
    # verbose=1 makes lbfgs print its iterations so training isn't a black box.
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced",
                             verbose=1 if verbose else 0)
    return Pipeline([("feats", feats), ("clf", clf)])


def train(authors=None, per_author=3000, test_frac=0.1, seed=1337,
          min_messages=300, max_authors=24, **_):
    """Train and persist the classifier. Returns a held-out accuracy report."""
    import time
    from collections import Counter
    rng = random.Random(seed)
    conn = chat_archive.connect()
    authors = _pick_authors(conn, authors, min_messages, max_authors)
    print(f"[1/4] loading messages for {len(authors)} authors (cap {per_author} each)...", flush=True)

    t0 = time.time()
    tr_X, tr_y, te_X, te_y = [], [], [], []
    for i, a in enumerate(authors, 1):
        all_msgs = [m for m in chat_archive.messages_for(a) if _usable(m)]
        rng.shuffle(all_msgs)
        msgs = all_msgs[:per_author]
        cut = int(len(msgs) * (1 - test_frac))
        for m in msgs[:cut]:
            tr_X.append(m); tr_y.append(a)
        for m in msgs[cut:]:
            te_X.append(m); te_y.append(a)
        print(f"   ({i}/{len(authors)}) {a}: using {len(msgs)} of {len(all_msgs):,}", flush=True)

    print(f"[2/4] {len(tr_X):,} train / {len(te_X):,} test messages loaded in "
          f"{time.time()-t0:.0f}s. building char+word TF-IDF features...", flush=True)
    pipe = _build_pipeline(verbose=True)
    print("[3/4] fitting logistic regression (lbfgs iterations print below; "
          "this is the slow part, usually a few minutes)...", flush=True)
    t1 = time.time()
    pipe.fit(tr_X, tr_y)
    print(f"[3/4] fit done in {time.time()-t1:.0f}s. evaluating on held-out...", flush=True)

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


def classify(text, top_k=5, restrict_to=None):
    """[(author, probability), ...] most-likely first, or [] if unusable/untrained.

    restrict_to: optional iterable of authors to consider (e.g. the people
    currently in a channel) — others are dropped and the probabilities
    renormalized among the rest, so ~whosaid only names chatters who are here.
    """
    if not text or not text.strip():
        return []
    try:
        model = load()
    except FileNotFoundError:
        return []
    pipe = model["pipe"]
    probs = pipe.predict_proba([text])[0]
    pairs = list(zip(pipe.classes_, probs))
    if restrict_to is not None:
        keep = {chat_archive.normalize_author(a) for a in restrict_to}
        pairs = [(a, p) for a, p in pairs if a in keep]
        z = sum(p for _, p in pairs) or 1.0
        pairs = [(a, p / z) for a, p in pairs]
    ranked = sorted(pairs, key=lambda kv: -kv[1])
    return [(a, float(p)) for a, p in ranked[:top_k]]


def build_centroids(per_author=2500, seed=7):
    """Precompute each author's mean (L2-normalized) TF-IDF vector and store it
    in the model pickle. Cosine between two centroids = how alike two people
    write — the basis for ~like. Run once after training (or it's recomputed
    by train()). Stored float32 to keep the pickle reasonable."""
    import numpy as np
    model = load()
    feats = model["pipe"].named_steps["feats"]
    rng = random.Random(seed)
    cents = {}
    for a in model["pipe"].classes_:
        msgs = [m for m in chat_archive.messages_for(a) if _usable(m)]
        rng.shuffle(msgs)
        msgs = msgs[:per_author]
        if not msgs:
            continue
        v = np.asarray(feats.transform(msgs).mean(axis=0)).ravel()
        cents[a] = (v / (np.linalg.norm(v) + 1e-9)).astype("float32")
    model["centroids"] = cents
    with open(config.CLASSIFIER_FILE, "wb") as fh:
        pickle.dump(model, fh)
    global _MODEL
    _MODEL = model
    return len(cents)


def most_like(author, n=6):
    """Chatters who write most like `author` — cosine similarity of mean
    TF-IDF style vectors (0..1; higher = more alike). Near-twins are likely
    alts. Only covers authors in the trained classifier."""
    model = load()
    cents = model.get("centroids")
    canon = chat_archive.normalize_author(author)
    if not cents or canon not in cents:
        return []
    v = cents[canon]
    sims = [(c, float(v.dot(w))) for c, w in cents.items() if c != canon]
    sims.sort(key=lambda kv: -kv[1])
    return sims[:n]


def signature_words(author, n=12, author_cap=5000, bg_cap=40000, min_count=3, seed=13):
    """Most distinctive words for ANY author (not just classifier classes).

    Log-odds-ratio of the author's word use vs a random background sample
    (Fightin' Words). Works for anyone in the archive — the reverse of
    ~whosaid. Reveals e.g. that a bilingual chatter's signature is German.
    Returns [(word, z), ...] highest-z first.
    """
    import math
    import random as _random
    from collections import Counter
    conn = chat_archive.connect()
    amsgs = chat_archive.messages_for(author)
    if not amsgs:
        return []
    rng = _random.Random(seed)
    rng.shuffle(amsgs)
    amsgs = amsgs[:author_cap]
    bg = [r[0] for r in conn.execute(
        "SELECT content FROM messages ORDER BY RANDOM() LIMIT ?", (bg_cap,)).fetchall()]

    def counts(msgs):
        c = Counter()
        for m in msgs:
            for w in _WORD_RE.findall((m or "").lower()):
                # drop digit-bearing tokens (mostly @usernames like name_12 /
                # user99) so markers are actual vocabulary, not who they ping
                if len(w) >= 2 and not any(ch.isdigit() for ch in w):
                    c[w] += 1
        return c

    ac, bc = counts(amsgs), counts(bg)
    na, nb = sum(ac.values()), sum(bc.values())
    if na == 0 or nb == 0:
        return []
    scored = []
    for w, ya in ac.items():
        if ya < min_count:
            continue
        yb = bc.get(w, 0)
        num_a, den_a = ya + 0.5, na - ya + 0.5
        num_b, den_b = yb + 0.5, nb - yb + 0.5
        z = (math.log(num_a / den_a) - math.log(num_b / den_b)) / \
            math.sqrt(1.0 / num_a + 1.0 / num_b)
        scored.append((w, z))
    scored.sort(key=lambda kv: -kv[1])
    return scored[:n]


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
