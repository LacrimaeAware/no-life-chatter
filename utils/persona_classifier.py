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


_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_KNOWN_USERS = None


def _known_usernames():
    """Archive authors (>=1000 msgs) — tokens to exclude from voice profiles.
    Mentioning another chatter isn't a 'signature word', it's addressing. Names
    here are username-shaped, so no real-vocabulary collateral."""
    global _KNOWN_USERS
    if _KNOWN_USERS is None:
        conn = chat_archive.connect()
        rows = conn.execute(
            "SELECT author FROM messages GROUP BY author HAVING COUNT(*) >= 1000"
        ).fetchall()
        _KNOWN_USERS = {a for a, in rows}
        _KNOWN_USERS |= {u.lower() for u in getattr(config, "EXCLUDE_USERS", set())}
    return _KNOWN_USERS


def _count_tokens(msgs):
    """(word counts, adjacent-pair counts) for voice profiles, in one pass.
    Three classes of non-voice tokens are dropped: URL shrapnel
    (https/com/youtube/status...), digit-bearing tokens (@usernames like
    name_12), and known chatter usernames (addressing, not vocabulary).
    Emotes and foreign-language tics are voice — they stay. Pairs ("favorite
    word associations") only form between surviving adjacent tokens."""
    from collections import Counter
    users = _known_usernames()
    words, pairs = Counter(), Counter()
    for m in msgs:
        toks = [w for w in _WORD_RE.findall(_URL_RE.sub(" ", (m or "")).lower())
                if len(w) >= 2 and not any(ch.isdigit() for ch in w) and w not in users]
        words.update(toks)
        pairs.update(f"{a} {b}" for a, b in zip(toks, toks[1:]) if a != b)
    return words, pairs


def _count_words(msgs):
    return _count_tokens(msgs)[0]


def _logodds_profile(ac, na, bc, nb, top, min_count=3, exclude=None):
    """Fightin' Words log-odds of an author's word counts (ac/na) vs a shared
    background (bc/nb); keep the top `top` positive (distinctive) terms,
    L2-normalized into a {word: weight} vector. `exclude` drops terms outright
    (the everyone-says-it stoplist — 'to'/'and' can't be a voice marker no
    matter how confidently overused)."""
    import math
    scored = []
    for w, ya in ac.items():
        if ya < min_count or (exclude and w in exclude):
            continue
        yb = bc.get(w, 0)
        num_a, den_a = ya + 0.5, na - ya + 0.5
        num_b, den_b = yb + 0.5, nb - yb + 0.5
        z = (math.log(num_a / den_a) - math.log(num_b / den_b)) / \
            math.sqrt(1.0 / num_a + 1.0 / num_b)
        if z > 0:
            scored.append((w, z))
    scored.sort(key=lambda kv: -kv[1])
    scored = scored[:top]
    norm = math.sqrt(sum(z * z for _, z in scored)) or 1.0
    return {w: z / norm for w, z in scored}


def _voice_profile(msgs, bg_counts, words_top=300, phrases_top=150):
    """{'words': {...}, 'phrases': {...}} — each category capped and
    independently normalized, per the favorite-words / favorite-associations
    model: a person is their top distinctive words plus their top distinctive
    adjacent word-pairs. Words everyone says (background top-100) are never
    markers; a pair is dropped only when BOTH halves are that common."""
    (bw, nbw), (bp, nbp) = bg_counts
    stop = {w for w, _ in bw.most_common(100)}
    stop_pairs = {p for p in bp if all(t in stop for t in p.split(" ", 1))}
    aw, ap = _count_tokens(msgs)
    naw, nap = sum(aw.values()), sum(ap.values())
    if naw < 500:
        return None
    return {
        "words": _logodds_profile(aw, naw, bw, nbw, words_top, exclude=stop),
        "phrases": (_logodds_profile(ap, nap, bp, nbp, phrases_top, exclude=stop_pairs)
                    if nap else {}),
    }


def _bg_counts(bg_cap=120000):
    conn = chat_archive.connect()
    bg = [r[0] for r in conn.execute(
        "SELECT content FROM messages ORDER BY RANDOM() LIMIT ?", (bg_cap,)).fetchall()]
    bw, bp = _count_tokens(bg)
    return (bw, sum(bw.values())), (bp, sum(bp.values()))


def build_style_profiles(roster=None, words_top=300, phrases_top=150,
                         bg_cap=120000, min_messages=2000, max_roster=80):
    """Per-author voice profile for ~like / ~markers: favorite words + favorite
    word-pairs, scored by Fightin' Words log-odds vs one shared background.

    Processes ALL of each person's archived messages (no sampling) — the
    profile itself is the cap: only the top `words_top`/`phrases_top` most
    distinctive entries per category are kept, so a 60k-message chatter and a
    3k-message chatter end up the same size. Covers a broad roster (top
    chatters by volume), not just the classifier classes."""
    conn = chat_archive.connect()
    if roster is None:
        exclude = {u.lower() for u in getattr(config, "EXCLUDE_USERS", set())}
        rows = conn.execute(
            "SELECT author, COUNT(*) c FROM messages GROUP BY author "
            "HAVING c >= ? ORDER BY c DESC LIMIT ?", (min_messages, max_roster * 2)).fetchall()
        roster = [a for a, _ in rows if a not in exclude and "bot" not in a][:max_roster]
    roster = _dedupe_canonical(roster)
    bg = _bg_counts(bg_cap)   # one shared background -> comparable scales
    profiles = {}
    for a in roster:
        msgs = [m for m in chat_archive.messages_for(a) if _usable(m)]
        prof = _voice_profile(msgs, bg, words_top, phrases_top)
        if prof:
            profiles[a] = prof
    model = load()
    model["profiles"] = profiles
    model.pop("style", None)
    model.pop("centroids", None)
    with open(config.CLASSIFIER_FILE, "wb") as fh:
        pickle.dump(model, fh)
    global _MODEL
    _MODEL = model
    return len(profiles)


def profile_for(author, author_cap=20000):
    """The stored voice profile for `author`, or a live-computed one (capped
    sample, fresh background) for chatters outside the prebuilt roster."""
    model = load()
    profiles = model.get("profiles") or {}
    canon = chat_archive.normalize_author(author)
    prof = profiles.get(canon)
    if prof and "words" in prof:
        return prof
    msgs = [m for m in chat_archive.messages_for(canon) if _usable(m)]
    if not msgs:
        return None
    rng = random.Random(13)
    rng.shuffle(msgs)
    return _voice_profile(msgs[:author_cap], _bg_counts(40000))


def most_like(author, n=6):
    """Chatters who share `author`'s distinctive voice — overlap of favorite
    words (60%) and favorite word-pairs (40%). Returns
    [(author, score, [shared markers]), ...] with pairs preferred as the shown
    evidence since they read as topics, not tics."""
    model = load()
    profiles = model.get("profiles")
    if not profiles:
        return []
    canon = chat_archive.normalize_author(author)
    target = profile_for(canon)
    if not target:
        return []
    sims = []
    for c, prof in profiles.items():
        if c == canon or "words" not in prof:
            continue
        shared_w = [(k, target["words"][k] * prof["words"][k])
                    for k in target["words"] if k in prof["words"]]
        shared_p = [(k, target["phrases"][k] * prof["phrases"][k])
                    for k in target.get("phrases", {}) if k in prof.get("phrases", {})]
        score = 0.6 * sum(v for _, v in shared_w) + 0.4 * sum(v for _, v in shared_p)
        shared_p.sort(key=lambda kv: -kv[1])
        shared_w.sort(key=lambda kv: -kv[1])
        evidence = [k for k, _ in shared_p[:2]] + [k for k, _ in shared_w[:4]]
        sims.append((c, score, evidence[:5]))
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

    ac, bc = _count_words(amsgs), _count_words(bg)
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
