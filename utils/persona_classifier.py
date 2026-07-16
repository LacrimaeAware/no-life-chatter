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
import time

import config
from utils import atomic_file, chat_archive, message_quality

_MODEL = None
_MODEL_STAMP = None
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def _usable(msg):
    return message_quality.usable_for_persona_exemplar(msg, max_chars=500)


def _unique_usable_messages(messages):
    """Keep one copy of each normalized line before train/test splitting."""
    out = []
    seen = set()
    for message in messages:
        if not _usable(message):
            continue
        key = chat_archive.line_match_key(message)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(message)
    return out


def _dedupe_canonical(names, max_authors=None):
    """Collapse alias accounts to one canonical label (e.g. altaccount→mainaccount),
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
    return chat_archive.canonical_author_roster(
        max_authors,
        min_messages=min_messages,
    )


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
        all_msgs = _unique_usable_messages(chat_archive.messages_for(a))
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

    # Preserve non-classifier keys (voice profiles, prevalence) — retraining
    # the who-said pipe must not wipe the ~markers/~like data living in the
    # same pickle.
    try:
        model = load()
    except (FileNotFoundError, ValueError):
        model = {}
    model.update({"pipe": pipe, "authors": list(pipe.classes_)})
    meta = dict(model.get("__meta__") or {})
    meta["classifier"] = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "authors": len(authors),
        "per_author": per_author,
        "min_messages": min_messages,
        "max_authors": max_authors,
        "test_frac": test_frac,
        "seed": seed,
        "quality_filter": "usable_for_persona_exemplar+normalized_exact_dedupe",
    }
    meta["alias_signature"] = chat_archive.alias_signature()
    model["__meta__"] = meta
    with atomic_file.open_atomic(config.CLASSIFIER_FILE, "wb") as fh:
        pickle.dump(model, fh)
    global _MODEL, _MODEL_STAMP
    _MODEL = model
    stat = os.stat(config.CLASSIFIER_FILE)
    _MODEL_STAMP = (stat.st_mtime_ns, stat.st_size)

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
    global _MODEL, _MODEL_STAMP
    stat = os.stat(config.CLASSIFIER_FILE)
    stamp = (stat.st_mtime_ns, stat.st_size)
    if _MODEL is None or stamp != _MODEL_STAMP:
        with open(config.CLASSIFIER_FILE, "rb") as fh:
            candidate = pickle.load(fh)
        meta = candidate.get("__meta__") or {}
        if meta.get("alias_signature") != chat_archive.alias_signature():
            _MODEL = None
            _MODEL_STAMP = None
            raise ValueError("stale classifier/style identity provenance")
        _MODEL = candidate
        _MODEL_STAMP = stamp
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
    collapsed = {}
    for author, prob in zip(pipe.classes_, probs):
        canon = chat_archive.normalize_author(author)
        if chat_archive._is_noise_author(canon):
            continue
        collapsed[canon] = collapsed.get(canon, 0.0) + float(prob)
    pairs = list(collapsed.items())
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
    """Archive authors (>=25 msgs) — tokens to exclude from voice profiles.
    Mentioning another chatter isn't a 'signature word', it's addressing. Names
    here are username-shaped (verified: no real-vocabulary collisions in the
    current archive), so the only 'words' lost are other people's names."""
    global _KNOWN_USERS
    if _KNOWN_USERS is None:
        conn = chat_archive.connect()
        rows = conn.execute(
            "SELECT author FROM messages GROUP BY author HAVING COUNT(*) >= 25"
        ).fetchall()
        _KNOWN_USERS = {a for a, in rows}
        _KNOWN_USERS |= {u.lower() for u in getattr(config, "EXCLUDE_USERS", set())}
    return _KNOWN_USERS


_MENTION_RE = re.compile(r"@([\w']+)")
# Invisible characters chat clients hide inside words — most commonly the
# anti-ping trick "@z<U+E0000>yrwoot", which would otherwise split into
# "z" + "yrwoot" and leak a decapitated username into voice profiles.
_INVISIBLE_RE = re.compile(
    "[​-‏⁠﻿󠀀-󠁿]")


_EMOTE_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]+$")


def _is_emote_token(tok: str) -> bool:
    """Emote-shaped: internal capital after position 0 (KEKW-as-camel like
    monkaS/PagChomp/FeelsDankMan) or an all-caps run >=3 (KEKW, LULW,
    OMEGALUL). Lowercase-typed emotes (xd, lekw) stay in the words channel —
    indistinguishable from slang, and that's fine."""
    if not _EMOTE_TOKEN_RE.match(tok):
        return False
    return bool(re.search(r"[a-z][A-Z0-9]|[A-Z][A-Z]", tok[0:]) and
                (any(c.isupper() for c in tok[1:]) or (tok.isupper() and len(tok) >= 3)))


def _count_tokens(msgs):
    """(word counts, adjacent-pair counts) for voice profiles, in one pass.
    Dropped as non-voice: URL shrapnel (https/com/youtube/status...),
    digit-bearing tokens (@usernames like name_12), known chatter usernames,
    and any token this corpus ever @-mentions — if you wrote "@somename" even
    once, every bare "somename" you typed is addressing, not vocabulary (this
    catches names that never chatted in the archive). Emotes and
    foreign-language tics are voice — they stay. Pairs ("favorite word
    associations") only form between surviving adjacent tokens."""
    from collections import Counter
    users = _known_usernames()
    words, pairs, emotes = Counter(), Counter(), Counter()
    mentioned = set()
    for m in msgs:
        raw = _URL_RE.sub(" ", _INVISIBLE_RE.sub("", m or ""))
        low = raw.lower()
        mentioned.update(_MENTION_RE.findall(low))
        # Emotes are their own channel (case-marked tokens, original casing
        # kept) — they'd otherwise dominate the words channel while ALSO
        # polluting the semantic layer. Per the favorite-emotes-matrix design.
        emote_toks = set()
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]+", raw):
            if _is_emote_token(tok) and tok.lower() not in users:
                emotes[tok] += 1
                emote_toks.add(tok.lower())
        toks = [w for w in _WORD_RE.findall(low)
                if len(w) >= 2 and not any(ch.isdigit() for ch in w)
                and w not in users and w not in emote_toks]
        words.update(toks)
        pairs.update(f"{a} {b}" for a, b in zip(toks, toks[1:]) if a != b)
    if mentioned:
        for w in mentioned:
            words.pop(w, None)
        pairs = Counter({p: c for p, c in pairs.items()
                         if not (set(p.split(" ", 1)) & mentioned)})
    return words, pairs, emotes


def _count_words(msgs):
    return _count_tokens(msgs)[0]


def strip_emote_tokens(text: str) -> str:
    """Remove emote-shaped tokens — for the semantic layer, which should embed
    meaning, not emote spam."""
    return " ".join(t for t in (text or "").split()
                    if not _is_emote_token(t.strip(".,!?\"'()")))


def _logodds_profile(ac, na, bc, nb, top, min_count=3, exclude=None,
                     prevalence=None, n_panel=0):
    """Fightin' Words log-odds of an author's word counts (ac/na) vs a shared
    background (bc/nb); keep the top `top` positive (distinctive) terms,
    L2-normalized into a {word: weight} vector. `exclude` drops terms outright.

    prevalence/n_panel: how many of the panel's chatters use each term at all.
    Raw log-odds favors high-frequency terms (more data = more statistical
    confidence), so 'ok'/'you' can outrank a person's actual catchphrases.
    Weighting by ln(panel/users) balances overuse against rarity: a term
    nearly everyone uses (>=80% of panel) is never a marker, and a term only
    1-2 people use beats a mildly-overused common one."""
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
        if z <= 0:
            continue
        if prevalence is not None and n_panel:
            users = prevalence.get(w, 1)
            if users / n_panel >= 0.8:
                continue
            z *= math.log(n_panel / users)
        scored.append((w, z))
    scored.sort(key=lambda kv: -kv[1])
    scored = scored[:top]
    norm = math.sqrt(sum(z * z for _, z in scored)) or 1.0
    return {w: z / norm for w, z in scored}


def _stops(bg_counts):
    """(stop words, stop pairs): background top-100 words are never markers;
    a pair is stopped only when BOTH halves are that common."""
    (bw, _), (bp, _), _ = bg_counts
    stop = {w for w, _ in bw.most_common(100)}
    return stop, {p for p in bp if all(t in stop for t in p.split(" ", 1))}


def _voice_profile_from_counts(aw, naw, ap, nap, bg_counts, prevalence=None,
                               words_top=300, phrases_top=150, emotes_top=100,
                               stops=None, ae=None, nae=0):
    (bw, nbw), (bp, nbp), (be, nbe) = bg_counts
    stop, stop_pairs = stops or _stops(bg_counts)
    n_panel = prevalence["n"] if prevalence else 0
    return {
        "words": _logodds_profile(
            aw, naw, bw, nbw, words_top, exclude=stop,
            prevalence=prevalence["words"] if prevalence else None, n_panel=n_panel),
        "phrases": (_logodds_profile(
            ap, nap, bp, nbp, phrases_top, exclude=stop_pairs,
            prevalence=prevalence["pairs"] if prevalence else None, n_panel=n_panel)
            if nap else {}),
        "emotes": (_logodds_profile(
            ae, nae, be, nbe, emotes_top,
            prevalence=prevalence.get("emotes") if prevalence else None,
            n_panel=n_panel)
            if ae and nae else {}),
    }


def _voice_profile(msgs, bg_counts, prevalence=None, words_top=300, phrases_top=150):
    """{'words': {...}, 'phrases': {...}} — each category capped and
    independently normalized, per the favorite-words / favorite-associations
    model: a person is their top distinctive words plus their top distinctive
    adjacent word-pairs, rarity-weighted when a prevalence panel is given."""
    aw, ap, ae = _count_tokens(msgs)
    naw, nap, nae = sum(aw.values()), sum(ap.values()), sum(ae.values())
    if naw < 500:
        return None
    return _voice_profile_from_counts(aw, naw, ap, nap, bg_counts, prevalence,
                                      words_top, phrases_top, ae=ae, nae=nae)


def _bg_counts(bg_cap=120000, channel=None):
    conn = chat_archive.connect()
    if channel:
        bg = [r[0] for r in conn.execute(
            "SELECT content FROM messages WHERE channel = ? ORDER BY RANDOM() LIMIT ?",
            (chat_archive.normalize_channel(channel), bg_cap)).fetchall()]
    else:
        bg = [r[0] for r in conn.execute(
            "SELECT content FROM messages ORDER BY RANDOM() LIMIT ?", (bg_cap,)).fetchall()]
    bw, bp, be = _count_tokens(bg)
    return (bw, sum(bw.values())), (bp, sum(bp.values())), (be, sum(be.values()))


def build_style_profiles(roster=None, words_top=300, phrases_top=150,
                         bg_cap=120000, min_messages=2000, max_roster=80):
    """Per-author voice profile for ~like / ~markers: favorite words + favorite
    word-pairs, scored by Fightin' Words log-odds vs one shared background and
    weighted by panel rarity (how few roster chatters use the term at all).

    Processes ALL of each person's archived messages (no sampling) — the
    profile itself is the cap: only the top `words_top`/`phrases_top` most
    distinctive entries per category are kept, so a 60k-message chatter and a
    3k-message chatter end up the same size. Covers a broad roster (top
    chatters by volume), not just the classifier classes."""
    from collections import Counter
    conn = chat_archive.connect()
    if roster is None:
        roster = chat_archive.canonical_author_roster(
            max_roster,
            min_messages=min_messages,
        )
    roster = _dedupe_canonical(roster)
    bg = _bg_counts(bg_cap)   # one shared background -> comparable scales
    stops = _stops(bg)

    # Pass 1: count everyone, pruned to scoreable terms (>=3 uses), so we can
    # learn each term's panel prevalence before scoring anyone.
    per_author = {}
    for a in roster:
        msgs = [m for m in chat_archive.messages_for(a) if _usable(m)]
        aw, ap, ae = _count_tokens(msgs)
        naw, nap, nae = sum(aw.values()), sum(ap.values()), sum(ae.values())
        if naw < 500:
            continue
        per_author[a] = (naw, Counter({w: c for w, c in aw.items() if c >= 3}),
                         nap, Counter({p: c for p, c in ap.items() if c >= 3}),
                         nae, Counter({e: c for e, c in ae.items() if c >= 3}))
    n_panel = len(per_author)
    wprev, pprev, eprev = Counter(), Counter(), Counter()
    for naw, aw, nap, ap, nae, ae in per_author.values():
        wprev.update(aw.keys())
        pprev.update(ap.keys())
        eprev.update(ae.keys())
    # store only >=2-user terms; a missing term means "1 user" (max rarity)
    prevalence = {"words": {w: c for w, c in wprev.items() if c >= 2},
                  "pairs": {p: c for p, c in pprev.items() if c >= 2},
                  "emotes": {e: c for e, c in eprev.items() if c >= 2},
                  "n": n_panel}

    # Pass 2: score each author with rarity weighting.
    profiles = {}
    for a, (naw, aw, nap, ap, nae, ae) in per_author.items():
        profiles[a] = _voice_profile_from_counts(
            aw, naw, ap, nap, bg, prevalence, words_top, phrases_top,
            stops=stops, ae=ae, nae=nae)
    model = load()
    model["profiles"] = profiles
    model["prevalence"] = prevalence
    meta = dict(model.get("__meta__") or {})
    meta["style_profiles"] = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profiles": len(profiles),
        "words_top": words_top,
        "phrases_top": phrases_top,
        "bg_cap": bg_cap,
        "min_messages": min_messages,
        "max_roster": max_roster,
        "quality_filter": "message_quality.usable_for_persona_exemplar",
    }
    meta["alias_signature"] = chat_archive.alias_signature()
    model["__meta__"] = meta
    model.pop("style", None)
    model.pop("centroids", None)
    with atomic_file.open_atomic(config.CLASSIFIER_FILE, "wb") as fh:
        pickle.dump(model, fh)
    global _MODEL, _MODEL_STAMP
    _MODEL = model
    stat = os.stat(config.CLASSIFIER_FILE)
    _MODEL_STAMP = (stat.st_mtime_ns, stat.st_size)
    return len(profiles)


_scoped_cache = {}


def profile_for(author, channel=None, year=None, author_cap=20000):
    """The voice profile for `author`. Stored full-history profile by default;
    with `channel` and/or `year`, computed live from just those messages,
    against the (channel-scoped) background — 'my markers are polluted with
    what I spammed five years ago in another community' is what this scopes
    away."""
    model = load()
    profiles = model.get("profiles") or {}
    canon = chat_archive.normalize_author(author)
    if not channel and not year:
        for key in [canon, *chat_archive.author_keys(canon)]:
            prof = profiles.get(key)
            if prof and "words" in prof:
                return prof
    key = (canon, chat_archive.normalize_channel(channel) if channel else None, year)
    if key in _scoped_cache:
        return _scoped_cache[key]
    msgs = [m for m in chat_archive.messages_for(canon, channel=channel, year=year)
            if _usable(m)]
    if not msgs:
        return None
    rng = random.Random(13)
    rng.shuffle(msgs)
    prof = _voice_profile(msgs[:author_cap], _bg_counts(40000, channel=channel),
                          prevalence=model.get("prevalence"))
    if len(_scoped_cache) > 200:
        _scoped_cache.clear()
    _scoped_cache[key] = prof
    return prof


def most_like(author, n=6, channel=None, year=None):
    """Chatters who share `author`'s distinctive voice — overlap of favorite
    words (60%) and favorite word-pairs (40%). Returns
    [(author, score, [shared markers]), ...] with pairs preferred as the shown
    evidence since they read as topics, not tics. `channel` scopes the
    TARGET's profile to one chat (the panel stays full-history)."""
    model = load()
    profiles = model.get("profiles")
    if not profiles:
        return []
    canon = chat_archive.normalize_author(author)
    target = profile_for(canon, channel=channel, year=year)
    if not target:
        return []
    sims = {}
    for c, prof in profiles.items():
        c_canon = chat_archive.normalize_author(c)
        if c_canon == canon or chat_archive._is_noise_author(c_canon) or "words" not in prof:
            continue
        shared_w = [(k, target["words"][k] * prof["words"][k])
                    for k in target["words"] if k in prof["words"]]
        shared_p = [(k, target["phrases"][k] * prof["phrases"][k])
                    for k in target.get("phrases", {}) if k in prof.get("phrases", {})]
        te, pe = target.get("emotes", {}), prof.get("emotes", {})
        shared_e = [(k, te[k] * pe[k]) for k in te if k in pe]
        score = (0.45 * sum(v for _, v in shared_w)
                 + 0.25 * sum(v for _, v in shared_p)
                 + 0.30 * sum(v for _, v in shared_e))
        shared_p.sort(key=lambda kv: -kv[1])
        shared_w.sort(key=lambda kv: -kv[1])
        # don't show a pair AND its own word ("hello emote · emote")
        shared_e.sort(key=lambda kv: -kv[1])
        pair_ev = [k for k, _ in shared_p[:2]]
        in_pairs = {t for p in pair_ev for t in p.split()}
        word_ev = [k for k, _ in shared_w if k not in in_pairs][:3]
        emote_ev = [k for k, _ in shared_e[:2]]
        row = (c_canon, score, (pair_ev + emote_ev + word_ev)[:5])
        if c_canon not in sims or score > sims[c_canon][1]:
            sims[c_canon] = row
    ranked = sorted(sims.values(), key=lambda kv: -kv[1])
    return ranked[:n]


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
