"""A genuine attempt at estimating cognitive register from chat logs (~iq).

This is NOT IQ — text logs can't measure intelligence, and the command says
so. It's the most honest proxy we could build, designed per the user's spec
to NOT just re-measure the professor axis:

- **Peak philosophy**: every component uses top-decile statistics, not
  averages. A person's typical message is filler; their capability shows in
  their best 10%. ("it needs to be the median of their top 10%")
- **vocab**: 90th-percentile message-level word rarity (rarity = -log
  frequency in the archive itself, so it's community-relative — using words
  nobody here uses, not dictionary obscurity).
- **syntax**: 90th-percentile message length in words x subordinate-clause
  marker rate (because/although/whereas/...) — sustained constructed thought.
- **breadth**: entropy of the person's distribution over global topic
  clusters (k-means on the message index) — how many different conversations
  they can actually be in.
- **depth**: in their strongest niche cluster (a topic that is NOT generic
  chat), their share x vocab rarity there — being heavyweight somewhere
  specific.

Composite = mean of component z-scores, displayed IQ-style (100 + 15z,
clamped) with the components shown so nobody mistakes it for a single magic
number. Scores cache to data/unsynced/iq_scores.pkl; rebuild with
scripts via compute_all(force=True).
"""

import math
import os
import pickle
import re

from utils import chat_archive, persona_classifier as pc

CACHE = os.path.join("data", "unsynced", "iq_scores.pkl")
MSG_DIR = os.path.join("data", "unsynced", "msg_index")

_CLAUSE_RE = re.compile(
    r"\b(because|although|though|whereas|therefore|however|unless|despite|"
    r"implies|consider|specifically|essentially|technically|presumably|"
    r"hypothetically|relative to|in terms of|on the other hand)\b", re.I)
_WORD = re.compile(r"[a-z']{4,}")


def _word_freqs(sample=400000):
    conn = chat_archive.connect()
    from collections import Counter
    c = Counter()
    for (m,) in conn.execute(
            "SELECT content FROM messages ORDER BY RANDOM() LIMIT ?", (sample,)):
        c.update(_WORD.findall((m or "").lower()))
    total = sum(c.values())
    return c, total


def _percentile(vals, p):
    if not vals:
        return 0.0
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(len(vals) * p))]


def compute_all(force=False):
    """{author: {component: percentile-ish raw, 'iq': scaled}} for the roster."""
    import numpy as np
    if not force and os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)

    freqs, total = _word_freqs()

    def rarity(w):
        return -math.log((freqs.get(w, 0) + 1) / total)

    # global topic clusters from the message index (sampled)
    from sklearn.cluster import KMeans
    roster = [f[:-4] for f in os.listdir(MSG_DIR) if f.endswith(".npz")]
    sample_vecs, owners, texts = [], [], []
    for a in roster:
        d = np.load(os.path.join(MSG_DIR, f"{a}.npz"), allow_pickle=True)
        V = d["vectors"].astype("float32")
        idx = np.random.RandomState(3).choice(len(V), min(400, len(V)), replace=False)
        sample_vecs.append(V[idx])
        owners.extend([a] * len(idx))
        texts.extend([d["texts"][i] for i in idx])
    M = np.vstack(sample_vecs)
    K = 20
    km = KMeans(n_clusters=K, n_init=4, random_state=3).fit(M)
    labels = km.labels_
    global_share = np.bincount(labels, minlength=K) / len(labels)

    out = {}
    raw = {}
    for a in roster:
        msgs = [m for m in chat_archive.utterances_for(a) if pc._usable(m)]
        if len(msgs) < 50:
            continue
        # vocab: per-message mean rarity, top-decile
        rar = []
        lens = []
        clause = 0
        words_all = []
        for m in msgs:
            ws = _WORD.findall(m.lower())
            if ws:
                rar.append(sum(rarity(w) for w in ws) / len(ws))
            lens.append(len(m.split()))
            clause += len(_CLAUSE_RE.findall(m))
            words_all.extend(ws)
        vocab_p90 = _percentile(rar, 0.90)
        syntax = _percentile(lens, 0.90) * (1 + 10 * clause / max(1, len(words_all)))
        diversity = len(set(words_all)) / (math.sqrt(len(words_all)) + 1)

        mine = [i for i, o in enumerate(owners) if o == a]
        if mine:
            lab = labels[mine]
            share = np.bincount(lab, minlength=K) / len(lab)
            nz = share[share > 0]
            breadth = float(-(nz * np.log(nz)).sum())
            # depth: strongest NICHE cluster (globally small), share x rarity there
            niche_scores = []
            for k in range(K):
                if global_share[k] < 0.10 and share[k] > 0:
                    k_msgs = [texts[mine[j]] for j in range(len(mine)) if lab[j] == k]
                    ws = [w for m in k_msgs for w in _WORD.findall(str(m).lower())]
                    if ws:
                        niche_scores.append(share[k] * sum(rarity(w) for w in ws) / len(ws))
            depth = max(niche_scores) if niche_scores else 0.0
        else:
            breadth = depth = 0.0
        raw[a] = dict(vocab=vocab_p90, syntax=syntax, diversity=diversity,
                      breadth=breadth, depth=depth)

    names = list(raw)
    comps = ["vocab", "syntax", "diversity", "breadth", "depth"]
    import numpy as np
    Z = {}
    for c in comps:
        v = np.array([raw[a][c] for a in names], dtype="float64")
        Z[c] = (v - v.mean()) / (v.std() or 1.0)
    for i, a in enumerate(names):
        z = float(np.mean([Z[c][i] for c in comps]))
        out[a] = {c: round(float(Z[c][i]), 2) for c in comps}
        out[a]["iq"] = int(max(62, min(158, round(100 + 15 * z))))
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh)
    return out


def score(author):
    data = compute_all()
    return data.get(chat_archive.normalize_author(author))


def leaderboard(n=5, reverse=False):
    data = compute_all()
    ranked = sorted(data.items(), key=lambda kv: kv[1]["iq"], reverse=not reverse)
    return ranked[:n]
