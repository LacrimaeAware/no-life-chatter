"""Does a message 'land' because of its CONTENT, or is it noise? — the dial for
the 'did-it-land value model' idea (persona quality = emulate the lines that
actually got a reaction, like a win-rate model emulates winning moves).

Method (deliberately simple + skeptical, per structured-transform-discovery):
  1. Label real messages: "landed" = it sparked a FRESH laugh from another user
     in the next 30s (the same net-spark signal ~funny uses; not already laughing
     before, self-laughs excluded, bots/commands excluded).
  2. Balance landed vs not-landed, embed both with bge-m3, train/test split.
  3. Build a "landing axis" = mean(landed) − mean(not-landed) on TRAIN only, score
     held-out messages by projection, report AUC.
  4. Benchmark against dumb baselines (length, caps, emotes, '@', '?').

If the landing axis doesn't beat the dumb baselines on held-out data, the value
model isn't worth building from content alone — and THAT is a useful result.

Run: .venv/Scripts/python.exe scripts/landing_probe.py   (needs LM Studio up)
"""
import json
import os
import sys
import urllib.request
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from utils import chat_archive  # noqa: E402
from utils import comedic_influence as ci  # noqa: E402

CHANNELS = ci.DEFAULT_CHANNELS   # conversational chats from gitignored config
MAX_PER_CLASS = 2000          # cap embeddings (cost); balanced classes
RNG = np.random.default_rng(7)


def labeled_messages(channel):
    """[(text, landed)] for eligible setup messages in one chat."""
    conn = chat_archive.connect()
    rows = conn.execute(
        "SELECT sent_at, author, content FROM messages WHERE channel=? ORDER BY sent_at, id",
        (chat_archive.normalize_channel(channel),),
    ).fetchall()
    n = len(rows)
    times = [ci._secs(r[0]) for r in rows]
    norm, noise = {}, {}
    auth = []
    for _s, a, _c in rows:
        if a not in norm:
            norm[a] = chat_archive.normalize_author(a)
        auth.append(norm[a])
    laugh = [ci._is_laugh(r[2]) for r in rows]
    out = []
    dq = deque()
    for i in range(n):
        ti = times[i]
        while dq and dq[0][0] < ti - ci.WINDOW_SECS:
            dq.popleft()
        ai = auth[i]
        content = rows[i][2]
        is_noise = noise.get(ai)
        if is_noise is None:
            is_noise = noise[ai] = chat_archive._is_noise_author(ai)
        if not laugh[i] and not ci._is_command(content) and not is_noise:
            after, others, cnt, j = set(), False, 0, i + 1
            while j < n and times[j] <= ti + ci.WINDOW_SECS and cnt < ci.FWD_MSG_CAP:
                aj = auth[j]
                if aj != ai:
                    others = True
                    if laugh[j]:
                        after.add(aj)
                        if len(after) >= ci.AFTER_CAP:
                            break
                j += 1
                cnt += 1
            if others:
                before = {a for (_t, a) in dq if a != ai}
                out.append((content, 1 if (after - before) else 0))
        if laugh[i]:
            dq.append((ti, ai))
    return out


def embed_batch(texts, batch=64):
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    out = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": config.LLM_EMBED_MODEL, "input": texts[i:i + batch]}).encode()
        req = urllib.request.Request(base + "/v1/embeddings", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            for e in json.load(r)["data"]:
                v = np.asarray(e["embedding"], dtype="float32")
                out.append(v / (np.linalg.norm(v) + 1e-9))
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.array(out)


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return 0.5
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def logreg_auc(Xtr, ytr, Xte, yte, l2=2.0, iters=400, step=0.5):
    """Full-embedding L2 logistic regression (numpy, no sklearn) — the strongest
    fair linear shot, so a weak result can't be blamed on the single-axis model."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    w = np.zeros(A.shape[1])
    b = 0.0
    y = ytr.astype(float)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(A @ w + b)))
        g = p - y
        w -= step * (A.T @ g / len(y) + l2 * w / len(y))
        b -= step * g.mean()
    return auc(B @ w + b, yte)


def baseline_feats(text):
    t = text or ""
    toks = t.split()
    caps = sum(c.isupper() for c in t)
    letters = sum(c.isalpha() for c in t) or 1
    emote_ish = sum(1 for w in toks if any(c.isupper() for c in w) and any(c.islower() for c in w))
    return {
        "length": len(t),
        "words": len(toks),
        "caps_ratio": caps / letters,
        "emote_ish": emote_ish,
        "has_@": 1.0 if "@" in t else 0.0,
        "has_?": 1.0 if t.rstrip().endswith("?") else 0.0,
    }


def main():
    print("labeling messages by laugh-spark ...")
    data = []
    for ch in CHANNELS:
        data += labeled_messages(ch)
    landed = [t for t, y in data if y == 1]
    notl = [t for t, y in data if y == 0]
    print(f"  total eligible: {len(data)} | landed: {len(landed)} "
          f"({len(landed)/max(1,len(data)):.1%}) | not-landed: {len(notl)}")

    k = min(MAX_PER_CLASS, len(landed), len(notl))
    landed = list(RNG.permutation(landed))[:k]
    notl = list(RNG.permutation(notl))[:k]
    texts = landed + notl
    labels = np.array([1] * k + [0] * k)
    print(f"balanced sample: {k} per class ({2*k} total). embedding ...")
    X = embed_batch(texts)

    # train/test split
    idx = RNG.permutation(2 * k)
    cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    ytr, yte = labels[tr], labels[te]

    # landing axis from the TRAIN contrast
    axis = X[tr][ytr == 1].mean(0) - X[tr][ytr == 0].mean(0)
    axis /= np.linalg.norm(axis) + 1e-9
    emb_auc = auc(X[te] @ axis, yte)
    lr_auc = logreg_auc(X[tr], ytr, X[te], yte)

    # baselines on the same test split
    feats = [baseline_feats(t) for t in texts]
    base_aucs = {}
    for name in feats[0]:
        col = np.array([f[name] for f in feats])
        a = auc(col[te], yte)
        base_aucs[name] = max(a, 1 - a)  # a feature predictive either direction

    print("\n=== held-out AUC (0.5 = coin flip) ===")
    print(f"  EMBEDDING landing-axis  : {emb_auc:.3f}")
    print(f"  EMBEDDING logistic-reg  : {lr_auc:.3f}")
    for name, a in sorted(base_aucs.items(), key=lambda kv: -kv[1]):
        print(f"  baseline {name:11}: {a:.3f}")
    best_base = max(base_aucs.values())
    best_emb = max(emb_auc, lr_auc)
    print(f"\n  best embedding model: {best_emb:.3f} | best dumb baseline: {best_base:.3f} "
          f"({best_emb - best_base:+.3f})")
    print("  VERDICT:", "content carries real signal — value model worth building"
          if best_emb > best_base + 0.03 else
          "no real lift over dumb features — landing is mostly NOT content-predictable "
          "(it's context/timing, not the words)")


if __name__ == "__main__":
    main()
