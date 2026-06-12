"""Generate oracle review queues into the ai-prompt-engineering dropoff.

Two queues, schema per HANDOFF_REVIEW_TOOL_V2_FEEDBACK.md (items with
answer:null; the review tool fills them; results expected back under
dropoff/<queue>/results/):

1. **irony**: uncertainty-sampled real messages + context. Items are chosen
   where the two automatic features (surface-sarcasm z, proposition-harm z)
   CONFLICT or sit near the boundary — exactly the messages the future
   classifier can't call on its own. Labels: sincere / playful-ironic /
   masking-ironic / cant-tell.
2. **renames**: account pairs that look like the same person before/after a
   name change — activity windows nearly disjoint with a small handoff gap,
   high compression similarity (NCD). Confirmed pairs become
   [archive.user_aliases] entries.

No GPU host needed for any of this: the irony head trains on CPU from these
labels; embeddings come from local LM Studio.

    python scripts/build_oracle_queues.py [--irony-n 60] [--renames-n 15]
"""

import argparse
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import config  # noqa: E402
from utils import chat_archive, persona_classifier as pc  # noqa: E402
from utils.persona_traits import _axis_vectors, _embed  # noqa: E402

DROPOFF = os.path.join("..", "ai-prompt-engineering", "dropoff")


def _write_queue(name, items):
    d = os.path.join(DROPOFF, name)
    os.makedirs(os.path.join(d, "results"), exist_ok=True)
    path = os.path.join(d, "queue.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  {len(items)} items -> {path}")


def harm_axis():
    pos = ['children deserve to suffer', 'i would enjoy hurting someone vulnerable',
           'causing pain to innocent people is a good thing',
           'i hope something terrible happens to them',
           'making others miserable brings me joy']
    neg = ['children deserve to be safe and happy', 'i would enjoy helping someone vulnerable',
           'protecting innocent people is a good thing',
           'i hope something wonderful happens to them',
           'making others happy brings me joy']
    e = _embed(pos + neg)
    P = np.asarray(e[:5], dtype="float32").mean(axis=0)
    N = np.asarray(e[5:], dtype="float32").mean(axis=0)
    v = P - N
    return v / (np.linalg.norm(v) + 1e-9)


def build_irony_queue(n):
    conn = chat_archive.connect()
    rows = conn.execute(
        "SELECT id, channel, author, content FROM messages "
        "WHERE LENGTH(content) > 25 AND sent_at >= '2025-01-01' "
        "ORDER BY RANDOM() LIMIT 400").fetchall()
    rows = [r for r in rows if pc._usable(r[3])][:350]
    iron = np.asarray(_axis_vectors()["ironic"])
    harm = harm_axis()
    texts = [pc.strip_emote_tokens(r[3]) for r in rows]
    embs = []
    for i in range(0, len(texts), 64):
        embs.extend(_embed(texts[i:i + 64]))
    E = np.asarray(embs, dtype="float32")
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    zi = E @ iron
    zh = E @ harm
    zi = (zi - zi.mean()) / zi.std()
    zh = (zh - zh.mean()) / zh.std()
    # uncertainty: near the surface boundary, or features conflict
    uncertainty = -np.abs(zi) + np.clip(zh, 0, None) * 0.8
    order = uncertainty.argsort()[::-1][:n]
    items = []
    for rank, idx in enumerate(order):
        mid, channel, author, content = rows[idx]
        window = chat_archive.context_window(mid, channel, before=4, after=2)
        ctx = "\n".join(f"{a}: {c[:120]}" for _i, a, c in window)
        items.append({
            "id": f"nlc-irony-{rank:04d}",
            "source": "NoLifeChatter",
            "kind": "single-classification",
            "question": "Is the MARKED message sincere or ironic (and which kind)?",
            "subject": {"author": author, "message": content, "context": ctx},
            "evidence": {"surface_irony_z": round(float(zi[idx]), 2),
                         "harm_proposition_z": round(float(zh[idx]), 2)},
            "options": ["sincere", "playful-ironic", "masking-ironic", "cant-tell"],
            "allow_other": True,
            "answer": None, "answer_note": None, "answered_at": None,
        })
    _write_queue("nolifechatter_irony_v1", items)


def build_rename_queue(n):
    conn = chat_archive.connect()
    aliased = set(chat_archive.USER_ALIASES) | set(chat_archive.USER_ALIASES.values())
    stats = [(a, mn, mx, c) for a, mn, mx, c in conn.execute(
        "SELECT author, MIN(sent_at), MAX(sent_at), COUNT(*) FROM messages "
        "GROUP BY author HAVING c >= 300"
        .replace("c >=", "COUNT(*) >="))
        if a not in aliased and "bot" not in a]
    import datetime
    def _d(s):
        return datetime.datetime.strptime(s[:10], "%Y-%m-%d")
    cands = []
    for a, amn, amx, ac in stats:
        for b, bmn, bmx, bc in stats:
            if a >= b:
                continue
            # a "rename": one account dies, the other is born within 90 days
            for old, old_end, new, new_start in ((a, amx, b, bmn), (b, bmx, a, amn)):
                gap = (_d(new_start) - _d(old_end)).days
                if -14 <= gap <= 90:
                    cands.append((old, new, gap))
    # score the few survivors with compression similarity
    blobs = {}
    def blob(author):
        if author not in blobs:
            msgs = chat_archive.messages_for(author)
            blobs[author] = ("\n".join(msgs)[:50000]).encode("utf-8", "ignore")
        return blobs[author]
    scored = []
    for old, new, gap in cands:
        ba, bb = blob(old), blob(new)
        ca, cb = len(zlib.compress(ba, 6)), len(zlib.compress(bb, 6))
        cab = len(zlib.compress(ba + bb, 6))
        ncd = (cab - min(ca, cb)) / max(ca, cb)
        scored.append((ncd, old, new, gap))
    scored.sort()
    items = []
    for rank, (ncd, old, new, gap) in enumerate(scored[:n]):
        items.append({
            "id": f"nlc-rename-{rank:04d}",
            "source": "NoLifeChatter",
            "kind": "pair-classification",
            "question": "Did this account get RENAMED into the other (same person, old name -> new name)?",
            "subject": {"old_account": old, "new_account": new,
                        "days_between_last_and_first_message": gap},
            "evidence": {"compression_ncd": round(ncd, 3),
                         "note": "lower NCD = more similar writing"},
            "options": ["same person (rename)", "different people", "i don't know"],
            "allow_other": True,
            "answer": None, "answer_note": None, "answered_at": None,
        })
    _write_queue("nolifechatter_renames_v1", items)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--irony-n", type=int, default=60)
    ap.add_argument("--renames-n", type=int, default=15)
    args = ap.parse_args()
    print("building irony queue...")
    build_irony_queue(args.irony_n)
    print("building rename-candidates queue...")
    build_rename_queue(args.renames_n)


if __name__ == "__main__":
    main()
