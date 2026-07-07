"""Usage-context emote semantics: what an emote MEANS, learned from our logs.

Emote names lie (deliberately irrelevant names), images aren't always
fetchable (dead emotes in old logs), and fake personal emotes exist. But
meaning-from-usage covers all of it: an emote's vector = mean embedding of
the messages it appears in, with the emote token itself removed. DansGame's
contexts are disgust, so its vector points at disgust — the stance-OPERATOR
meaning the name embedding can never carry.

Per emote: sample up to --contexts messages containing it (FTS), strip the
emote, require >=3 remaining words, embed, mean-pool. Emotes with too few
usable contexts are skipped (callers fall back to name embeddings).

    python scripts/build_emote_semantics.py [--top 2000] [--contexts 160]
    python scripts/build_emote_semantics.py --emotes BatChest,KEKW --contexts 200 --refresh
"""

import argparse
import hashlib
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from utils import chat_archive, persona_classifier as pc  # noqa: E402
from scripts.build_persona_embeddings import embed_batch  # noqa: E402

OUT = os.path.join("data", "unsynced", "emote_semantics.pkl")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=2000)
    ap.add_argument("--contexts", type=int, default=160)
    ap.add_argument("--emotes", default="",
                    help="comma-separated emotes to build instead of the frequency top-N")
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild existing vectors; with --emotes only those keys are replaced")
    args = ap.parse_args()

    # Source: the ground-truth registry UNION the most frequent emote-shaped
    # tokens in the archive. Sourcing from per-person DISTINCTIVE profiles
    # (the old way) filtered out the universal meaning-bearers (DansGame,
    # Sadge, KEKW) precisely because everyone uses them.
    import json
    from collections import Counter
    reg = {}
    reg_path = os.path.join("data", "unsynced", "emote_registry.json")
    if os.path.exists(reg_path):
        reg = json.load(open(reg_path, encoding="utf-8"))
    conn = chat_archive.connect()
    if args.emotes.strip():
        emotes = [e.strip() for e in args.emotes.split(",") if e.strip()]
    else:
        freq = Counter()
        for (m,) in conn.execute("SELECT content FROM messages ORDER BY RANDOM() LIMIT 400000"):
            for tok in (m or "").split():
                t = tok.strip(".,!?;:")
                if t in reg or pc._is_emote_token(t):
                    freq[t] += 1
        emotes = [e for e, n in freq.most_common(args.top) if n >= 5]
    print(f"building context vectors for {len(emotes)} emotes "
          f"(registry {len(reg)} + archive-frequent)...")

    conn = chat_archive.connect()
    out = {}
    done = 0
    if os.path.exists(OUT):
        with open(OUT, "rb") as fh:
            out = pickle.load(fh)
    if args.refresh:
        if args.emotes.strip():
            lows = {e.lower() for e in emotes}
            for key in list(out):
                if key.lower() in lows:
                    out.pop(key, None)
        else:
            out = {}
    for i, emote in enumerate(emotes, 1):
        if emote in out:
            continue
        limit = max(args.contexts * 8, 400)
        salt = int(hashlib.sha1(emote.lower().encode("utf-8")).hexdigest()[:8], 16)
        try:
            rows = conn.execute(
                "SELECT m.content FROM messages_fts f JOIN messages m ON m.id = f.rowid "
                "WHERE f.messages_fts MATCH ? "
                "ORDER BY ((m.id * ?) % 2147483647) LIMIT ?",
                (f'"{emote}"', (salt % 1000003) * 2 + 1, limit)).fetchall()
        except Exception:
            rows = conn.execute(
                "SELECT content FROM messages WHERE content LIKE ? LIMIT ?",
                (f"%{emote}%", limit)).fetchall()
        ctxs = []
        for (m,) in rows:
            stripped = " ".join(t for t in m.split() if t.strip(".,!?") != emote)
            if len(stripped.split()) >= 3:
                ctxs.append(stripped[:300])
            if len(ctxs) >= args.contexts:
                break
        if len(ctxs) < 8:
            continue
        embs = embed_batch(ctxs)
        v = np.asarray(embs, dtype="float32").mean(axis=0)
        out[emote] = {"vector": (v / (np.linalg.norm(v) + 1e-9)).astype("float16"),
                      "n": len(ctxs)}
        done += 1
        if done % 100 == 0:
            with open(OUT, "wb") as fh:
                pickle.dump(out, fh)
            print(f"  ({i}/{len(emotes)}) {len(out)} vectors saved...", flush=True)
    with open(OUT, "wb") as fh:
        pickle.dump(out, fh)
    print(f"done: {len(out)} emote context-vectors -> {OUT}")

    # sanity: do operator emotes point where they should?
    probes = {"disgust ew gross nasty": None, "sad crying unhappy": None,
              "happy nice wholesome": None}
    P = {k: np.asarray(v, dtype="float32") for k, v in
         zip(probes, embed_batch(list(probes)))}
    for k in P:
        P[k] /= np.linalg.norm(P[k])
    for emote in ["DansGame", "Sadge", "FeelsBadMan", "ApuDoomer", "FeelsOkayMan"]:
        d = out.get(emote)
        if not d:
            print(f"  {emote}: (no vector)")
            continue
        v = np.asarray(d["vector"], dtype="float32")
        best = max(P, key=lambda k: float(v @ P[k]))
        print(f"  {emote}: closest probe = '{best}' "
              + " ".join(f"{k.split()[0]}:{float(v @ P[k]):+.2f}" for k in P))


if __name__ == "__main__":
    main()
