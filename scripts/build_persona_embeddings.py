"""Build per-chatter semantic vectors from a local embedding model.

The token-statistics layer (~markers/~like) can only see which exact words
someone overuses; this is the semantic layer that sees what they talk ABOUT.
For every voice-profile roster author: sample their usable messages, embed
each via LM Studio's /v1/embeddings (config [llm] embed_model — runs fully
local), mean-pool into one L2-normalized person vector, and store the lot in
a gitignored pickle. Downstream: semantic ~like / ~vibes, clustering for the
personality-map idea, trait axes (docs/CHAT_PERSONALITY_RESEARCH.md).

    python scripts/build_persona_embeddings.py [--per-author 200] [--report]
"""

import argparse
import json
import os
import pickle
import random
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils import chat_archive, message_quality, persona_classifier  # noqa: E402

OUT = os.path.join("data", "unsynced", "persona_embeddings.pkl")


def embed_batch(texts):
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    body = json.dumps({"model": config.LLM_EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(
        base + "/v1/embeddings", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return [d["embedding"] for d in data["data"]]


def person_vector(author, per_author, rng):
    """Mean-pooled, L2-normalized embedding of a sample of their messages.

    Emote tokens and URLs are stripped BEFORE embedding and a message must
    still carry >=4 real words — the semantic layer should embed meaning;
    emote usage is its own profile channel. (One-emote lines taught the old
    vectors nothing and diluted everyone toward the same point.)"""
    import numpy as np
    msgs = []
    for m in chat_archive.messages_for(author):
        cleaned = message_quality.semantic_text(m, min_words=4, max_words=60)
        if cleaned:
            msgs.append(cleaned)
    if len(msgs) < 30:
        return None, 0
    rng.shuffle(msgs)
    msgs = msgs[:per_author]
    vecs = []
    for i in range(0, len(msgs), 64):
        vecs.extend(embed_batch(msgs[i:i + 64]))
    v = np.asarray(vecs, dtype="float32").mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9), len(msgs)


def main():
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-author", type=int, default=1000)
    ap.add_argument("--report", action="store_true",
                    help="print nearest neighbors per author when done")
    args = ap.parse_args()
    if not config.LLM_EMBED_MODEL:
        print("Set [llm] embed_model in config.toml first.")
        return

    model = persona_classifier.load()
    roster = sorted((model.get("profiles") or {}).keys())
    print(f"embedding {len(roster)} chatters, ~{args.per_author} msgs each...")
    rng = random.Random(7)
    vectors = {}
    for i, a in enumerate(roster, 1):
        v, n = person_vector(a, args.per_author, rng)
        if v is None:
            print(f"  ({i}/{len(roster)}) {a}: too few usable messages, skipped")
            continue
        vectors[a] = v
        print(f"  ({i}/{len(roster)}) {a}: {n} msgs embedded", flush=True)

    with open(OUT, "wb") as fh:
        pickle.dump({"model": config.LLM_EMBED_MODEL,
                     "per_author": args.per_author,
                     "vectors": vectors}, fh)
    print(f"\n{len(vectors)} person vectors -> {OUT}")

    if args.report and vectors:
        names = list(vectors)
        M = np.vstack([vectors[a] for a in names])
        sims = M @ M.T
        print("\nnearest semantic neighbors:")
        for i, a in enumerate(names):
            order = sims[i].argsort()[::-1]
            nn = [(names[j], sims[i][j]) for j in order if j != i][:3]
            print(f"  {a:26} " + " | ".join(f"{b} {s:.2f}" for b, s in nn))


if __name__ == "__main__":
    main()
