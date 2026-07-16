"""Build per-chatter semantic vectors from a local embedding model.

The token-statistics layer (~markers/~like) can only see which exact words
someone overuses; this is the semantic layer that sees what they talk ABOUT.
For every voice-profile roster author: sample their usable utterances, embed
each via LM Studio's /v1/embeddings (config [llm] embed_model — runs fully
local), mean-pool into one L2-normalized person vector, and store the lot in
a gitignored pickle. Downstream: semantic ~like / ~vibes, clustering for the
personality-map idea, trait axes (docs/CHAT_PERSONALITY_RESEARCH.md).

    python scripts/build_persona_embeddings.py [--per-author 3000] [--report]
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils import atomic_file, chat_archive, message_quality, persona_classifier  # noqa: E402

OUT = os.path.join("data", "unsynced", "persona_embeddings.pkl")
INDEX_DIR = os.path.join("data", "unsynced", "msg_index")


def embed_batch(texts):
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    body = json.dumps({"model": config.LLM_EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(
        base + "/v1/embeddings", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return [d["embedding"] for d in data["data"]]


def _semantic_units(author: str, unit: str) -> list[str]:
    source = (
        chat_archive.utterances_for(author)
        if unit == "utterance"
        else chat_archive.messages_for(author)
    )
    out = []
    for text in source:
        cleaned = message_quality.semantic_text(text, min_words=4, max_words=70)
        if cleaned:
            out.append(cleaned)
    return out


def person_vector(author, per_author, rng, unit="utterance"):
    """Mean-pooled, L2-normalized embedding of a sample of their messages.

    Emote tokens and URLs are stripped BEFORE embedding and a message must
    still carry >=4 real words — the semantic layer should embed meaning;
    emote usage is its own profile channel. (One-emote lines taught the old
    vectors nothing and diluted everyone toward the same point.)"""
    import numpy as np
    msgs = _semantic_units(author, unit)
    if len(msgs) < 30:
        return None, 0
    rng.shuffle(msgs)
    msgs = msgs[:per_author]
    vecs = []
    for i in range(0, len(msgs), 64):
        vecs.extend(embed_batch(msgs[i:i + 64]))
    v = np.asarray(vecs, dtype="float32").mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9), len(msgs)


def person_vector_from_index(author, per_author, unit="utterance"):
    """Mean the unbiased coverage lane from the shared semantic index."""
    import numpy as np

    path = os.path.join(INDEX_DIR, f"{chat_archive.normalize_author(author)}.npz")
    if not os.path.exists(path):
        return None, 0
    with np.load(path, allow_pickle=True) as data:
        found_unit = str(data["unit"].item()) if "unit" in data.files else "message"
        found_model = str(data["model"].item()) if "model" in data.files else ""
        found_aliases = (
            str(data["alias_signature"].item())
            if "alias_signature" in data.files else ""
        )
        found_utterance_version = (
            int(data["utterance_version"].item())
            if "utterance_version" in data.files else 0
        )
        if found_unit != unit:
            raise ValueError(f"{author}: index unit={found_unit}, expected {unit}")
        if found_model and found_model != config.LLM_EMBED_MODEL:
            raise ValueError(
                f"{author}: index model={found_model}, expected {config.LLM_EMBED_MODEL}"
            )
        if found_aliases != chat_archive.alias_signature():
            raise ValueError(f"{author}: index identity metadata is stale or missing")
        if (
            unit == "utterance"
            and found_utterance_version != chat_archive.UTTERANCE_VERSION
        ):
            raise ValueError(
                f"{author}: index utterance version={found_utterance_version}, "
                f"expected {chat_archive.UTTERANCE_VERSION}"
            )
        vectors = data["vectors"].astype("float32")
        if "kinds" in data.files:
            kinds = data["kinds"].astype(str)
            coverage = vectors[kinds == "coverage"]
            if len(coverage):
                vectors = coverage
    if per_author > 0:
        vectors = vectors[:per_author]
    if len(vectors) < 30:
        return None, len(vectors)
    v = vectors.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9), len(vectors)


def main():
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-author", type=int, default=3000)
    ap.add_argument("--unit", choices=("utterance", "message"), default="utterance",
                    help="semantic unit to embed; utterance merges same-author bursts")
    ap.add_argument("--report", action="store_true",
                    help="print nearest neighbors per author when done")
    ap.add_argument(
        "--from-message-index",
        action="store_true",
        help="reuse the shared semantic index instead of embedding a duplicate sample",
    )
    args = ap.parse_args()
    if not config.LLM_EMBED_MODEL:
        print("Set [llm] embed_model in config.toml first.")
        return

    model = persona_classifier.load()
    roster = sorted({
        chat_archive.normalize_author(a)
        for a in (model.get("profiles") or {}).keys()
        if not chat_archive._is_noise_author(a)
    })
    action = "pooling indexed" if args.from_message_index else "embedding"
    print(f"{action} {len(roster)} chatters, ~{args.per_author} {args.unit}s each...")
    rng = random.Random(7)
    vectors = {}
    for i, a in enumerate(roster, 1):
        if args.from_message_index:
            v, n = person_vector_from_index(a, args.per_author, unit=args.unit)
        else:
            v, n = person_vector(a, args.per_author, rng, unit=args.unit)
        if v is None:
            print(f"  ({i}/{len(roster)}) {a}: too few usable {args.unit}s, skipped")
            continue
        vectors[a] = v
        print(f"  ({i}/{len(roster)}) {a}: {n} {args.unit}s embedded", flush=True)

    with atomic_file.open_atomic(OUT, "wb") as fh:
        pickle.dump({"model": config.LLM_EMBED_MODEL,
                     "per_author": args.per_author,
                     "unit": args.unit,
                     "utterance_version": (
                         chat_archive.UTTERANCE_VERSION
                         if args.unit == "utterance" else 0
                     ),
                     "source": "message_index" if args.from_message_index else "direct",
                     "alias_signature": chat_archive.alias_signature(),
                     "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
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
