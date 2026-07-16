"""Per-utterance embedding index for roster authors (Bucket C #17b/#18 base).

Person-level mean vectors answer "who is this person overall"; per-unit
vectors unlock the two things means can't do:
- burst traits: someone who is extremely doomer in 10% of messages and
  neutral otherwise averages to mild — per-message projections catch the tail
- semantic persona retrieval: find the author's messages closest in MEANING
  to the live conversation, instead of FTS keyword overlap

Storage: data/unsynced/msg_index/<author>.npz with float16 vectors + the
texts (aligned). ~3MB per 1000 units per author; loaded lazily.

    python scripts/build_message_index.py [--per-author 3000]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils import atomic_file, chat_archive, message_quality, persona_classifier  # noqa: E402
from scripts.build_persona_embeddings import embed_batch  # noqa: E402

OUT_DIR = os.path.join("data", "unsynced", "msg_index")


def main():
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-author", type=int, default=3000)
    ap.add_argument("--unit", choices=("utterance", "message"), default="utterance",
                    help="semantic unit to embed; utterance merges same-author bursts")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing per-author index files")
    args = ap.parse_args()
    if not config.LLM_EMBED_MODEL:
        print("Set [llm] embed_model in config.toml first.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    model = persona_classifier.load()
    roster = sorted({
        chat_archive.normalize_author(a)
        for a in (model.get("profiles") or {}).keys()
        if not chat_archive._is_noise_author(a)
    })
    if args.force:
        keep = {f"{a}.npz" for a in roster}
        stale = [
            name for name in os.listdir(OUT_DIR)
            if name.endswith(".npz") and name not in keep
        ]
        for name in stale:
            os.remove(os.path.join(OUT_DIR, name))
        if stale:
            print(f"pruned {len(stale)} stale message-index files", flush=True)
    for i, a in enumerate(roster, 1):
        out = os.path.join(OUT_DIR, f"{a}.npz")
        if os.path.exists(out) and not args.force:
            print(f"  ({i}/{len(roster)}) {a}: exists, skipped", flush=True)
            continue
        if os.path.exists(out):
            print(f"  ({i}/{len(roster)}) {a}: rebuilding", flush=True)
        source = (
            chat_archive.utterances_for(a)
            if args.unit == "utterance"
            else chat_archive.messages_for(a)
        )
        rows, selection = message_quality.select_semantic_units(
            source,
            cap=args.per_author,
            label=f"{a}:{args.unit}:semantic-index",
            min_words=4,
            max_words=70,
        )
        if len(rows) < 30:
            print(f"  ({i}/{len(roster)}) {a}: too few {args.unit}s, skipped", flush=True)
            continue
        vecs = []
        for j in range(0, len(rows), 64):
            vecs.extend(embed_batch([row["clean"] for row in rows[j:j + 64]]))
        V = np.asarray(vecs, dtype="float32")
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        with atomic_file.open_atomic(out, "wb") as handle:
            np.savez_compressed(
                handle,
                vectors=V.astype("float16"),
                texts=np.array([row["text"] for row in rows], dtype=object),
                kinds=np.array([row["kind"] for row in rows], dtype=object),
                unit=np.array(args.unit, dtype=object),
                utterance_version=np.array(
                    chat_archive.UTTERANCE_VERSION if args.unit == "utterance" else 0
                ),
                model=np.array(config.LLM_EMBED_MODEL, dtype=object),
                alias_signature=np.array(chat_archive.alias_signature(), dtype=object),
                selection_version=np.array(message_quality.SELECTION_VERSION),
                source_count=np.array(selection["source_count"]),
                usable_unique=np.array(selection["usable_unique"]),
                built_at=np.array(time.strftime("%Y-%m-%d %H:%M:%S"), dtype=object),
            )
        print(
            f"  ({i}/{len(roster)}) {a}: {len(rows)} {args.unit}s indexed "
            f"({selection['coverage']} coverage + {selection['high_signal']} high-signal; "
            f"{selection['usable_unique']} usable unique)",
            flush=True,
        )
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
