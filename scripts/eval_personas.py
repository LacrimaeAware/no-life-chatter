"""Objective persona eval: does a generated persona line read as the target?

For each author, generate N lines with the persona engine (the live RAG path
against LM Studio), run each through the authorship classifier, and measure how
often the line is attributed to the intended author (top-1) and the average
probability mass the classifier puts on them. Compares against the classifier's
attribution rate on that author's REAL held-out lines (its ceiling), so you can
read generated-vs-real on the same scale.

No fine-tuned weights required — this scores whatever `persona_llm.generate`
currently uses (RAG today). Point it at a LoRA-backed endpoint later to compare.

    python scripts/eval_personas.py [--authors a,b,c] [--per-author 6]
                                    [--channel <ch>] [--mode normal|hyper]
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import llm  # noqa: E402
from utils import chat_archive, persona_classifier, persona_llm  # noqa: E402


async def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--authors", default="", help="default = the classifier's authors")
    ap.add_argument("--per-author", type=int, default=6)
    ap.add_argument("--channel", default="")
    ap.add_argument("--mode", default="normal")
    args = ap.parse_args()

    if not await llm.available():
        print("LM Studio endpoint not reachable — start the local server first.")
        return

    model = persona_classifier.load()  # raises if untrained
    authors = [a.strip() for a in args.authors.split(",") if a.strip()] or model["authors"]
    channel = args.channel or (chat_archive.connect().execute(
        "SELECT channel FROM messages GROUP BY channel ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone() or [""])[0]

    overall_gen, overall_real = [], []
    rows = []
    for author in authors:
        if author not in model["authors"]:
            continue
        # generated lines through the live persona engine
        gen_hits, gen_probs, confusions, samples = 0, [], {}, []
        for _ in range(args.per_author):
            line = await persona_llm.generate(author, channel, mode=args.mode,
                                               invoked_by="eval")
            if not line:
                continue
            ranked = persona_classifier.classify(line, top_k=2)
            if not ranked:
                continue
            top, p = ranked[0]
            target_p = next((pr for a, pr in ranked if a == author), 0.0)
            gen_probs.append(target_p)
            if top == author:
                gen_hits += 1
            else:
                confusions[top] = confusions.get(top, 0) + 1
            if len(samples) < 2:
                samples.append(f"[{top} {p:.0%}] {line}")
        n_gen = len(gen_probs)

        # classifier's ceiling on this author's REAL lines — a RANDOM sample
        # across all their messages (not the earliest, which is unrepresentative).
        import random
        real = [m for m in chat_archive.messages_for(author)
                if m and len(m.split()) >= 2]
        random.Random(1).shuffle(real)
        real = real[:50]
        real_hits = sum(1 for m in real
                        if (r := persona_classifier.classify(m, top_k=1)) and r[0][0] == author)
        real_rate = real_hits / len(real) if real else 0.0

        gen_rate = gen_hits / n_gen if n_gen else 0.0
        avg_p = sum(gen_probs) / n_gen if n_gen else 0.0
        top_conf = max(confusions.items(), key=lambda kv: kv[1])[0] if confusions else "-"
        rows.append({"author": author, "generated_as_target": round(gen_rate, 2),
                     "avg_target_prob": round(avg_p, 2), "real_lines_rate": round(real_rate, 2),
                     "mistaken_for": top_conf, "n": n_gen, "samples": samples})
        if n_gen:
            overall_gen.append(gen_rate)
            overall_real.append(real_rate)
        print(f"{author:26} gen→target {gen_rate:.0%}  (avg p {avg_p:.0%})  "
              f"| real-line ceiling {real_rate:.0%}  | mistaken for {top_conf}", flush=True)

    if overall_gen:
        print()
        print(f"OVERALL  generated-as-target {sum(overall_gen)/len(overall_gen):.0%}"
              f"   vs real-line ceiling {sum(overall_real)/len(overall_real):.0%}"
              f"   ({len(overall_gen)} authors, channel #{channel}, mode {args.mode})")
    out = os.path.join("data", "unsynced", "persona_eval.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"channel": channel, "mode": args.mode, "rows": rows}, fh,
                  ensure_ascii=False, indent=2)
    print(f"detail (with sample lines) -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
