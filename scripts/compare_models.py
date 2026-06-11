"""A/B two LM Studio models on the persona task (e.g. plain Llama vs the LoRA).

For each persona, generate N lines from EACH model (same RAG prompt) and score
every line with the authorship classifier: how often does the line read as the
intended author? Prints the sample lines side by side plus a per-model
"sounds-like-them" rate, so you can both read the outputs and see a number.

Both models must be loaded in LM Studio at once (it serves several; the request
routes by model id). List ids:  curl -s http://127.0.0.1:1234/v1/models

    python scripts/compare_models.py --models meta-llama-3.1-8b-instruct,persona-lora-v2/persona_merged_q4_k_m
                                     [--authors a,b,c] [--per-author 6] [--mode normal] [--channel ch]
"""

import argparse
import asyncio
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from services import llm  # noqa: E402
from utils import chat_archive, persona_classifier, persona_llm  # noqa: E402


def list_models():
    try:
        base = config.LLM_ENDPOINT.split("/v1/")[0]
        d = json.load(urllib.request.urlopen(base + "/v1/models", timeout=5))
        return [m["id"] for m in d.get("data", [])]
    except Exception:
        return []


async def run_model(model_id, authors, channel, per_author, mode):
    config.LLM_MODEL = model_id  # llm.chat reads this at call time -> routes here
    out = {}
    for a in authors:
        hits, probs, samples = 0, [], []
        for _ in range(per_author):
            line = await persona_llm.generate(a, channel, mode=mode, invoked_by="compare")
            if not line:
                continue
            ranked = persona_classifier.classify(line, top_k=60)
            if not ranked:
                continue
            top = ranked[0][0]
            tp = next((p for au, p in ranked if au == a), 0.0)
            probs.append(tp)
            if top == a:
                hits += 1
            samples.append(f"[{top} {ranked[0][1]:.0%}] {line}")
        n = len(probs)
        out[a] = {
            "rate": hits / n if n else 0.0,
            "avg_prob": sum(probs) / n if n else 0.0,
            "n": n,
            "samples": samples,
        }
        print(f"  ...{model_id} | {a}: {hits}/{n} read as them", flush=True)
    return out


async def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="", help="two comma-separated LM Studio model ids")
    ap.add_argument("--authors", default="")
    ap.add_argument("--per-author", type=int, default=6)
    ap.add_argument("--mode", default="normal")
    ap.add_argument("--channel", default="")
    args = ap.parse_args()

    available = list_models()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) != 2:
        print("Pass exactly two with --models. Currently loaded in LM Studio:")
        for m in available:
            print("   ", m)
        return
    for m in models:
        if available and m not in available:
            print(f"WARNING: '{m}' is not in the loaded model list — load it in LM Studio.")

    cmodel = persona_classifier.load()
    authors = [a.strip() for a in args.authors.split(",") if a.strip()] or cmodel["authors"][:8]
    authors = [a for a in authors if a in cmodel["authors"]]
    channel = args.channel or (chat_archive.connect().execute(
        "SELECT channel FROM messages GROUP BY channel ORDER BY COUNT(*) DESC LIMIT 1").fetchone() or [""])[0]

    print(f"Comparing on #{channel}, {args.per_author} lines/author, mode {args.mode}\n")
    a_res = await run_model(models[0], authors, channel, args.per_author, args.mode)
    b_res = await run_model(models[1], authors, channel, args.per_author, args.mode)

    print("\n================ RESULTS ================")
    for a in authors:
        print(f"\n### {a}")
        for label, res in ((models[0], a_res[a]), (models[1], b_res[a])):
            print(f"  {label}  — reads-as-them {res['rate']:.0%} (avg {res['avg_prob']:.0%}, n={res['n']})")
            for s in res["samples"][:3]:
                print(f"      {s}")
    print("\n================ OVERALL ================")
    for label, res in ((models[0], a_res), (models[1], b_res)):
        rates = [res[a]["rate"] for a in authors if res[a]["n"]]
        print(f"  {label}: reads-as-them {sum(rates)/len(rates):.0%}" if rates else f"  {label}: no data")
    out = os.path.join("data", "unsynced", "model_compare.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"channel": channel, "models": models,
                   "a": a_res, "b": b_res}, fh, ensure_ascii=False, indent=2)
    print(f"\nfull side-by-side -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
