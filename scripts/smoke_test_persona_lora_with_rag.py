"""Smoke-test a trained LoRA adapter with exported RAG evidence.

Run this on RunPod. The cases JSON is exported locally by
scripts/export_lora_rag_smoke_cases.py so the GPU machine does not need the
private SQLite archive.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path


def _match_key(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[`'\".,!?;:()\[\]{}<>]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _copy_flag(text: str, examples: list[str]) -> str:
    key = _match_key(text)
    if not key:
        return ""
    for example in examples:
        if key == _match_key(example):
            return " [EXACT PROMPT COPY]"
    for example in examples:
        ex_key = _match_key(example)
        if not ex_key:
            continue
        if difflib.SequenceMatcher(None, key, ex_key).ratio() >= 0.94:
            return " [NEAR PROMPT COPY]"
    return ""


def _first_line(text: str, persona: str) -> str:
    text = (text or "").strip()
    text = re.sub(rf"^{re.escape(persona)}\s*[:>-]\s*", "", text, flags=re.IGNORECASE)
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text.splitlines()[0].strip() if text else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="/workspace/nlc_persona/persona_lora")
    ap.add_argument("--cases", default="/workspace/nlc_persona/persona_lora_rag_smoke_cases.json")
    ap.add_argument("--out", default="/workspace/nlc_persona/persona_lora_rag_smoke_test.txt")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--hyper-temperature", type=float, default=1.05)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--samples-per-prompt", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    adapter = Path(args.adapter)
    cases_path = Path(args.cases)
    if not adapter.exists():
        raise SystemExit(f"Adapter path not found: {adapter}")
    if not cases_path.exists():
        raise SystemExit(
            f"Cases JSON not found: {cases_path}\n"
            "Export it locally with 10-export-lora-rag-smoke-cases.bat, then "
            "upload it to /workspace/nlc_persona/ on RunPod."
        )

    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not cases:
        raise SystemExit(f"No cases found in {cases_path}")

    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch

    torch.manual_seed(args.seed)
    print(f"Loading LoRA adapter: {adapter}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    results = []
    for idx, case in enumerate(cases, 1):
        persona = case["persona"]
        mode = case["mode"]
        examples = case.get("signature_examples", []) + case.get("relevant_examples", [])
        prompt = tokenizer.apply_chat_template(
            case["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        lines = []
        print(f"{idx}/{len(cases)} {persona} [{mode}]: {case['direct']}", flush=True)
        for sample_i in range(max(1, args.samples_per_prompt)):
            torch.manual_seed(args.seed + (idx * 101) + sample_i)
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            temperature = args.hyper_temperature if mode == "hyper" else args.temperature
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True,
            )
            line = _first_line(generated, persona)
            flag = _copy_flag(line, examples)
            lines.append((line, flag))
            print(f"  {sample_i + 1}. {line}{flag}", flush=True)
        results.append((case, lines))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("NoLifeChatter LoRA + RAG smoke test\n")
        f.write(f"adapter: {adapter}\n")
        f.write(f"cases: {cases_path}\n\n")
        for idx, (case, lines) in enumerate(results, 1):
            f.write("=" * 72 + "\n")
            f.write(f"{idx}. {case['persona']} ({case['mode']})\n")
            f.write(f"direct: {case['direct']}\n")
            f.write(f"old bot output: {case['old_bot']}\n")
            f.write(
                "examples: "
                f"{len(case.get('signature_examples', []))} random + "
                f"{len(case.get('relevant_examples', []))} relevant\n\n"
            )
            f.write("context:\n")
            f.write(case["context"] + "\n\n")
            f.write("top relevant examples:\n")
            for example in case.get("relevant_examples", [])[:8]:
                f.write(f"- {example}\n")
            if not case.get("relevant_examples"):
                f.write("- (none)\n")
            f.write("\nLoRA + RAG outputs:\n")
            for sample_i, (line, flag) in enumerate(lines, 1):
                f.write(f"{sample_i}. {line}{flag}\n")
            f.write("\n")
    print(f"\nSaved LoRA+RAG smoke-test outputs to {out}", flush=True)


if __name__ == "__main__":
    main()
