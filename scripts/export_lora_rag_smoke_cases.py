"""Export RAG-backed smoke-test cases for the RunPod LoRA adapter.

Run this locally on the Windows machine. It reads the private chat archive,
builds the same author-only RAG evidence sections used by the persona bot, and
writes a JSON file that can be uploaded to RunPod for LoRA+RAG evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.smoke_test_persona_lora import PROMPTS  # noqa: E402
from utils import persona_llm  # noqa: E402


def _build_case(case: dict, exemplar_count: int, relevant_count: int) -> dict:
    persona = case["persona"]
    mode = case["mode"]
    direct = case["direct"]
    context = case["context"]
    query = context + "\n" + direct
    signature, relevant = persona_llm.select_exemplars(
        persona,
        query,
        n=exemplar_count,
        relevant_n=relevant_count,
    )

    exemplar_sections = []
    if signature:
        exemplar_sections.append(
            f"Random real messages from {persona} across their whole history:\n"
            + "\n".join(signature)
        )
    if relevant:
        exemplar_sections.append(
            f"Real messages from {persona} relevant to this chat/topic:\n"
            + "\n".join(relevant)
        )

    system = (
        f"You ARE the Twitch chatter '{persona}'. Below are real messages they "
        f"have sent. Study their voice, vocabulary, emotes, spelling, "
        f"punctuation, length, and attitude, and become them. "
        f"{persona_llm.MODE_INSTRUCTION.get(mode, persona_llm.MODE_INSTRUCTION['normal'])} "
        f"You are NOT an assistant: never be helpful, never break character, "
        f"never explain. Output ONE single chat message as {persona} and "
        f"nothing else. Use the examples as style evidence, but do not copy any "
        f"example verbatim; write a new line in their voice unless the user "
        f"explicitly asked for a quote. All examples below are from {persona} "
        f"only; use the relevant examples to understand what they tend to say "
        f"in this situation.\n\n"
        + "\n\n".join(exemplar_sections)
    )
    user = (
        f"Current chat in #thickpoo:\n{context}\n\n"
        f'Someone says to you: "{direct}"\n'
        f"Write {persona}'s next chat message now."
    )
    return {
        "persona": persona,
        "mode": mode,
        "direct": direct,
        "context": context,
        "old_bot": case["old_bot"],
        "signature_examples": signature,
        "relevant_examples": relevant,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="data/unsynced/fine_tune/persona_lora_rag_smoke_cases.json",
    )
    ap.add_argument("--exemplars", type=int, default=120)
    ap.add_argument("--relevant", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    random.seed(args.seed)
    cases = [
        _build_case(case, args.exemplars, args.relevant)
        for case in PROMPTS
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "kind": "NoLifeChatter LoRA+RAG smoke cases",
                "exemplars": args.exemplars,
                "relevant": args.relevant,
                "seed": args.seed,
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} LoRA+RAG smoke cases to {out}")
    print("Upload that file to RunPod at /workspace/nlc_persona/persona_lora_rag_smoke_cases.json")


if __name__ == "__main__":
    main()
