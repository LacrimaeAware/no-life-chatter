"""Compare a RunPod LoRA smoke-test file with local LM Studio + RAG outputs.

This does not modify the live bot. It reuses the same public smoke-test cases,
loads the RunPod text output if provided, then asks the current local LM Studio
model using the project's persona RAG prompt shape.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import llm  # noqa: E402
from utils import chat_archive, persona_llm  # noqa: E402
from scripts.smoke_test_persona_lora import PROMPTS  # noqa: E402


def _parse_lora_outputs(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = re.split(r"^={20,}\s*$", text, flags=re.MULTILINE)
    out = {}
    for section in sections:
        persona = re.search(r"^persona:\s*(.+)$", section, flags=re.MULTILINE)
        mode = re.search(r"^mode:\s*(.+)$", section, flags=re.MULTILINE)
        direct = re.search(r"^direct:\s*(.+)$", section, flags=re.MULTILINE)
        outputs = re.findall(r"^\d+\.\s*(.+)$", section, flags=re.MULTILINE)
        if persona and mode and direct:
            key = _case_key(persona.group(1), mode.group(1), direct.group(1))
            out[key] = outputs
    return out


def _case_key(persona: str, mode: str, direct: str) -> str:
    return f"{persona.lower()}|{mode.lower()}|{direct.strip().lower()}"


async def _local_rag(case: dict, sample_i: int, exemplar_count: int,
                     relevant_count: int) -> tuple[str, bool, int, int]:
    persona = case["persona"]
    context = case["context"]
    direct = case["direct"]
    mode = case["mode"]
    signature, relevant = persona_llm.select_exemplars(
        persona,
        context + "\n" + direct,
        n=exemplar_count,
        relevant_n=relevant_count,
    )
    sections = []
    if signature:
        sections.append(
            f"Random real messages from {persona} across their whole history:\n"
            + "\n".join(signature)
        )
    if relevant:
        sections.append(
            f"Real messages from {persona} relevant to this chat/topic:\n"
            + "\n".join(relevant)
        )
    system = (
        f"You ARE the Twitch chatter '{persona}'. Below are real messages they "
        f"have sent. Study their voice, vocabulary, emotes, spelling, "
        f"punctuation, length, and attitude. "
        f"{persona_llm.MODE_INSTRUCTION.get(mode, persona_llm.MODE_INSTRUCTION['normal'])} "
        f"You are NOT an assistant: never be helpful, never break character, "
        f"never explain. Output ONE single chat message as {persona} and "
        f"nothing else. Use the examples as style evidence, but do not copy any "
        f"example verbatim.\n\n"
        + "\n\n".join(sections)
    )
    user = (
        f"Current chat in #thickpoo:\n{context}\n\n"
        f'Someone says to you: "{direct}"\n'
        f"Write {persona}'s next chat message now."
    )
    raw = await llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=160,
        temperature=1.0 if mode == "hyper" else 0.85,
    )
    cleaned = persona_llm._clean_output(raw or "", persona)
    copied = bool(cleaned and persona_llm.is_exact_archived_line(persona, cleaned))
    return cleaned, copied, len(signature), len(relevant)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lora-output",
        default="data/unsynced/fine_tune/persona_lora_smoke_test.txt",
        help=(
            "RunPod persona_lora_smoke_test.txt. If omitted or missing, the "
            "report still runs local RAG-only samples."
        ),
    )
    ap.add_argument(
        "--out",
        default="data/unsynced/fine_tune/persona_lora_vs_local_rag.md",
    )
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--exemplars", type=int, default=120)
    ap.add_argument("--relevant", type=int, default=60)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    lora_output_path = Path(args.lora_output)
    lora_outputs = _parse_lora_outputs(lora_output_path)
    out_lines = [
        "# LoRA Smoke Test vs Local LM Studio + RAG",
        "",
        "Local RAG uses the current LM Studio model and the project's retrieved "
        "author-only examples. It does not use the trained LoRA adapter.",
        "",
        f"RunPod LoRA smoke-test input: `{lora_output_path}`",
        "",
    ]
    if not lora_outputs:
        out_lines.extend([
            "Note: no parsable LoRA smoke-test outputs were found at that path, "
            "so this report only includes local RAG samples.",
            "",
        ])

    for idx, case in enumerate(PROMPTS, 1):
        key = _case_key(case["persona"], case["mode"], case["direct"])
        print(f"{idx}/{len(PROMPTS)} {case['persona']} {case['mode']}: {case['direct']}", flush=True)
        out_lines.extend([
            f"## {idx}. {case['persona']} ({case['mode']})",
            "",
            f"Direct: `{case['direct']}`",
            "",
            "Context:",
            "```text",
            case["context"],
            "```",
            "",
            f"Old bot output: {case['old_bot']}",
            "",
            "RunPod LoRA-only smoke outputs:",
        ])
        for line in lora_outputs.get(key, []):
            out_lines.append(f"- {line}")
        if not lora_outputs.get(key):
            out_lines.append("- (not found in supplied smoke-test file)")
        out_lines.extend(["", "Local LM Studio + RAG outputs:"])
        for sample_i in range(max(1, args.samples)):
            rag, copied, sig_n, rel_n = await _local_rag(
                case, sample_i, args.exemplars, args.relevant
            )
            suffix = " [ARCHIVE COPY]" if copied else ""
            out_lines.append(
                f"- {rag or '(empty)'}{suffix} "
                f"(examples: {sig_n} random + {rel_n} relevant)"
            )
            print(f"  RAG {sample_i + 1}: {rag}", flush=True)
        out_lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
