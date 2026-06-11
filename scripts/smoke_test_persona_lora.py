"""Smoke-test a trained NoLifeChatter persona LoRA on a GPU machine.

This is meant for RunPod after training finishes. It loads the LoRA adapter,
generates a few short persona replies with the same chat-style prompt shape
used during SFT, and writes the results to a text file for inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a local Twitch chat persona model. The active persona is written "
    "as <persona=name>. Given recent chat, write the next single Twitch chat "
    "message from that persona. Match their style, vocabulary, casing, emotes, "
    "punctuation, humor, and usual length. Do not explain. Output one chat "
    "message only."
)


PROMPTS = [
    (
        "ebbel",
        "forsenstares: why are you typing like that\n"
        "ebbel: Phone.\n"
        "gero_30: eat something man\n"
        "forsenstares: boomer",
        "Write ebbel's next chat message.",
    ),
    (
        "gero_30",
        "forsenstares: dobray ray no\n"
        "earnestsinceresugmamale: what does that even mean\n"
        "apu_wrc: Lemon\n"
        "forsenstares: explain yourself",
        "Write gero_30's next chat message.",
    ),
    (
        "forsenstares",
        "earnestsinceresugmamale: is this thing actually learning\n"
        "gero_30: maybe\n"
        "ebbel: Idk\n"
        "apu_wrc: Lemon",
        "Write forsenstares's next chat message.",
    ),
    (
        "earnestsinceresugmamale",
        "ebbel: what model do you use\n"
        "forsenstares: ask the bot\n"
        "gero_30: WAITING\n"
        "apu_wrc: Lemon",
        "Write earnestsinceresugmamale's next chat message.",
    ),
    (
        "99froxy",
        "earnestsinceresugmamale: thoughts on gero_30\n"
        "forsenstares: be honest\n"
        "gero_30: ?????\n"
        "apu_wrc: Lime",
        "Write 99froxy's next chat message.",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="/workspace/nlc_persona/persona_lora")
    ap.add_argument("--out", default="/workspace/nlc_persona/persona_lora_smoke_test.txt")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch

    adapter = Path(args.adapter)
    if not adapter.exists():
        raise SystemExit(f"Adapter path not found: {adapter}")

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
    for persona, context, instruction in PROMPTS:
        user = (
            f"<persona={persona}>\n"
            f"Recent chat in #thickpoo:\n{context}\n\n"
            f"{instruction}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()
        line = generated.splitlines()[0].strip() if generated else ""
        results.append((persona, user, line))
        print(f"{persona}: {line}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("NoLifeChatter persona LoRA smoke test\n")
        f.write(f"adapter: {adapter}\n\n")
        for persona, user, line in results:
            f.write("=" * 72 + "\n")
            f.write(f"persona: {persona}\n\n")
            f.write(user + "\n\n")
            f.write("output:\n")
            f.write(line + "\n\n")
    print(f"\nSaved smoke-test outputs to {out}", flush=True)


if __name__ == "__main__":
    main()
