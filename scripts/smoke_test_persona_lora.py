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
    {
        "persona": "99froxy",
        "mode": "normal",
        "direct": "@99froxy thoughts on gero_30? Lime",
        "context": (
            "earnestsinceresugmamale: thoughts on gero_30? Lime\n"
            "forsenstares: be honest\n"
            "gero_30: ?????\n"
            "apu_wrc: Lime"
        ),
        "old_bot": "99froxy: he just dropped a bomb and no one noticed Lime",
    },
    {
        "persona": "ebbel",
        "mode": "normal",
        "direct": "@ebbel hows world of warcraft been treating you?",
        "context": (
            "forsenstares: FeelsOkayMan\n"
            "earnestsinceresugmamale: he do be liking Cinema\n"
            "forsenstares: try ebbel\n"
            "earnestsinceresugmamale: PauseChamp"
        ),
        "old_bot": "Couldn't generate ebbel",
    },
    {
        "persona": "ebbel",
        "mode": "normal",
        "direct": "@ebbel why are you typing like that",
        "context": (
            "forsenstares: why are you typing like that\n"
            "ebbel: Phone.\n"
            "gero_30: eat something man\n"
            "forsenstares: boomer"
        ),
        "old_bot": "ebbel: im hungry but i dont wanna make something",
    },
    {
        "persona": "gero_30",
        "mode": "normal",
        "direct": "@gero_30 movie tonight?",
        "context": (
            "earnestsinceresugmamale: unlike gero_30 PokShadowmew\n"
            "earnestsinceresugmamale: its not gonna be as accurate ITSOVER\n"
            "earnestsinceresugmamale: heMightBeRight\n"
            "earnestsinceresugmamale: he does hate cops"
        ),
        "old_bot": "gero_30: gangs of new york very youtube shorts type movie",
    },
    {
        "persona": "gero_30",
        "mode": "normal",
        "direct": "@gero_30 why are you fascinated with bots repeating our lines dathinker",
        "context": (
            "forsenstares: its like the bot narded all the personas\n"
            "gero_30: why are you fascinated with bots repeating our lines dathinker\n"
            "earnestsinceresugmamale: heMightBeRight\n"
            "earnestsinceresugmamale: but im going to feed it all of our logs since we were born fernardoAnalysis"
        ),
        "old_bot": "gero_30: ive been saying that for hours and u just repeat it back like a parrot wtf is wrong with u @forsenstares",
    },
    {
        "persona": "gero_30",
        "mode": "hyper",
        "direct": "@gero_30 dobray ray no",
        "context": (
            "forsenstares: dobray ray no\n"
            "earnestsinceresugmamale: what does that even mean\n"
            "apu_wrc: Lemon\n"
            "forsenstares: explain yourself"
        ),
        "old_bot": "gero_30: you just dobray'd your last hand-no buddy",
    },
    {
        "persona": "forsenstares",
        "mode": "normal",
        "direct": "@forsenstares what do you think about the new patch",
        "context": (
            "gero_30: $brb\n"
            "supibot: gero_30 is going to be right back\n"
            "earnestsinceresugmamale: ok turing test was that me or ai\n"
            "gero_30: YOU wrote it"
        ),
        "old_bot": "forsenstares: i havent played in like 3 months so idk what u r talking about @gero_30 is back tho wooo tickpoo",
    },
    {
        "persona": "forsenstares",
        "mode": "hyper",
        "direct": "@forsenstares why are you fascinated with bots repeating our lines dathinker",
        "context": (
            "earnestsinceresugmamale: LOLOL this is them @forsenstares\n"
            "forsenstares: LOLOL\n"
            "earnestsinceresugmamale: they share consciousness now\n"
            "gero_30: ?????"
        ),
        "old_bot": "forsenstares: NO DID YOU?",
    },
    {
        "persona": "earnestsinceresugmamale",
        "mode": "normal",
        "direct": "@earnestsinceresugmamale thoughts on AI",
        "context": (
            "earnestsinceresugmamale: I'm training it on the mega AI gpu servers rn Lemon TeaTime\n"
            "forsenstares: you mean your gpu that ran fatekeeper at 40 fps\n"
            "earnestsinceresugmamale: no im renting elons personal gpu\n"
            "forsenstares: monkaS"
        ),
        "old_bot": "earnestsinceresugmamale: i thought u guys were done stealing my lines @forsenstares @ebbel but i guess u cant resist the power of truth and actuality Lemon",
    },
    {
        "persona": "earnestsinceresugmamale",
        "mode": "hyper",
        "direct": "@earnestsinceresugmamale im so Ogre today",
        "context": (
            "forsenstares: monkaS\n"
            "earnestsinceresugmamale: monkaS\n"
            "forsenstares: you mean your gpu that ran fatekeeper at 40 fps\n"
            "ebbel: Idk"
        ),
        "old_bot": "earnestsinceresugmamale: $remind forsenstares https://streamable.com/6qwfwv OOOO ITSOVER OGREFIED",
    },
    {
        "persona": "fernardo",
        "mode": "normal",
        "direct": "@fernardo whats your favorite thing in the whole wide world?",
        "context": (
            "forsenstares: wooo tickpoo\n"
            "earnestsinceresugmamale: it was appropriate tickpooJAWLINE\n"
            "forsenstares: yes atleast it paired it will\n"
            "earnestsinceresugmamale: just block him"
        ),
        "old_bot": "fernardo: hahahaha that ur asking me that LMAOOOOOOOOOO @forsenstares is right here u can just ask him since hes always repeating my lines",
    },
    {
        "persona": "bluepigman5000",
        "mode": "hyper",
        "direct": "@bluepigman5000 are you a diamond pig?",
        "context": (
            "earnestsinceresugmamale: y u heff to get mad? @apu_wrc Lemon\n"
            "apu_wrc: Lemon it's just game\n"
            "forsenstares: try bluepig\n"
            "gero_30: ?????"
        ),
        "old_bot": "bluepigman5000: ARE YOU FOR REAL THAT'S THE BEST INSULT YOU GOT COOKED",
    },
    {
        "persona": "bander423",
        "mode": "normal",
        "direct": "@bander423 is sushi good?",
        "context": (
            "forsenstares: thats like saying is sushi good\n"
            "earnestsinceresugmamale: oh right\n"
            "forsenstares: we finally have the answer FeelsStrongMan\n"
            "apu_wrc: Lime"
        ),
        "old_bot": "bander423: its a good map",
    },
]


MODE_EXTRA = {
    "normal": (
        "Reply naturally. If the direct message is boring, it is fine to be "
        "boring, dismissive, confused, or short."
    ),
    "hyper": (
        "Hyper mode: exaggerate the persona's recognizable habits for comedy, "
        "but keep it as a plausible Twitch chat line from that person."
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="/workspace/nlc_persona/persona_lora")
    ap.add_argument("--out", default="/workspace/nlc_persona/persona_lora_smoke_test.txt")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--hyper-temperature", type=float, default=1.05)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--samples-per-prompt", type=int, default=2)
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
    for case in PROMPTS:
        persona = case["persona"]
        context = case["context"]
        direct = case["direct"]
        mode = case["mode"]
        user = (
            f"<persona={persona}>\n"
            f"Recent chat in #thickpoo:\n{context}\n\n"
            f'Someone says to you: "{direct}"\n'
            f"Write {persona}'s next chat message."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + " " + MODE_EXTRA[mode]},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_results = []
        for sample_i in range(max(1, args.samples_per_prompt)):
            torch.manual_seed(args.seed + (len(results) * 17) + sample_i)
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
            ).strip()
            line = generated.splitlines()[0].strip() if generated else ""
            prompt_results.append(line)
            print(f"{persona} [{mode}] #{sample_i + 1}: {line}", flush=True)
        results.append((case, user, prompt_results))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("NoLifeChatter persona LoRA smoke test\n")
        f.write(f"adapter: {adapter}\n\n")
        for case, user, lines in results:
            f.write("=" * 72 + "\n")
            f.write(f"persona: {case['persona']}\n")
            f.write(f"mode: {case['mode']}\n")
            f.write(f"direct: {case['direct']}\n")
            f.write(f"old bot output: {case['old_bot']}\n\n")
            f.write(user + "\n\n")
            f.write("outputs:\n")
            for i, line in enumerate(lines, 1):
                f.write(f"{i}. {line}\n")
            f.write("\n")
    print(f"\nSaved smoke-test outputs to {out}", flush=True)


if __name__ == "__main__":
    main()
