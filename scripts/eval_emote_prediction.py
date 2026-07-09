"""Emote-prediction dial: can a model pick the emote a chatter actually used?

Masks the (single) registry emote in real archived messages and asks the
local LLM to choose it from a lineup of distractors. Every archived message
with an emote is a free labeled example, so this is an unlimited,
self-supervised validation target for "does the system understand emotes" —
usable to compare prompts, models, or a LoRA before/after.

Baselines reported alongside accuracy:
  - chance        = 1/len(options)
  - frequency     = always guess the globally most-common emote in the lineup

    python scripts/eval_emote_prediction.py                 # n=40
    python scripts/eval_emote_prediction.py --n 100 --options 8
    python scripts/eval_emote_prediction.py --author someuser   # one chatter
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import chat_archive, emote_meaning  # noqa: E402
from services import llm  # noqa: E402

_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def _emote_frequencies(registry: set[str], sample: int = 120_000) -> Counter:
    """Global usage counts for registry emotes over a random archive sample."""
    conn = chat_archive.connect()
    counts: Counter = Counter()
    for (content,) in conn.execute(
            "SELECT content FROM messages ORDER BY RANDOM() LIMIT ?", (sample,)):
        for tok in (content or "").split():
            if tok in registry:
                counts[tok] += 1
    return counts


def _eval_items(registry: set[str], n: int, author: str | None,
                min_words: int, seed: int) -> list[dict]:
    """Messages containing exactly ONE registry emote plus >=min_words words."""
    conn = chat_archive.connect()
    rng = random.Random(seed)
    if author:
        keys = chat_archive.author_keys(author)
        ph, params = chat_archive._in_clause(keys)
        sql = (f"SELECT author, content FROM messages WHERE author IN ({ph}) "
               "ORDER BY RANDOM() LIMIT 60000")
    else:
        sql, params = "SELECT author, content FROM messages ORDER BY RANDOM() LIMIT 60000", []
    items, seen = [], set()
    for row_author, content in conn.execute(sql, params):
        content = content or ""
        if content.lstrip().startswith("~") or "http" in content:
            continue
        toks = content.split()
        emotes = [t for t in toks if t in registry]
        if len(emotes) != 1:
            continue
        words = [t for t in toks if t not in registry and _WORD_RE.fullmatch(t)]
        if len(words) < min_words:
            continue
        key = chat_archive.line_match_key(content)
        if not key or key in seen:
            continue
        seen.add(key)
        masked = " ".join("____" if t == emotes[0] else t for t in toks)
        items.append({"author": chat_archive.normalize_author(row_author),
                      "masked": masked, "answer": emotes[0]})
        if len(items) >= n * 3:
            break
    rng.shuffle(items)
    return items[:n]


async def _ask(item: dict, options: list[str]) -> str | None:
    numbered = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(options))
    out = await llm.chat(
        [{"role": "system", "content":
            "You are an expert on Twitch chat culture and 7TV emote usage. "
            "An emote was removed from a real chat message (shown as ____). "
            "Pick which of the listed emotes the chatter actually used. "
            "Answer with the emote name only, nothing else."},
         {"role": "user", "content":
            f"Message by {item['author']}:\n{item['masked']}\n\nOptions:\n{numbered}\n\nAnswer:"}],
        max_tokens=20, temperature=0.0,
    )
    if not out:
        return None
    out = out.strip().splitlines()[0].strip().strip(".")
    # accept "3", "3.", "OMEGALUL", "3. OMEGALUL"
    m = re.match(r"^(\d{1,2})\b", out)
    if m and 1 <= int(m.group(1)) <= len(options):
        return options[int(m.group(1)) - 1]
    for opt in options:
        if opt.casefold() in out.casefold():
            return opt
    return None


async def run(args) -> int:
    registry = set(emote_meaning.registry())
    if not registry:
        print("No emote registry (run scripts/build_emote_registry.py first).")
        return 1
    freqs = _emote_frequencies(registry)
    common = [e for e, _ in freqs.most_common(150)]
    if len(common) < args.options:
        print(f"Only {len(common)} emotes with usage data — need >= {args.options}.")
        return 1
    items = _eval_items(registry, args.n, args.author, args.min_words, args.seed)
    if not items:
        print("No eligible messages found.")
        return 1

    rng = random.Random(args.seed)
    correct = freq_correct = answered = 0
    for i, item in enumerate(items, 1):
        distractors = [e for e in common if e != item["answer"]]
        options = rng.sample(distractors, args.options - 1) + [item["answer"]]
        rng.shuffle(options)
        guess = await _ask(item, options)
        if guess is None:
            continue
        answered += 1
        freq_guess = max(options, key=lambda e: freqs.get(e, 0))
        freq_correct += (freq_guess == item["answer"])
        hit = guess == item["answer"]
        correct += hit
        print(f"[{i:>3}] {'Y' if hit else 'n'} true={item['answer']:<18} "
              f"guess={guess:<18} | {item['masked'][:70]}")

    if not answered:
        print("LLM unreachable — no items answered.")
        return 1
    print(f"\nanswered {answered}/{len(items)}   options per item: {args.options}")
    print(f"model accuracy:     {correct / answered:.1%}   ({correct}/{answered})")
    print(f"frequency baseline: {freq_correct / answered:.1%}")
    print(f"chance baseline:    {1 / args.options:.1%}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--options", type=int, default=8)
    ap.add_argument("--author", default=None)
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
