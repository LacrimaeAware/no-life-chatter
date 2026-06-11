"""LLM persona engine: speak as a real chatter, using their own messages.

Many-shot voice cloning (no training): the prompt is ~150 of the person's real
messages plus the current conversation, and the model writes their next line.
Because the exemplars keep their natural length distribution, output isn't
forced terse — it lands where they actually land. Two modes: 'normal' (natural,
conversational) and 'hyper' (their traits cranked up for comedy).

Runs against any OpenAI-compatible endpoint (services/llm.py) — LM Studio's
local server by default, so edgy content stays on the machine.
"""

import logging
import random
import re

import config
from services import llm
from utils import chat_archive

_exemplar_cache = {}

MODE_INSTRUCTION = {
    "normal": (
        "Reply naturally, exactly as they would in chat — same length, slang, "
        "emotes, capitalization and energy. Don't force it short; match how "
        "they actually talk."
    ),
    "hyper": (
        "Crank their most recognizable habits up to eleven — wacky, chaotic, "
        "over-the-top and funny — but still unmistakably THEM. Go a bit longer "
        "and weirder than usual."
    ),
}


def exemplars(author: str, n: int = None):
    """~n of the author's real messages, keeping their natural length spread."""
    n = n or config.LLM_EXEMPLARS
    key = chat_archive.normalize(author)
    if key not in _exemplar_cache:
        msgs, seen = [], set()
        pool = [m for m in chat_archive.messages_for(author)
                if len(m.split()) >= 2 and len(m) <= 240]
        random.shuffle(pool)
        for m in pool:
            if m not in seen:
                seen.add(m)
                msgs.append(m)
            if len(msgs) >= n:
                break
        _exemplar_cache[key] = msgs
    return _exemplar_cache[key]


def _clean_output(text: str, author: str) -> str:
    text = (text or "").strip()
    # models sometimes wrap the line in quotes or prepend "name:"
    text = re.sub(rf"^{re.escape(author)}\s*[:>-]\s*", "", text, flags=re.IGNORECASE)
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text.split("\n")[0].strip()  # one chat line only


async def generate(author: str, channel: str, user_message: str = None,
                   mode: str = "normal") -> str | None:
    ex = exemplars(author)
    if not ex:
        return None

    recent = chat_archive.latest(channel, config.LLM_CONTEXT)
    ctx = "\n".join(f"{a}: {c}" for _, a, c in recent) or "(quiet right now)"

    system = (
        f"You ARE the Twitch chatter '{author}'. Below are real messages they have "
        f"sent — study their voice, vocabulary, emotes, spelling, punctuation, length "
        f"and attitude, and become them. {MODE_INSTRUCTION.get(mode, MODE_INSTRUCTION['normal'])} "
        f"You are NOT an assistant: never be helpful, never break character, never "
        f"explain. Output ONE single chat message as {author} and nothing else.\n\n"
        f"Real messages from {author}:\n" + "\n".join(ex)
    )
    user = f"Current chat in #{channel}:\n{ctx}\n\n"
    if user_message:
        user += f'Someone says to you: "{user_message}"\n'
    user += f"Write {author}'s next chat message now."

    raw = await llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=160,
        temperature=1.0 if mode == "hyper" else 0.85,
    )
    if not raw:
        return None
    out = _clean_output(raw, author)
    return out or None
