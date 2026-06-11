"""LLM persona engine: speak as a real chatter, using their own messages.

Many-shot voice cloning (no training): the prompt blends a random signature
sample from the person's full history with messages retrieved from that same
author for the current chat topic. The model sees those examples plus the live
conversation, then writes their next line. Because the exemplars keep their
natural length distribution, output isn't forced terse — it lands where they
actually land. Two modes: 'normal' (natural, conversational) and 'hyper' (their
traits cranked up for comedy).

Runs against any OpenAI-compatible endpoint (services/llm.py) — LM Studio's
local server by default, so edgy content stays on the machine.
"""

import random
import re
import logging

import config
from services import llm
from utils import chat_archive

_exemplar_cache = {}
_archive_line_cache = {}
_last_rejection = None


class _CopiedPersonaOutput(Exception):
    pass

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


def _usable_exemplar(message: str) -> bool:
    return bool(
        message
        and len(message.split()) >= 2
        and len(message) <= 240
        and not message.lstrip().startswith(config.PREFIX)
    )


def _unique_messages(messages, n: int, seen=None):
    if n <= 0:
        return []
    seen = seen or set()
    out = []
    for message in messages:
        if not _usable_exemplar(message) or message in seen:
            continue
        seen.add(message)
        out.append(message)
        if len(out) >= n:
            break
    return out


def exemplars(author: str, n: int = None):
    """~n random messages from the author, across their whole history."""
    n = n or config.LLM_EXEMPLARS
    key = (chat_archive.normalize_author(author), n)
    if key not in _exemplar_cache:
        pool = [m for m in chat_archive.messages_for(author) if _usable_exemplar(m)]
        random.shuffle(pool)
        _exemplar_cache[key] = _unique_messages(pool, n)
    return _exemplar_cache[key]


def relevant_exemplars(author: str, query_text: str, n: int = None):
    """Author-only examples relevant to the current chat topic."""
    n = n if n is not None else getattr(config, "LLM_RELEVANT_EXEMPLARS", 0)
    if n <= 0 or not (query_text or "").strip():
        return []
    rows = chat_archive.search_author(author, query_text, limit=max(n * 4, 20))
    return _unique_messages((content for _, _, content in rows), n)


def select_exemplars(author: str, query_text: str, n: int = None,
                     relevant_n: int = None):
    """Blend stable random voice samples with per-call retrieved examples."""
    n = n or config.LLM_EXEMPLARS
    relevant_budget = (
        getattr(config, "LLM_RELEVANT_EXEMPLARS", 0)
        if relevant_n is None else relevant_n
    )
    relevant_target = min(n, max(0, relevant_budget), max(0, int(n * 0.6)))
    relevant = relevant_exemplars(author, query_text, relevant_target)

    seen = set(relevant)
    signature = _unique_messages(exemplars(author, n), n - len(relevant), seen)
    return signature, relevant


def _conversation_rows(recent):
    return [
        row for row in recent
        if not (row[2] or "").lstrip().startswith(config.PREFIX)
    ]


def _retrieval_text(recent, user_message: str | None) -> str:
    parts = []
    if user_message:
        parts.append(user_message)
    # Use content only, not author labels; names and command words are noisy
    # retrieval anchors, while the actual message text carries the topic.
    for _, _, content in _conversation_rows(recent)[-12:]:
        parts.append(content)
    return "\n".join(parts)


def _clean_output(text: str, author: str) -> str:
    text = (text or "").strip()
    # models sometimes wrap the line in quotes or prepend "name:"
    text = re.sub(rf"^{re.escape(author)}\s*[:>-]\s*", "", text, flags=re.IGNORECASE)
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text.split("\n")[0].strip()  # one chat line only


def _copy_key(text: str) -> str:
    return chat_archive.line_match_key(text)


def last_rejection() -> str | None:
    return _last_rejection


def _set_rejection(reason: str | None) -> None:
    global _last_rejection
    _last_rejection = reason


def is_exact_archived_line(author: str, text: str) -> bool:
    """True when generated text is a normalized exact old line from this author."""
    key = chat_archive.normalize_author(author)
    if key not in _archive_line_cache:
        _archive_line_cache[key] = {
            _copy_key(message)
            for message in chat_archive.messages_for(author)
            if _copy_key(message)
        }
    return _copy_key(text) in _archive_line_cache[key]


def _near_example_copy(text: str, examples) -> str | None:
    """Return the copied example when output is too close to a prompt line."""
    for example in examples:
        if chat_archive.line_similarity(text, example) >= 0.94:
            return example
    return None


def _copied_source(author: str, text: str, examples) -> str | None:
    if not text:
        return None
    if is_exact_archived_line(author, text):
        return text
    return _near_example_copy(text, examples)


async def _repair_copied_output(author: str, channel: str, user_message: str | None,
                                mode: str, copied_output: str, copied_source: str,
                                signature, relevant, ctx: str) -> str | None:
    examples = _unique_messages(
        [*relevant[:10], *signature[:10]],
        16,
        seen={copied_source, copied_output},
    )
    if not examples:
        return None
    system = (
        f"You are rewriting a Twitch persona line for '{author}'. The previous "
        f"draft copied an archived line too closely. Write ONE new chat message "
        f"in {author}'s voice. Do not quote, paraphrase, or reuse the copied "
        f"line. Stay in character and output only the message."
    )
    user = (
        f"Current chat in #{channel}:\n{ctx}\n\n"
        f"Message directed at the persona: {user_message or '(none)'}\n\n"
        f"Copied draft to avoid:\n{copied_output}\n\n"
        f"Archived line it was too close to:\n{copied_source}\n\n"
        f"Small style sample from {author}:\n" + "\n".join(examples)
        + f"\n\nWrite a new {author} chat line now."
    )
    raw = await llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=120,
        temperature=1.05 if mode == "hyper" else 0.95,
    )
    repaired = _clean_output(raw, author)
    if not repaired:
        return None
    if _copied_source(author, repaired, examples):
        logging.info("Rejected copied persona repair for %s: %r", author, repaired)
        return None
    return repaired


async def generate(author: str, channel: str, user_message: str = None,
                   mode: str = "normal", exemplar_count: int = None,
                   context_count: int = None,
                   copy_strategy: str = "drop") -> str | None:
    exemplar_count = exemplar_count or config.LLM_EXEMPLARS
    context_count = context_count or config.LLM_CONTEXT
    recent = chat_archive.latest(channel, context_count)
    ctx_rows = _conversation_rows(recent)
    ctx = "\n".join(f"{a}: {c}" for _, a, c in ctx_rows) or "(quiet right now)"
    signature, relevant = select_exemplars(
        author, _retrieval_text(recent, user_message), n=exemplar_count
    )
    if not signature and not relevant:
        return None

    exemplar_sections = []
    if signature:
        exemplar_sections.append(
            f"Random real messages from {author} across their whole history:\n"
            + "\n".join(signature)
        )
    if relevant:
        exemplar_sections.append(
            f"Real messages from {author} relevant to this chat/topic:\n"
            + "\n".join(relevant)
        )

    system = (
        f"You ARE the Twitch chatter '{author}'. Below are real messages they have "
        f"sent — study their voice, vocabulary, emotes, spelling, punctuation, length "
        f"and attitude, and become them. {MODE_INSTRUCTION.get(mode, MODE_INSTRUCTION['normal'])} "
        f"You are NOT an assistant: never be helpful, never break character, never "
        f"explain. Output ONE single chat message as {author} and nothing else. "
        f"Use the examples as style evidence, but do not copy any example verbatim; "
        f"write a new line in their voice unless the user explicitly asked for a quote. "
        f"All examples below are from {author} only; use the relevant examples "
        f"to understand what they tend to say in this situation.\n\n"
        + "\n\n".join(exemplar_sections)
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
    copied_example = _copied_source(author, out, [*signature, *relevant])
    if copied_example:
        logging.info(
            "Rejected copied persona output for %s: %r copied from %r",
            author, out, copied_example,
        )
        if copy_strategy == "repair":
            repaired = await _repair_copied_output(
                author, channel, user_message, mode, out, copied_example,
                signature, relevant, ctx,
            )
            if repaired:
                return repaired
            raise _CopiedPersonaOutput
        return None
    return out or None


async def generate_with_retry(author: str, channel: str, user_message: str = None,
                              mode: str = "normal") -> str | None:
    """Generate once with the full prompt, then retry compactly on failure.

    Local LM Studio can time out on heavy prompts, especially when two commands
    land close together. The compact retry keeps commands responsive without
    disabling the richer default prompt for normal cases.
    """
    _set_rejection(None)
    try:
        out = await generate(
            author, channel, user_message, mode=mode, copy_strategy="repair"
        )
    except _CopiedPersonaOutput:
        _set_rejection("model copied an archived line and the cheap repair failed")
        return None
    if out:
        return out
    if not chat_archive.messages_for(author):
        return None
    retry_exemplars = getattr(config, "LLM_RETRY_EXEMPLARS", 0)
    retry_context = getattr(config, "LLM_RETRY_CONTEXT", 0)
    if retry_exemplars <= 0:
        return None
    if retry_exemplars >= config.LLM_EXEMPLARS and retry_context >= config.LLM_CONTEXT:
        return None
    return await generate(
        author,
        channel,
        user_message,
        mode=mode,
        exemplar_count=retry_exemplars,
        context_count=retry_context,
        copy_strategy="drop",
    )
