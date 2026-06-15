import asyncio

from utils.chat_archive import (
    normalize_author,
    normalize_channel,
    random_match,
    strip_quoted_log_prefix,
)

description = (
    "A random real quote from the archive containing a word/phrase. Anyone by "
    "default; narrow it with user=<name> and/or chat=<channel>.\n"
    "  ~random <word>   ·   ~random user=duardo cars   ·   ~random chat=forsen fart"
)


def _clip(text, n=260):
    return text if len(text) <= n else text[: n - 1] + "…"


def _snippet(content, phrase, n=260):
    """Show a window of `content` centered on the matched term so the hit is
    visible even when it sits deep in a long message. Falls back to a head clip
    if the term can't be located (e.g. FTS matched a stemmed form)."""
    body = content.strip()
    if len(body) <= n:
        return body
    low = body.lower()
    terms = [phrase.strip().lower()] + sorted(
        (w.lower() for w in phrase.split() if len(w) >= 2), key=len, reverse=True)
    pos = next((low.find(t) for t in terms if t and low.find(t) != -1), -1)
    if pos == -1:
        return body[: n - 1] + "…"
    start = max(0, pos - n // 3)
    end = min(len(body), start + n)
    chunk = body[start:end]
    if start > 0:
        chunk = "…" + chunk
    if end < len(body):
        chunk = chunk + "…"
    return chunk


def _parse(params):
    user = channel = None
    rest = []
    for p in params:
        low = p.lower()
        if low.startswith(("user=", "by=")):
            user = p.split("=", 1)[1].strip().lstrip("@") or None
        elif low.startswith("chat="):
            value = p.split("=", 1)[1].strip().lstrip("#")
            channel = None if value in ("", "*", "all") else value
        else:
            rest.append(p)
    return user, channel, " ".join(rest).strip()


async def handle_random(bot, message, params):
    user, channel, phrase = _parse(params or [])
    if not phrase:
        await message.channel.send("Usage: ~random <word> [user=<name>] [chat=<channel>]")
        return
    row = await asyncio.to_thread(random_match, phrase, user, channel)
    if not row:
        scope = (f" by {normalize_author(user)}" if user else "")
        scope += (f" in #{normalize_channel(channel)}" if channel else "")
        await message.channel.send(f"No archived quote with \"{_clip(phrase, 60)}\"{scope}.")
        return
    sent_at, ch, author, content = row
    # drop a pasted 'HH:MM Name:' chatlog-quote prefix; keep the date in the header
    content, quoted_name = strip_quoted_log_prefix(content)
    if not author and quoted_name:
        author = normalize_author(quoted_name)
    who = f"{author}#{ch}" if author else f"#{ch}"
    await message.channel.send(f"🎲 {who} {sent_at[:10]}: \"{_snippet(content, phrase)}\"")
