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
    await message.channel.send(f"🎲 {who} {sent_at[:10]}: \"{_clip(content)}\"")
