import asyncio

from utils.chat_archive import said_most, normalize_channel, display_name

description = (
    "~saidmost <phrase> [chat=<channel>] [n] — who has said a phrase the most "
    "(ranked by how many of their messages contain it)."
)


def _clip(text, n=50):
    return text if len(text) <= n else text[: n - 1] + "…"


def _parse(params):
    channel = None
    rest = []
    for p in params:
        low = p.lower()
        if low.startswith("chat="):
            value = low.split("=", 1)[1].strip().lstrip("#")
            channel = None if value in ("", "*", "all") else value
        else:
            rest.append(p)
    n = 5
    if len(rest) > 1 and rest[-1].isdigit():
        n = max(1, min(int(rest.pop()), 15))
    return channel, n, " ".join(rest).strip()


async def handle_saidmost(bot, message, params):
    channel, n, phrase = _parse(params or [])
    if not phrase:
        await message.channel.send("Usage: ~saidmost <phrase> [chat=<channel>] [n]")
        return
    rows = await asyncio.to_thread(said_most, phrase, channel, n)
    if not rows:
        scope = f" in #{normalize_channel(channel)}" if channel else ""
        await message.channel.send(f"Nobody on record saying \"{_clip(phrase)}\"{scope}.")
        return
    scope = f" in #{normalize_channel(channel)}" if channel else ""
    parts = [f"{i}. {display_name(author)} ({count})"
             for i, (author, count) in enumerate(rows, 1)]
    await message.channel.send(
        f"Most \"{_clip(phrase, 40)}\"{scope}: " + " · ".join(parts)
    )
