from utils.chat_archive import author_name_search, normalize_channel

description = (
    "Find archived usernames matching a regex. Defaults to all chats; "
    "chat=<channel> scopes it.\n"
    "  ~userregex <pattern> [chat=<channel>]"
)


def _clip(text, n=80):
    return text if len(text) <= n else text[: n - 1] + "..."


def _parse_chat_scope(params):
    rest, channel = [], None
    for p in params:
        low = p.lower()
        if low.startswith("chat="):
            value = low.split("=", 1)[1].strip().lstrip("#")
            channel = None if value in ("", "*", "all") else value
        else:
            rest.append(p)
    return rest, channel


async def handle_userregex(bot, message, params):
    params, channel = _parse_chat_scope(params)
    if not params:
        await message.channel.send("Usage: ~userregex <pattern> [chat=<channel>]")
        return
    pattern = " ".join(params)
    rows = author_name_search(pattern, channel=channel, limit=12)
    if rows is None:
        await message.channel.send(f"Bad username regex: {_clip(pattern, 60)}")
        return
    scope = f" in #{normalize_channel(channel)}" if channel else ""
    if not rows:
        await message.channel.send(f"No archived usernames matching /{_clip(pattern, 60)}/{scope}.")
        return
    parts = [
        f"{author} ({count:,}; {first[:10]}..{last[:10]})"
        for author, count, first, last in rows
    ]
    await message.channel.send(f"Users /{_clip(pattern, 50)}/{scope}: " + " | ".join(parts))
