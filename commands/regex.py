from utils.chat_archive import regex_search

description = (
    "Regex-search the chat archive (case-insensitive). User can be a name, or "
    "'*'/'anyone' for everyone.\n"
    "  ~regex <user|anyone> <pattern>   e.g. ~regex someuser fi+d  |  ~regex anyone lo+l"
)


def _clip(text, n=140):
    return text if len(text) <= n else text[: n - 1] + "…"


async def handle_regex(bot, message, params):
    if len(params) < 2:
        await message.channel.send("Usage: ~regex <user|anyone> <pattern>")
        return
    user, pattern = params[0].lstrip("@"), " ".join(params[1:])
    rows = regex_search(pattern, author=user, limit=3)
    if rows is None:
        await message.channel.send(f"Bad regex: {_clip(pattern, 60)}")
        return
    everyone = user.lower() in ("*", "anyone", "everyone")
    if not rows:
        who = "anyone" if everyone else user
        await message.channel.send(f"No match for /{_clip(pattern, 60)}/ by {who}.")
        return
    if everyone:
        parts = [f"{a}: \"{_clip(c, 90)}\"" for _s, _ch, a, c in rows]
    else:
        parts = [f"\"{_clip(c, 120)}\"" for _s, _ch, _a, c in rows]
    await message.channel.send("🔎 " + " | ".join(parts))
