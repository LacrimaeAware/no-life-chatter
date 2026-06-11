from utils.chat_archive import nearest_author_lines, said

description = (
    "Search the chat archive: did a user ever say something? Exact first, "
    "then a normalized close-match fallback.\n"
    "  ~said <user> <phrase>   e.g. ~said someuser top 5 worst movies"
)


def _clip(text, n=180):
    return text if len(text) <= n else text[: n - 1] + "…"


async def handle_said(bot, message, params):
    if len(params) < 2:
        await message.channel.send("Usage: ~said <user> <phrase>")
        return
    user, phrase = params[0], " ".join(params[1:])
    total, rows = said(user, phrase, limit=1)
    if total == 0:
        near = nearest_author_lines(user, phrase, limit=1)
        if near:
            score, sent_at, channel, content = near[0]
            await message.channel.send(
                f"No exact record for {user}. Closest ({score:.0%}) on {sent_at[:10]}: "
                f"\"{_clip(content)}\""
            )
            return
        await message.channel.send(f"No record of {user} saying \"{_clip(phrase, 80)}\".")
        return
    sent_at, channel, content = rows[0]
    date = sent_at[:10]
    times = "once" if total == 1 else f"{total} times"
    await message.channel.send(
        f"{user} said that {times} — first on {date}: \"{_clip(content)}\""
    )
