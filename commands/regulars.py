from utils.chat_archive import channel_regulars, display_name

description = (
    "Top regulars in a channel, ignoring obvious bots by default.\n"
    "  ~regulars [channel] [min_messages] [limit]"
)


def _fmt_n(n):
    return f"{n:,}"


async def handle_regulars(bot, message, params):
    channel = message.channel.name if message.channel else ""
    min_messages = 5000
    limit = 10

    if params:
        if params[0].isdigit():
            min_messages = int(params[0])
        else:
            channel = params[0].lstrip("#").lower()
            if len(params) >= 2 and params[1].isdigit():
                min_messages = int(params[1])
            if len(params) >= 3 and params[2].isdigit():
                limit = int(params[2])

    limit = max(1, min(limit, 15))
    min_messages = max(1, min_messages)
    rows = channel_regulars(channel, min_messages=min_messages, limit=limit)
    if not rows:
        await message.channel.send(f"No regulars found for #{channel} above {_fmt_n(min_messages)} messages.")
        return

    parts = [f"{display_name(author)} {_fmt_n(count)}" for author, count in rows]
    await message.channel.send(
        f"#{channel} regulars >= {_fmt_n(min_messages)} msgs: " + " | ".join(parts)
    )
