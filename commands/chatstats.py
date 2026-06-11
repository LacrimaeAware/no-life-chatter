from utils.chat_archive import stats

description = (
    "Archive stats for a user: message count, first/last seen, busiest hour.\n"
    "  ~chatstats <user>"
)


async def handle_chatstats(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~chatstats <user>")
        return
    s = stats(params[0])
    if not s:
        await message.channel.send(f"Nothing archived for {params[0]} yet.")
        return
    hour = f"{s['busiest_hour']:02d}:00" if s["busiest_hour"] is not None else "?"
    await message.channel.send(
        f"{params[0]}: {s['messages']:,} messages archived, first seen {s['first'][:10]}, "
        f"last {s['last'][:10]}, avg {s['avg_chars']} chars, busiest hour {hour}"
    )
