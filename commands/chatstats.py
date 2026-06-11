from utils.chat_archive import stats
from commands.markers import _parse_scope

description = (
    "Archive stats for a user: message count, first/last seen, busiest hour. "
    "All chats by default; chat=<channel> scopes to one.\n"
    "  ~chatstats <user> [chat=<channel>]"
)


async def handle_chatstats(bot, message, params):
    # default = all chats (unlike ~markers): totals are the point here
    params, channel, _year = _parse_scope(params, None)
    if not params:
        await message.channel.send("Usage: ~chatstats <user> [chat=<channel>]")
        return
    user = params[0].lstrip("@")
    s = stats(user, channel=channel)
    where = f" in #{channel}" if channel else ""
    if not s:
        await message.channel.send(f"Nothing archived for {user}{where} yet.")
        return
    hour = f"{s['busiest_hour']:02d}:00" if s["busiest_hour"] is not None else "?"
    await message.channel.send(
        f"{user}{where}: {s['messages']:,} messages archived, first seen {s['first'][:10]}, "
        f"last {s['last'][:10]}, avg {s['avg_chars']} chars, busiest hour {hour}"
    )
