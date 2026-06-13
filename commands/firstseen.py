from utils.chat_archive import first_seen, normalize_author

description = (
    "First archived message from a user.\n"
    "  ~firstseen <user>"
)


async def handle_firstseen(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~firstseen <user>")
        return
    params[0] = normalize_author(params[0])
    row = first_seen(params[0])
    if not row:
        await message.channel.send(f"Nothing archived for {params[0]} yet.")
        return
    sent_at, channel, content = row
    if len(content) > 250:
        content = content[:249] + "…"
    await message.channel.send(
        f"First sighting of {params[0]}: {sent_at[:10]} in #{channel} — \"{content}\""
    )
