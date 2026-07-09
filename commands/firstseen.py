from utils.chat_archive import first_seen, normalize_author, display_name

description = (
    "First archived message from a user.\n"
    "  ~firstseen <user>"
)


async def handle_firstseen(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~firstseen <user>")
        return
    user = normalize_author(params[0])
    display = display_name(user)
    row = first_seen(user)
    if not row:
        await message.channel.send(f"Nothing archived for {display} yet.")
        return
    sent_at, channel, content = row
    if len(content) > 250:
        content = content[:249] + "…"
    await message.channel.send(
        f"First sighting of {display}: {sent_at[:10]} in #{channel} — \"{content}\""
    )
