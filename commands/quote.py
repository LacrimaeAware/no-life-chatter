from utils.chat_archive import random_quote

description = (
    "Random real quote from the chat archive.\n"
    "  ~quote <user>"
)


async def handle_quote(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~quote <user>")
        return
    row = random_quote(params[0])
    if not row:
        await message.channel.send(f"Nothing archived for {params[0]} yet.")
        return
    sent_at, channel, content = row
    if len(content) > 300:
        content = content[:299] + "…"
    await message.channel.send(f"\"{content}\" — {params[0]}, {sent_at[:10]}")
