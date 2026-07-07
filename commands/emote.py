import asyncio

from utils import emote_explain

description = (
    "What the bot thinks an emote means from registry facts, learned usage "
    "neighbors, and cached vector geometry.\n"
    "  ~emote <emote> [raw]"
)


async def handle_emote(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~emote <emote> [raw]")
        return
    token = params[0].lstrip("@")
    raw = any(p.lower() in {"raw", "scores", "vector"} for p in params[1:])
    report = await asyncio.to_thread(emote_explain.analyze, token)
    await message.channel.send(await emote_explain.chat_response(report, detail=False, raw=raw))
