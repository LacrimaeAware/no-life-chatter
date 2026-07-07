import asyncio

from utils import emote_explain

description = (
    "Explain learned bot artifacts from their evidence. Currently supports "
    "emote meaning reports.\n"
    "  ~explain emote <emote> [raw]"
)


async def handle_explain(bot, message, params):
    if len(params) < 2 or params[0].lower() not in {"emote", "emotes"}:
        await message.channel.send("Usage: ~explain emote <emote> [raw]")
        return
    raw = any(p.lower() in {"raw", "scores", "vector"} for p in params[2:])
    report = await asyncio.to_thread(emote_explain.analyze, params[1].lstrip("@"))
    await message.channel.send(await emote_explain.chat_response(report, detail=True, raw=raw))
