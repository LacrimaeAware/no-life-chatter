import asyncio

from utils import irony

description = (
    "~irony <message> [context=...] — does a message read ironic or sincere "
    "(sarcasm + content-extremity axes). Emotes resolved first."
)


async def handle_irony(bot, message, params):
    context = ""
    rest = []
    for p in params:
        if p.lower().startswith("context="):
            context = p.split("=", 1)[1].strip("()").replace(",", " ")
        else:
            rest.append(p)
    text = " ".join(rest).strip()
    if not text:
        await message.channel.send("Usage: ~irony <message> [context=...]")
        return
    verdict, zi, zh = await asyncio.to_thread(irony.read, text, context)
    await message.channel.send(
        f"{verdict} (sarcasm {zi:+.1f} | content-extremity {zh:+.1f})"
    )
