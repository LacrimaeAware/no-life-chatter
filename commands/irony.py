import asyncio

from utils import irony

description = (
    "Guess whether ONE typed message reads ironic or sincere (and why). It scores "
    "only the text you type — it cannot see who'd say it or the surrounding chat, "
    "so it's a blind toy, not a verdict. Emotes are resolved first; add context= to "
    "feed it surrounding lines manually.\n"
    "  ~irony <message>   |   ~irony context=women DansGame asldkasd"
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
    tail = "" if context else " — blind 1-msg guess (no speaker/context)"
    await message.channel.send(
        f"{verdict} (sarcasm {zi:+.1f} | content-extremity {zh:+.1f}){tail}"
    )
