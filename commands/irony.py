import asyncio

from utils import irony

description = (
    "~irony [user=<name>] [chat=here|all] <message> — experimental intent read "
    "from surface wording, community repetition, unusual claims, and confirmed user history."
)


async def handle_irony(bot, message, params):
    context = ""
    author = None
    channel = getattr(message.channel, "name", None)
    rest = []
    for p in params:
        low = p.lower()
        if low.startswith("context="):
            context = p.split("=", 1)[1].strip("()").replace(",", " ")
        elif low.startswith(("user=", "author=")):
            author = p.split("=", 1)[1].strip().lstrip("@") or None
        elif low.startswith("chat="):
            value = p.split("=", 1)[1].strip().lstrip("#").lower()
            channel = None if value in {"", "all", "*"} else (
                getattr(message.channel, "name", None) if value in {"here", "this"} else value
            )
        else:
            rest.append(p)
    text = " ".join(rest).strip()
    if not text:
        await message.channel.send(
            "Usage: ~irony [user=<name>] [chat=here|all] <message>"
        )
        return
    result = await asyncio.to_thread(
        irony.analyze, text, context, author=author, channel=channel
    )
    await message.channel.send(irony.format_analysis(result))
