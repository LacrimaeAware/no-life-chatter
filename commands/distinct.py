import asyncio

from utils.persona_axes import most_distinct

description = (
    "Who has the most distinct personality: chatters who deviate furthest from "
    "the room average across all five trait axes, with the traits that define "
    "them.\n"
    "  ~distinct [top|bottom] [n]"
)


async def handle_distinct(bot, message, params):
    reverse = False
    n = 5
    rest = list(params or [])
    if rest and rest[0].lower() in {"top", "most"}:
        rest.pop(0)
    elif rest and rest[0].lower() in {"bottom", "least"}:
        reverse = True
        rest.pop(0)
    if rest and rest[0].isdigit():
        n = max(1, min(int(rest[0]), 10))

    rows = await asyncio.to_thread(most_distinct, n, reverse)
    if not rows:
        await message.channel.send("No semantic vectors built yet.")
        return
    parts = [
        f"{i}. {a} ({total:.1f} sigma: {'/'.join(labels[:2])})"
        for i, (a, total, labels) in enumerate(rows, 1)
    ]
    label = "least" if reverse else "most"
    await message.channel.send(f"{label} distinct personalities: " + " | ".join(parts))
