import asyncio

from utils.persona_axes import most_distinct

description = (
    "Who has the most distinct personality — the chatters who deviate furthest "
    "from the room average across all five trait axes (summed standard "
    "deviations), with the traits that define them.\n"
    "  ~distinct [n]"
)


async def handle_distinct(bot, message, params):
    n = 5
    if params and params[0].isdigit():
        n = max(1, min(int(params[0]), 10))
    rows = await asyncio.to_thread(most_distinct, n)
    if not rows:
        await message.channel.send("No semantic vectors built yet.")
        return
    parts = [f"{i}. {a} ({total:.1f}σ: {'/'.join(labels[:2])})"
             for i, (a, total, labels) in enumerate(rows, 1)]
    await message.channel.send("🦄 most distinct personalities: " + " · ".join(parts))
