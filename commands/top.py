import asyncio

from utils.persona_axes import top
from utils.persona_traits import pole_map

description = (
    "Trait leaderboard — who leans hardest toward ANY trait. Built-in axes "
    "(menace/wholesome, ironic/sincere, unhinged/chill, professor/brainrot, "
    "doomer/optimist) answer instantly; any other word gets a new axis built "
    "on the fly by the local model, saved, and merged with near-duplicate "
    "axes. Scores blend prose and emote-name semantics.\n"
    "  ~top <trait> [n]   e.g. ~top unhinged · ~top wholesome 10 · ~top <anything>"
)


async def handle_top(bot, message, params):
    if not params:
        await message.channel.send(
            f"Usage: ~top <trait> [n] — built-ins: {', '.join(sorted(pole_map()))} "
            "(or any word — I'll build the axis)")
        return
    trait = params[0].lower()
    n = 5
    if len(params) > 1 and params[1].isdigit():
        n = max(1, min(int(params[1]), 10))
    rows, note = await asyncio.to_thread(top, trait, n)
    if rows is None:
        await message.channel.send(f"Couldn't build an axis for '{trait}' — try another word.")
        return
    parts = [f"{i}. {a} ({z:+.1f}σ)" for i, (a, z) in enumerate(rows, 1)]
    msg = f"🏆 most {trait}: " + " · ".join(parts)
    if note:
        msg += f"  [{note}]"
    await message.channel.send(msg)
