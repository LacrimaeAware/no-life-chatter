import asyncio

from utils.persona_traits import leaderboard, pole_map

description = (
    "Trait leaderboard — who leans hardest toward a pole. Both directions "
    "work: menace/wholesome, ironic/sincere, unhinged/chill, "
    "professor/brainrot, doomer/optimist.\n"
    "  ~top <trait> [n]   e.g. ~top unhinged · ~top wholesome 10"
)


async def handle_top(bot, message, params):
    poles = ", ".join(sorted(pole_map()))
    if not params:
        await message.channel.send(f"Usage: ~top <trait> [n] — traits: {poles}")
        return
    trait = params[0].lower()
    n = 5
    if len(params) > 1 and params[1].isdigit():
        n = max(1, min(int(params[1]), 10))
    rows = await asyncio.to_thread(leaderboard, trait, n)
    if rows is None:
        await message.channel.send(f"Unknown trait '{trait}'. Traits: {poles}")
        return
    parts = [f"{i}. {a} ({z:+.1f}σ)" for i, (a, z) in enumerate(rows, 1)]
    await message.channel.send(f"🏆 most {trait}: " + " · ".join(parts))
