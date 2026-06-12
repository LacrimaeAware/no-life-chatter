import asyncio

from utils.persona_traits import AXES, traits_for

description = (
    "A chatter's personality readout — their semantic vector projected onto "
    "trait axes (wholesome↔menace, sincere↔ironic, chill↔unhinged, "
    "brainrot↔professor, optimist↔doomer). Scores are vs the average chatter "
    "here (+2 = two standard deviations). A fun mirror, not a diagnosis.\n"
    "  ~traits <user>"
)


async def handle_traits(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~traits <user>")
        return
    user = params[0].lstrip("@")
    # first call embeds the axis pole sentences — keep it off the event loop
    traits = await asyncio.to_thread(traits_for, user)
    if not traits:
        await message.channel.send(f"No semantic vector for {user} (not in the roster yet).")
        return
    # all five axes, strongest deviation first; the label names the pole they
    # lean toward, σ = standard deviations from the roster average
    parts = []
    for axis, z in traits:
        neg, pos = AXES[axis][0], AXES[axis][1]
        label = pos if z >= 0 else neg
        parts.append(f"{label} {abs(z):.1f}σ")
    await message.channel.send(f"🧪 {user}: " + " · ".join(parts))
