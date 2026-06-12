import asyncio

from utils.persona_traits import AXES
from utils import chat_archive

description = (
    "A chatter's personality readout — their semantic vector (prose + "
    "emote-name semantics) projected onto the five core trait axes "
    "(wholesome↔menace, sincere↔ironic, chill↔unhinged, brainrot↔professor, "
    "optimist↔doomer). σ = standard deviations vs the average chatter here. "
    "A fun mirror, not a diagnosis.\n"
    "  ~traits <user>"
)


def _readout(user):
    from utils.persona_axes import axis_scores
    canon = chat_archive.normalize_author(user)
    out = []
    for axis in AXES:
        scores = axis_scores(axis)
        if canon not in scores:
            return None
        out.append((axis, float(scores[canon])))
    out.sort(key=lambda kv: -abs(kv[1]))
    return out


async def handle_traits(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~traits <user>")
        return
    user = params[0].lstrip("@")
    # first call may embed axis poles/emote names — keep it off the event loop
    traits = await asyncio.to_thread(_readout, user)
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
