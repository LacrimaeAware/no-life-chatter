import asyncio

from utils.persona_traits import AXES
from utils import chat_archive

description = (
    "~traits <user> — their lean on the 5 core axes (wholesome↔menace, "
    "sincere↔ironic, chill↔unhinged, brainrot↔professor, optimist↔doomer), in σ "
    "vs the room. ⚡ = spans both poles."
)

# Flag an axis as performative only when the person BOTH occupies both poles
# (contradiction z above this) AND actually leans there (|σ| above MIN_LEAN) —
# caveating a near-zero lean is noise. See scripts/contradiction.py and
# docs/GROUND_TRUTH.md known limitations.
CONTRA_FLAG_Z = 1.0
MIN_LEAN_FOR_FLAG = 0.5


def _readout(user):
    from utils.persona_axes import axis_scores
    from utils import persona_msg_index as pmi
    canon = chat_archive.normalize_author(user)
    have_contra = pmi.available()
    out = []
    for axis in AXES:
        scores = axis_scores(axis)
        if canon not in scores:
            return None
        contra = pmi.contradiction_scores(axis).get(canon) if have_contra else None
        out.append((axis, float(scores[canon]), contra))
    out.sort(key=lambda t: -abs(t[1]))
    return out


async def handle_traits(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~traits <user>")
        return
    user = chat_archive.normalize_author(params[0].lstrip("@"))
    # first call may embed axis poles/emote names — keep it off the event loop
    traits = await asyncio.to_thread(_readout, user)
    if not traits:
        await message.channel.send(f"No semantic vector for {user} (not in the roster yet).")
        return
    # all five axes, strongest deviation first; the label names the pole they
    # lean toward, σ = standard deviations from the roster average. ⚡ flags an
    # axis where they occupy BOTH poles, so the lean is a mean-pool artifact.
    parts = []
    for axis, z, contra in traits:
        neg, pos = AXES[axis][0], AXES[axis][1]
        label = pos if z >= 0 else neg
        tag = "⚡" if (contra is not None and contra > CONTRA_FLAG_Z
                       and abs(z) >= MIN_LEAN_FOR_FLAG) else ""
        parts.append(f"{label} {abs(z):.1f}σ{tag}")
    await message.channel.send(f"🧪 {user}: " + " · ".join(parts))
