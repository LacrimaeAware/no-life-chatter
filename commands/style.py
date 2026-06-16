import asyncio

from utils import behavior_profile, chat_archive

description = (
    "~style <user> — how a chatter TYPES (emote rate, caps, verbosity, @s, "
    "profanity, vocab) vs the room. The structural read; unlike ~traits it "
    "ignores topic."
)


def _readout(user):
    rows = behavior_profile.profile(user)
    out = []
    for _feat, _val, z, label in rows:
        if label is None or abs(z) < 0.6:   # skip weak / unlabelled directions
            continue
        out.append((label, z))
    return out[:5]


async def handle_style(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~style <user>")
        return
    if not behavior_profile.available():
        await message.channel.send("No message index yet — ~style needs the per-message data.")
        return
    user = chat_archive.normalize_author(params[0].lstrip("@"))
    rows = await asyncio.to_thread(_readout, user)
    if not rows:
        await message.channel.send(f"No behavioral profile for {user} (not in the roster, or nothing stands out).")
        return
    parts = [f"{label} {abs(z):.1f}σ" for label, z in rows]
    await message.channel.send(f"⌨️ {user} types like: " + " · ".join(parts))
