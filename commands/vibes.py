from utils.persona_embeddings import available, neighbors

description = (
    "Who a chatter is semantically most similar to — embedding-space twins "
    "(same topics/energy, even with zero shared catchphrases). The meaning "
    "counterpart of ~like's vocabulary overlap.\n"
    "  ~vibes <user>"
)


async def handle_vibes(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~vibes <user>")
        return
    user = params[0].lstrip("@")
    if not available():
        await message.channel.send(
            "No semantic vectors built yet (run scripts/build_persona_embeddings.py)."
        )
        return
    sims = neighbors(user, n=4)
    if not sims:
        await message.channel.send(f"No semantic vector for {user} (not in the roster yet).")
        return
    parts = [f"{a} ({s:.2f})" for a, s in sims]
    await message.channel.send(f"🧬 {user}'s closest vibes: " + " · ".join(parts))
