from utils.persona_embeddings import available, neighbors

description = (
    "Who a chatter is semantically most similar to — embedding-space twins "
    "(same topics/energy, even with zero shared catchphrases). The meaning "
    "counterpart of ~like's vocabulary overlap. Scale: ~0.6 is what a person "
    "scores against THEMSELVES re-sampled, 0 = average stranger — so 0.5+ is "
    "alt-tier similarity.\n"
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
    # ~0.58 is what the same person scores against themselves (split-half
    # test) — flag matches in that band, they're alt-tier.
    parts = [f"{a} ({s:.2f}{' ≈same-person!' if s >= 0.55 else ''})"
             for a, s in sims]
    await message.channel.send(f"🧬 {user}'s closest vibes: " + " · ".join(parts))
