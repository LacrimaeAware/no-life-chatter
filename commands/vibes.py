from utils import chat_archive
from utils.persona_embeddings import available, load, neighbors

description = (
    "Who a chatter is semantically most similar to — embedding-space twins "
    "(same topics/energy, even with zero shared catchphrases). The meaning "
    "counterpart of ~like's vocabulary overlap. The ≈same-person flag is "
    "calibrated against measured self-similarity, not a fixed number.\n"
    "  ~vibes <user>"
)


async def handle_vibes(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~vibes <user>")
        return
    user = chat_archive.normalize_author(params[0].lstrip("@"))
    if not available():
        await message.channel.send(
            "No semantic vectors built yet (run scripts/build_persona_embeddings.py)."
        )
        return
    sims = neighbors(user, n=4)
    if not sims:
        await message.channel.send(f"No semantic vector for {user} (not in the roster yet).")
        return
    # Alt-tier flag: midpoint between the measured same-person ceiling
    # (split-half self-similarity) and the 95th-pct stranger score, stored in
    # the pickle at calibration time so a rebuild can't leave it stale. (A
    # fixed 0.55 from the old noisier vectors flagged normal community
    # similarity as alts.)
    d = load()
    flag_at = (d.get("self_sim_ceiling", 0.82) + d.get("cross_p95", 0.52)) / 2
    parts = [f"{a} ({s:.2f}{' ≈same-person!' if s >= flag_at else ''})"
             for a, s in sims]
    await message.channel.send(f"🧬 {user}'s closest vibes: " + " · ".join(parts))
