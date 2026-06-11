from utils.persona_classifier import most_like

description = (
    "Who writes most like a chatter, using the classifier's style vectors. "
    "Trained authors only.\n"
    "  ~like <user>"
)


async def handle_like(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~like <user>")
        return
    user = params[0].lstrip("@")
    sims = most_like(user, n=5)
    if not sims:
        await message.channel.send(
            f"{user} is not in the trained classifier, or the classifier has no style vectors yet."
        )
        return
    parts = [f"{a} ({s:.2f})" for a, s in sims]
    await message.channel.send(f"Writes most like {user}: " + " | ".join(parts))
