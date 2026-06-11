from utils.persona_classifier import most_like

description = (
    "Who writes most like a chatter — stylistic twins (likely alts/alts of each "
    "other), from the classifier's learned style vectors. Trained authors only.\n"
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
            f"{user} isn't in the trained classifier, so there's no style vector "
            "to compare. Retrain including them first."
        )
        return
    parts = [f"{a} ({s:.2f})" for a, s in sims]
    await message.channel.send(f"👯 writes most like {user}: " + " · ".join(parts))
