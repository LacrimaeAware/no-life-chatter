from utils.persona_classifier import most_like

description = (
    "Who shares a chatter's distinctive voice — scored on signature vocabulary "
    "(log-odds vs the average chatter), and shows WHICH markers they share.\n"
    "  ~like <user>"
)


async def handle_like(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~like <user>")
        return
    user = params[0].lstrip("@")
    sims = most_like(user, n=4)
    if not sims:
        await message.channel.send(
            f"Not enough archived messages for {user} to build a voice profile."
        )
        return
    top_author, top_score, shared = sims[0]
    msg = f"👯 {user} sounds most like {top_author} ({top_score:.2f})"
    if shared:
        msg += f" — both overuse: {', '.join(shared[:4])}"
    rest = " · ".join(f"{a} ({s:.2f})" for a, s, _ in sims[1:])
    if rest:
        msg += f" · then {rest}"
    await message.channel.send(msg)
