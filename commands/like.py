from utils.persona_classifier import most_like
from commands.markers import _parse_chat

description = (
    "Who shares a chatter's distinctive voice — scored on signature vocabulary "
    "(log-odds vs the average chatter), and shows WHICH markers they share. "
    "Compares full history by default; chat=<channel> scopes the person's "
    "profile to one chat.\n"
    "  ~like <user> [chat=<channel>]"
)


async def handle_like(bot, message, params):
    # default = full history: alt detection wants everything they've written
    params, channel = _parse_chat(params, None)
    if not params:
        await message.channel.send("Usage: ~like <user> [chat=<channel>]")
        return
    user = params[0].lstrip("@")
    sims = most_like(user, n=4, channel=channel)
    if not sims:
        await message.channel.send(
            f"Not enough archived messages for {user} to build a voice profile."
        )
        return
    top_author, top_score, shared = sims[0]
    scope = f" (in #{channel})" if channel else ""
    msg = f"👯 {user}{scope} sounds most like {top_author} ({top_score:.2f})"
    if shared:
        msg += f" — both overuse: {', '.join(shared[:4])}"
    rest = " · ".join(f"{a} ({s:.2f})" for a, s, _ in sims[1:])
    if rest:
        msg += f" · then {rest}"
    await message.channel.send(msg)
