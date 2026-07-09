from utils.persona_classifier import most_like
from commands.markers import _parse_scope, _scope_label
from utils import chat_archive

description = (
    "Who shares a chatter's distinctive voice — scored on signature vocabulary, "
    "shows WHICH markers they share. Defaults to THIS chat's logs; chat=all "
    "compares their full history (best for alt-hunting), year=<YYYY> scopes "
    "to a year.\n"
    "  ~like <user> [chat=all|<channel>] [year=2023]"
)


async def handle_like(bot, message, params):
    params, channel, year = _parse_scope(params, message.channel.name)
    if not params:
        await message.channel.send("Usage: ~like <user> [chat=all|<channel>] [year=2023]")
        return
    user = chat_archive.normalize_author(params[0].lstrip("@"))
    sims = most_like(user, n=4, channel=channel, year=year)
    if not sims:
        await message.channel.send(
            f"Not enough archived messages for {chat_archive.display_name(user)} "
            f"in {_scope_label(channel, year)}.")
        return
    top_author, top_score, shared = sims[0]
    msg = (f"👯 {chat_archive.display_name(user)} ({_scope_label(channel, year)}) sounds most like "
           f"{chat_archive.display_name(top_author)} ({top_score:.2f})")
    if shared:
        msg += f" — both overuse: {', '.join(shared[:4])}"
    rest = " · ".join(f"{chat_archive.display_name(a)} ({s:.2f})" for a, s, _ in sims[1:])
    if rest:
        msg += f" · then {rest}"
    await message.channel.send(msg)
