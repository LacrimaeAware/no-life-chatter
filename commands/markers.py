from utils.persona_classifier import profile_for
from utils import chat_archive

description = (
    "A chatter's voice profile — favorite words and favorite word-pairs vs the "
    "average chatter. Defaults to THIS chat's logs; chat=all = full history, "
    "chat=<channel> = another chat, year=<YYYY> = one year only.\n"
    "  ~markers <user> [chat=all|<channel>] [year=2023]"
)


def _parse_scope(params, default_channel):
    """(remaining params, channel-or-None, year-or-None).
    chat=all -> channel None (full history)."""
    rest, channel, year = [], default_channel, None
    for p in params:
        low = p.lower()
        if low.startswith("chat="):
            v = low.split("=", 1)[1].strip().lstrip("#")
            channel = None if v in ("all", "*", "") else v
        elif low.startswith("year=") and low.split("=", 1)[1].isdigit():
            year = int(low.split("=", 1)[1])
        else:
            rest.append(p)
    return rest, channel, year


def _scope_label(channel, year):
    parts = [f"#{channel}" if channel else "all chats"]
    if year:
        parts.append(str(year))
    return " ".join(parts)


async def handle_markers(bot, message, params):
    params, channel, year = _parse_scope(params, message.channel.name)
    if not params:
        await message.channel.send("Usage: ~markers <user> [chat=all|<channel>] [year=2023]")
        return
    user = chat_archive.normalize_author(params[0].lstrip("@"))
    prof = profile_for(user, channel=channel, year=year)
    if not prof:
        await message.channel.send(
            f"Not enough archived messages for {user} in {_scope_label(channel, year)}.")
        return
    # profiles are insertion-ordered most-distinctive-first
    words = list(prof.get("words", {}))[:8]
    emotes = list(prof.get("emotes", {}))[:5]
    pairs = list(prof.get("phrases", {}))[:4]
    msg = f"🔖 {user}'s voice ({_scope_label(channel, year)}) — words: {', '.join(words)}"
    if emotes:
        msg += f" · emotes: {' '.join(emotes)}"
    if pairs:
        msg += f" · pairs: {' / '.join(pairs)}"
    await message.channel.send(msg)
