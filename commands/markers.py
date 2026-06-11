from utils.persona_classifier import profile_for

description = (
    "A chatter's voice profile — favorite words and favorite word-pairs vs the "
    "average chatter. Defaults to THIS chat's logs only; chat=all uses their "
    "full history, chat=<channel> scopes to another chat.\n"
    "  ~markers <user> [chat=all | chat=<channel>]"
)


def _parse_chat(params, default):
    """(remaining params, channel-or-None). chat=all -> None (full history)."""
    rest, channel = [], default
    for p in params:
        if p.lower().startswith("chat="):
            v = p.split("=", 1)[1].strip().lstrip("#")
            channel = None if v.lower() in ("all", "*", "") else v
        else:
            rest.append(p)
    return rest, channel


async def handle_markers(bot, message, params):
    params, channel = _parse_chat(params, message.channel.name)
    if not params:
        await message.channel.send("Usage: ~markers <user> [chat=all | chat=<channel>]")
        return
    user = params[0].lstrip("@")
    prof = profile_for(user, channel=channel)
    if not prof:
        where = f"in #{channel}" if channel else "anywhere"
        await message.channel.send(f"Not enough archived messages for {user} {where}.")
        return
    # profiles are insertion-ordered most-distinctive-first
    words = list(prof.get("words", {}))[:10]
    pairs = list(prof.get("phrases", {}))[:5]
    scope = f"#{channel}" if channel else "all chats"
    msg = f"🔖 {user}'s voice ({scope}) — words: {', '.join(words)}"
    if pairs:
        msg += f" · pairs: {' / '.join(pairs)}"
    await message.channel.send(msg)
