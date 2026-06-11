from utils.persona_classifier import profile_for

description = (
    "A chatter's voice profile — favorite words and favorite word-pairs, "
    "scored against the average chatter over their FULL history. The reverse "
    "of ~whosaid; works for anyone in the archive.\n"
    "  ~markers <user>"
)


async def handle_markers(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~markers <user>")
        return
    user = params[0].lstrip("@")
    prof = profile_for(user)
    if not prof:
        await message.channel.send(f"Not enough archived messages for {user}.")
        return
    # profiles are insertion-ordered most-distinctive-first
    words = list(prof.get("words", {}))[:10]
    pairs = list(prof.get("phrases", {}))[:5]
    msg = f"🔖 {user}'s voice — words: {', '.join(words)}"
    if pairs:
        msg += f" · pairs: {' / '.join(pairs)}"
    await message.channel.send(msg)
