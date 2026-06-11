from utils.persona_classifier import signature_words

description = (
    "Show a chatter's most distinctive words/markers — the reverse of ~whosaid. "
    "Works for anyone in the archive.\n"
    "  ~markers <user>"
)


async def handle_markers(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~markers <user>")
        return
    user = params[0].lstrip("@")
    terms = signature_words(user, n=12)
    if not terms:
        await message.channel.send(f"Not enough archived messages for {user}.")
        return
    words = ", ".join(w for w, _ in terms)
    await message.channel.send(f"🔖 {user}'s signature words: {words}")
