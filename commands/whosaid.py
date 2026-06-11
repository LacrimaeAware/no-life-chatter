from utils.chat_archive import recent_authors
from utils.persona_classifier import classify

description = (
    "Guess which chatter who's here is most likely to have said a line. "
    "Ranks only people active in this channel; add 'anyone' for the whole archive.\n"
    "  ~whosaid <sentence>   |   ~whosaid anyone <sentence>"
)


async def handle_whosaid(bot, message, params):
    anyone = bool(params) and params[0].lower() in ("anyone", "*", "everyone")
    if anyone:
        params = params[1:]
    text = " ".join(params).strip()
    if not text:
        await message.channel.send("Usage: ~whosaid <sentence>  (or ~whosaid anyone <sentence>)")
        return

    if anyone:
        ranked, scope = classify(text, top_k=3), "anyone"
    else:
        present = recent_authors(message.channel.name, scan=1500)
        ranked, scope = classify(text, top_k=3, restrict_to=present), "in chat"
        if not ranked:
            ranked, scope = classify(text, top_k=3), "anyone"

    if not ranked:
        await message.channel.send(
            "No classifier trained yet (run scripts/train_classifier.py), "
            "or that line had nothing to go on."
        )
        return
    parts = [f"{author} ({prob:.0%})" for author, prob in ranked]
    await message.channel.send(f"Sounds most like ({scope}) " + " | ".join(parts))
