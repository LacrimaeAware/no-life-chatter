from utils.persona_classifier import classify

description = (
    "Guess which archived chatter is most likely to have said a line (works on "
    "novel sentences, not just real quotes).\n"
    "  ~whosaid <sentence>"
)


async def handle_whosaid(bot, message, params):
    text = " ".join(params).strip()
    if not text:
        await message.channel.send("Usage: ~whosaid <sentence>")
        return
    ranked = classify(text, top_k=3)
    if not ranked:
        await message.channel.send(
            "No classifier trained yet (run scripts/train_classifier.py), "
            "or that line had nothing to go on."
        )
        return
    parts = [f"{author} ({prob:.0%})" for author, prob in ranked]
    await message.channel.send("🎯 sounds most like " + " · ".join(parts))
