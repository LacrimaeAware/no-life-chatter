import asyncio

from utils import emote_meaning

description = (
    "What the bot thinks an emote means — from registry facts (alias, original "
    "name, 7TV tags) and learned usage (which plain words and other emotes its "
    "messages sit closest to).\n"
    "  ~emote <emote>"
)


def _analyze(token):
    name, info = emote_meaning.lookup(token)
    words = emote_meaning.meaning_words(token, n=3)
    near = emote_meaning.nearest_emotes(token, n=4)
    return name, info, words, near


async def handle_emote(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~emote <emote>")
        return
    token = params[0].lstrip("@")
    name, info, words, near = await asyncio.to_thread(_analyze, token)
    bits = []
    if info:
        if info.get("original") and info["original"] != name:
            bits.append(f"alias of {info['original']}")
        if info.get("tags"):
            bits.append("tags: " + ", ".join(info["tags"][:4]))
    if words:
        bits.append("usage reads as: " + ", ".join(w.split()[0] for w, _ in words))
    if near:
        bits.append("like emotes: " + " ".join(a for a, _ in near))
    if not bits:
        await message.channel.send(
            f"'{token}': not in the registry and no usage vector yet "
            "(might be dead/rare, or not actually an emote).")
        return
    await message.channel.send(f"🪙 {name or token} — " + " · ".join(bits))
