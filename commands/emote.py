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
    near = emote_meaning.nearest_emotes(token, n=5)
    # clean meaning words come from the emote's OWN 7TV tags; the nearest
    # emotes are the usage-meaning evidence
    return name, info, near


async def handle_emote(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~emote <emote>")
        return
    token = params[0].lstrip("@")
    name, info, near = await asyncio.to_thread(_analyze, token)
    bits = []
    if info:
        if info.get("original") and info["original"] != name:
            bits.append(f"alias of {info['original']}")
        if info.get("tags"):
            bits.append("tags: " + ", ".join(info["tags"][:4]))
    if near:
        bits.append("used like: " + " ".join(a for a, _ in near))
    if not bits:
        await message.channel.send(
            f"'{token}': not in the registry and no usage vector yet "
            "(might be dead/rare, or not actually an emote).")
        return
    await message.channel.send(f"🪙 {name or token} — " + " · ".join(bits))
