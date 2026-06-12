import asyncio

from utils.persona_iq import leaderboard, score

description = (
    "A text-register estimate styled like IQ — NOT actual IQ (text can't "
    "measure intelligence; this is calibrated to this chat, average here = "
    "100). Built from top-decile vocabulary rarity, sustained syntax, lexical "
    "diversity, topic breadth and niche depth — deliberately different "
    "machinery from the professor axis.\n"
    "  ~iq <user>   ·   ~iq top [n]   ·   ~iq bottom [n]"
)


def _fmt(a, d):
    parts = []
    for c in ("vocab", "syntax", "breadth", "depth"):
        parts.append(f"{c} {d[c]:+.1f}")
    return f"{a}: {d['iq']} ({' · '.join(parts)})"


async def handle_iq(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~iq <user> | ~iq top | ~iq bottom")
        return
    sub = params[0].lower().lstrip("@")
    n = 5
    if len(params) > 1 and params[1].isdigit():
        n = max(1, min(int(params[1]), 10))
    if sub in ("top", "bottom"):
        rows = await asyncio.to_thread(leaderboard, n, sub == "bottom")
        parts = [f"{i}. {a} ({d['iq']})" for i, (a, d) in enumerate(rows, 1)]
        await message.channel.send(f"🧠 {sub} estimated text-IQ: " + " · ".join(parts))
        return
    d = await asyncio.to_thread(score, sub)
    if not d:
        await message.channel.send(f"Not enough data for {sub} (needs 50+ utterances in the index).")
        return
    await message.channel.send(f"🧠 {_fmt(sub, d)}  [chat-relative estimate, not real IQ]")
