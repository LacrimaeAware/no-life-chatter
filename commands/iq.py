import asyncio

from utils.persona_iq import leaderboard, score
from utils import chat_archive

description = (
    "~iq <user> | ~iq top|bottom [n] — roster-relative \"text-IQ\": peak "
    "expressed cognition in chat (a toy, not real IQ)."
)


def _fmt(a, d):
    parts = []
    for c in ("reasoning", "abstraction", "vocab", "syntax", "breadth", "depth"):
        if c in d:
            parts.append(f"{c} {d[c]:+.1f}")
    tail = f"pct {d.get('percentile', '?')} | conf {d.get('confidence', '?')}"
    return f"{a}: {d['iq']} ({' | '.join(parts)}) [{tail}]"


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
        parts = [
            f"{i}. {chat_archive.display_name(a)} ({d['iq']}, {d.get('confidence', '?')})"
            for i, (a, d) in enumerate(rows, 1)
        ]
        await message.channel.send(f"{sub} text-IQ: " + " | ".join(parts))
        return
    display_user = chat_archive.display_name(sub)
    d = await asyncio.to_thread(score, sub)
    if not d:
        await message.channel.send(
            f"Not enough data for {display_user} (needs enough merged utterances in the archive)."
        )
        return
    await message.channel.send(_fmt(display_user, d))
