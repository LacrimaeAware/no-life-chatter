import asyncio

from utils.persona_iq import leaderboard, score
from utils import chat_archive

description = (
    "~iq <user> | ~iq top|bottom [n] | ~iq how — roster-relative \"text-IQ\": peak "
    "expressed cognition in chat (a toy, not real IQ). 'how' explains each dimension."
)

# Honest derivation notes, one per dimension. "reasoning" is embedding
# similarity to reasoning-SOUNDING sentences — register, not verified logic —
# and users deserve to know that before they read the number as real.
HOW_TEXT = (
    "text-IQ is a roster-relative toy: every dimension is a z-score vs this "
    "roster (~34 chatters), blended and rescaled to 100±15 (clamped 62-158). "
    "reasoning/abstraction = how much your messages RESEMBLE reasoning-/abstract-"
    "sounding example sentences in embedding space (register, not verified logic) · "
    "vocab = word rarity (emotes/usernames excluded) + diversity · syntax = "
    "clause/length shape · breadth = topic spread · depth = niche focus. "
    "Each dimension scores your top ~10% of messages (peaks, not average). "
    "No LLM grades anyone."
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
        await message.channel.send("Usage: ~iq <user> | ~iq top | ~iq bottom | ~iq how")
        return
    sub = params[0].lower().lstrip("@")
    if sub in ("how", "why", "explain", "method"):
        await message.channel.send(HOW_TEXT[:480])
        return
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
