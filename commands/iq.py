import asyncio

from utils.persona_iq import leaderboard, score
from utils import chat_archive

description = (
    "~iq <user> | ~iq top|bottom [n] | ~iq how [dim] — roster-relative \"text-IQ\": peak "
    "expressed cognition in chat (a toy, not real IQ). 'how' shows per-dimension derivation tags."
)

# Compact provenance, not meta-prose: a tag says HOW each number is derived.
# emb = embedding resemblance to hand-written example sentences (register,
# not verified logic — hence ε). count = transparent counting. All dims are
# roster-relative z-scores over peak (top ~10%) messages. No LLM grading.
DIM_HOW = {
    "reasoning": "reasoning [emb ε]: resemblance of peak messages to reasoning-shaped "
                 "sentences — register, not verified logic",
    "abstraction": "abstraction [emb ε]: resemblance to abstract/technical-register sentences",
    "vocab": "vocab [count]: word rarity (emotes, usernames, non-English excluded) "
             "+ lexical diversity",
    "syntax": "syntax [count]: clause density × length of peak messages",
    "breadth": "breadth [emb]: topic spread across embedding clusters",
    "depth": "depth [emb]: how far your topics sit from the roster average (niche focus)",
}
HOW_TEXT = (
    "text-IQ: roster-relative toy, z vs ~34 chatters → 100±15. "
    "reasoning ε · abstraction ε · breadth · depth = [emb] | vocab · syntax = [count]. "
    "ε = register-read, not verified logic. peaks (top ~10% msgs), no LLM grading. "
    "~iq how <dim> for one dimension."
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
        dim = params[1].lower() if len(params) > 1 else ""
        await message.channel.send(DIM_HOW.get(dim, HOW_TEXT)[:480])
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
