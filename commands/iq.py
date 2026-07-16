import asyncio

from utils.persona_iq import cache_info, cache_problem, leaderboard, score
from utils import chat_archive

description = (
    "~iq <user> | ~iq top|bottom [n] | ~iq how [dim] | ~iq why <user> [dim] "
    "— roster-relative peak expressed cognition. 'why' shows stored top-tail examples."
)

# Compact provenance, not meta-prose: a tag says HOW each number is derived.
# emb = embedding resemblance to hand-written example sentences (register,
# not verified logic, hence ε). mix combines embeddings and direct counts.
# The optional offline local judge checks selected receipts for actual reasoning
# moves; it never runs from the live command. All dimensions are roster-relative
# z-scores over peak (top ~10%) messages.
DIM_HOW = {
    "reasoning": "reasoning [mix]: 35% reasoning-register embeddings ε, 40% direct "
                 "clause/reason markers, 25% structured questions; an offline local "
                 "judge supplies 45% of the final dimension when present",
    "abstraction": "abstraction [mix]: 60% abstract/technical-register embeddings ε; "
                   "an offline local judge supplies 40% when present",
    "vocab": "vocab [count]: word rarity (emotes, usernames, non-English excluded) "
             "+ lexical diversity",
    "syntax": "syntax [count]: capped length plus multi-clause structure in peak messages",
    "breadth": "breadth [emb]: topic spread across embedding clusters",
    "depth": "depth [emb]: how far your topics sit from the roster average (niche focus)",
}
HOW_TEXT = (
    "text-IQ: roster-relative text-cognition estimate, current-roster z → 100±15. "
    "reasoning = [mix] | abstraction ε · breadth · depth = [emb] | vocab · syntax = [count]. "
    "ε = register-read, not verified logic. fixed evidence budget; peaks use top ~10%. "
    "A rebuild can blend an offline local judge over selected receipts. "
    "~iq how <dim> for one dimension."
)


def _fmt(a, d):
    parts = []
    for c in ("reasoning", "abstraction", "vocab", "syntax", "breadth", "depth"):
        if c in d:
            parts.append(f"{c} {d[c]:+.1f}")
    tail = f"pct {d.get('percentile', '?')} | conf {d.get('confidence', '?')}"
    return f"{a}: {d['iq']} ({' | '.join(parts)}) [{tail}]"


def _short(text, limit=145):
    text = " ".join((text or "").split()).replace("|", "/")
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _missing_text(display: str) -> str:
    problem = cache_problem()
    if problem:
        return f"IQ cache needs a maintenance rebuild ({problem})."
    return f"Not enough data for {display}."


def _judge_suffix() -> str:
    meta = cache_info()
    judged = int(meta.get("judge_authors") or 0)
    total = int(meta.get("authors") or 0)
    if judged:
        return f" | local judge: {judged}/{total or '?'}"
    note = str(meta.get("llm_judge", ""))
    if note.startswith("judged ") and not note.startswith("judged 0/"):
        return " | local judge: yes"
    return " | local judge: no"


def _fmt_receipts(author, data, dimension=""):
    receipts = data.get("receipts") or {}
    dims = ("reasoning", "abstraction", "vocab", "syntax", "breadth", "depth")
    if dimension:
        if dimension not in dims:
            return f"Unknown dimension. Use: {', '.join(dims)}"
        rows = receipts.get(dimension) or []
        if not rows:
            return f"{author}: no stored {dimension} examples; rebuild the IQ artifact."
        examples = " | ".join(
            f"{row.get('feature', dimension)}: \"{_short(row.get('text', ''))}\""
            for row in rows[:2]
        )
        return f"{author} {dimension} {data.get(dimension, 0):+.1f}: {examples}"[:480]

    available = [dim for dim in dims if receipts.get(dim)]
    if not available:
        return f"{author}: no stored examples; rebuild the IQ artifact."
    strongest = max(available, key=lambda dim: data.get(dim, 0.0))
    weakest = min(available, key=lambda dim: data.get(dim, 0.0))
    selected = [strongest] + ([weakest] if weakest != strongest else [])
    parts = []
    for dim in selected:
        row = receipts[dim][0]
        parts.append(
            f"{dim} {data.get(dim, 0):+.1f} [{row.get('feature', dim)}] "
            f"\"{_short(row.get('text', ''), 120)}\""
        )
    return f"{author} IQ {data['iq']}: " + " | ".join(parts)


async def handle_iq(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~iq <user> | ~iq top | ~iq bottom | ~iq how")
        return
    sub = params[0].lower().lstrip("@")
    if sub in ("how", "explain", "method"):
        dim = params[1].lower() if len(params) > 1 else ""
        await message.channel.send((DIM_HOW.get(dim, HOW_TEXT) + _judge_suffix())[:480])
        return
    if sub in ("why", "examples", "receipts"):
        if len(params) < 2:
            await message.channel.send("Usage: ~iq why <user> [dimension]")
            return
        # Preserve the old `~iq why reasoning` method shortcut.
        if len(params) == 2 and params[1].lower() in DIM_HOW:
            await message.channel.send(
                (DIM_HOW[params[1].lower()] + _judge_suffix())[:480]
            )
            return
        target = params[1].lower().lstrip("@")
        dim = params[2].lower() if len(params) > 2 else ""
        data = await asyncio.to_thread(score, target)
        display = chat_archive.display_name(target)
        if not data:
            await message.channel.send(_missing_text(display))
            return
        await message.channel.send(_fmt_receipts(display, data, dim)[:480])
        return
    n = 5
    if len(params) > 1 and params[1].isdigit():
        n = max(1, min(int(params[1]), 10))
    if sub in ("top", "bottom"):
        rows = await asyncio.to_thread(leaderboard, n, sub == "bottom")
        if not rows and cache_problem():
            await message.channel.send(_missing_text("that user"))
            return
        parts = [
            f"{i}. {chat_archive.display_name(a)} ({d['iq']}, {d.get('confidence', '?')})"
            for i, (a, d) in enumerate(rows, 1)
        ]
        await message.channel.send(f"{sub} text-IQ: " + " | ".join(parts))
        return
    if len(params) > 1 and params[1].lower() in ("why", "examples", "receipts"):
        dim = params[2].lower() if len(params) > 2 else ""
        data = await asyncio.to_thread(score, sub)
        display = chat_archive.display_name(sub)
        if not data:
            await message.channel.send(_missing_text(display))
            return
        await message.channel.send(_fmt_receipts(display, data, dim)[:480])
        return
    display_user = chat_archive.display_name(sub)
    d = await asyncio.to_thread(score, sub)
    if not d:
        await message.channel.send(_missing_text(display_user))
        return
    await message.channel.send(_fmt(display_user, d))
