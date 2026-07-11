import asyncio

from auth import is_super_admin
from utils import persona_axes

description = (
    "~axismerge <canonical> <dup> [flip] — fold a duplicate dynamic axis into "
    "another (super admins). Same pole by default; add 'flip' when <dup> names "
    "the OPPOSITE pole. ~axismerge list shows saved axes. Auto-merge is off on "
    "purpose: embedding cosine can't tell synonyms from distinct concepts, so "
    "dedup is a human call."
)


def _list_axes(limit):
    custom = persona_axes._load_custom()
    rows = []
    for name, d in sorted(custom.items()):
        al = d.get("aliases", [])
        nal = d.get("neg_aliases", [])
        tag = ""
        if al:
            tag += " =" + ",".join(al[:4])
        if nal:
            tag += " ↔" + ",".join(nal[:4])
        rows.append(f"{name}/{d.get('neg_label', '?')}{tag}")
    return rows[:limit], len(custom)


async def handle_axismerge(bot, message, params):
    if not message.author or not is_super_admin(message.author.name):
        return
    if params and params[0].lower() == "list":
        n = int(params[1]) if len(params) > 1 and params[1].isdigit() else 20
        rows, total = await asyncio.to_thread(_list_axes, n)
        await message.channel.send(f"Saved axes ({total}): " + " | ".join(rows))
        return
    if len(params) < 2:
        await message.channel.send(
            "Usage: ~axismerge <canonical> <dup> [flip] — fold <dup> into <canonical> "
            "(flip = <dup> is the opposite pole). ~axismerge list to inspect."
        )
        return
    canonical = params[0].lower().lstrip("@")
    dup = params[1].lower().lstrip("@")
    flip = len(params) > 2 and params[2].lower() in ("flip", "opposite", "opp", "-")
    try:
        summary = await asyncio.to_thread(
            persona_axes.merge_axes, canonical, dup, opposite=flip
        )
    except (KeyError, ValueError) as exc:
        await message.channel.send(f"Can't merge: {exc}")
        return
    await message.channel.send(summary)
