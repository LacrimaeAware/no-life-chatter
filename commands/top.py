import asyncio

from utils import chat_archive
from utils.persona_axes import top, rank, axis_error_message
from utils.persona_traits import pole_map

description = (
    "~top <trait> [n] [burst] — who leans most toward a trait. Add a username "
    "(~top <trait> <user>) to see that user's rank + neighbours instead. "
    "burst = peak moments."
)


def _parse(params):
    user = None
    burst = False
    rest = []
    for p in params:
        low = p.lower()
        if low == "burst":
            burst = True
        elif low.startswith("user="):
            user = p.split("=", 1)[1].lstrip("@") or None
        elif p.startswith("@"):
            user = p.lstrip("@") or None
        else:
            rest.append(p)
    return user, burst, rest


async def handle_top(bot, message, params):
    user, burst, rest = _parse(params or [])
    trait = rest[0].lower() if rest else ""
    if not trait:
        await message.channel.send(
            f"Usage: ~top <trait> [n] [user] — built-ins: {', '.join(sorted(pole_map()))}"
        )
        return
    # bare "~top <trait> <user>": a 2nd non-numeric word is a username, so you
    # don't need the @ or user= (traits are single words, so it's unambiguous).
    if not user and len(rest) > 1 and not rest[1].isdigit():
        user = rest[1].lstrip("@")

    if user:
        info = await asyncio.to_thread(rank, trait, user)
        if not info:
            await message.channel.send(
                f"No '{trait}' read for {chat_archive.display_name(user)} "
                "(not in the roster, or the axis couldn't be built)."
            )
            return
        bits = []
        if info["above"]:
            bits.append(f"↑{chat_archive.display_name(info['above'][0])} {info['above'][1]:+.1f}σ")
        if info["below"]:
            bits.append(f"↓{chat_archive.display_name(info['below'][0])} {info['below'][1]:+.1f}σ")
        ctx = (" · " + " · ".join(bits)) if bits else ""
        await message.channel.send(
            f"📊 {chat_archive.display_name(info['user'])}: #{info['rank']}/{info['total']} most {trait} "
            f"({info['z']:+.1f}σ){ctx}"[:480]
        )
        return

    n = 5
    if len(rest) > 1 and rest[1].isdigit():
        n = max(1, min(int(rest[1]), 10))
    rows, note = await asyncio.to_thread(top, trait, n * 3, burst)
    if rows is None:
        reason = axis_error_message(trait)
        await message.channel.send(
            f"Couldn't build a '{trait}' axis — {reason}. "
            "Queued commands run one at a time now."
        )
        return

    seen = set()
    shown = []
    for author, z in rows:
        canon = chat_archive.normalize_author(author)
        if canon in seen or chat_archive._is_noise_author(canon):
            continue
        seen.add(canon)
        shown.append((canon, z))
        if len(shown) >= n:
            break

    parts = [f"{i}. {chat_archive.display_name(author)} ({z:+.1f}σ)"
             for i, (author, z) in enumerate(shown, 1)]
    mode = " (peak moments)" if burst else ""
    msg = f"most {trait}{mode}: " + " | ".join(parts)
    if note:
        msg += f"  [{note}]"
    await message.channel.send(msg)
