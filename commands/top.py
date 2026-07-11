import asyncio

from utils import chat_archive
from utils.persona_axes import top, rank, axis_error_message
from utils.persona_traits import pole_map

description = (
    "~top <trait> [n] [burst] — who leans most toward a trait (a leaderboard). "
    "The trait can be more than one word (~top anal lover). For ONE person's "
    "rank + neighbours use an explicit @user or user=<name> (~top intelligent "
    "@bob). burst = peak moments."
)

MAX_TRAIT_WORDS = 3


def _parse(params):
    """(user, burst, n, trait, nwords). A user is taken ONLY from @name or
    user=name — bare words are the (possibly multi-word) trait, so ~top never
    silently reads a trait word as a username."""
    user = None
    burst = False
    n = 5
    words = []
    for p in params:
        low = p.lower()
        if low == "burst":
            burst = True
        elif low.startswith("user="):
            user = p.split("=", 1)[1].lstrip("@") or None
        elif p.startswith("@"):
            user = p.lstrip("@") or None
        elif low.isdigit():
            n = max(1, min(int(low), 10))
        else:
            words.append(low)
    return user, burst, n, " ".join(words).strip(), len(words)


async def handle_top(bot, message, params):
    user, burst, n, trait, nwords = _parse(params or [])
    if not trait:
        await message.channel.send(
            f"Usage: ~top <trait> [n] — built-ins: {', '.join(sorted(pole_map()))}. "
            "For one person's rank: ~top <trait> @user"
        )
        return
    if nwords > MAX_TRAIT_WORDS:
        await message.channel.send(
            f"That's a lot of words for a trait — keep it to {MAX_TRAIT_WORDS} or fewer. "
            "For a single person's rank use ~top <trait> @user."
        )
        return

    if user:
        info = await asyncio.to_thread(rank, trait, user)
        if info.get("error") == "axis":
            await message.channel.send(
                f"Couldn't build a '{trait}' axis — {axis_error_message(trait)}."
            )
            return
        if info.get("error") == "roster":
            await message.channel.send(
                f"{chat_archive.display_name(user)} isn't in the ranked roster yet "
                "(needs more archived messages to score)."
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

    rows, note = await asyncio.to_thread(top, trait, n * 3, burst)
    if rows is None:
        await message.channel.send(
            f"Couldn't build a '{trait}' axis — {axis_error_message(trait)}."
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
