import asyncio

from utils import chat_archive
from utils.persona_axes import top, axis_error_message
from utils.persona_traits import pole_map

description = (
    "~bottom <trait> [n] [burst] — who leans LEAST toward a trait (most toward "
    "its opposite). Same axes as ~top."
)


async def handle_bottom(bot, message, params):
    if not params:
        await message.channel.send(
            f"Usage: ~bottom <trait> [n] -- built-ins: {', '.join(sorted(pole_map()))} "
            "(or any word; the trait can be more than one word)"
        )
        return
    burst = False
    n = 5
    words = []
    for p in params:
        low = p.lower()
        if low == "burst":
            burst = True
        elif low.isdigit():
            n = max(1, min(int(low), 10))
        else:
            words.append(low)
    trait = " ".join(words).strip()
    if not trait:
        await message.channel.send("Usage: ~bottom <trait> [n]")
        return
    if len(words) > 3:
        await message.channel.send(
            "That's a lot of words for a trait — keep it to 3 or fewer."
        )
        return
    rows, note = await asyncio.to_thread(top, trait, n * 3, burst, True)
    if rows is None:
        reason = axis_error_message(trait)
        await message.channel.send(
            f"Couldn't build a '{trait}' axis -- {reason}. "
            "Queued commands run one at a time now."
        )
        return

    seen, shown = set(), []
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
    msg = f"least {trait}{mode}: " + " | ".join(parts)
    if note:
        msg += f"  [{note}]"
    await message.channel.send(msg)
