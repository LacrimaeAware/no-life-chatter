import asyncio

from utils import chat_archive
from utils.persona_axes import top
from utils.persona_traits import pole_map

description = (
    "Trait leaderboard -- who leans hardest toward ANY trait. Built-in axes "
    "(menace/wholesome, ironic/sincere, unhinged/chill, professor/brainrot, "
    "doomer/optimist) answer instantly; any other word gets a new axis built "
    "on the fly by the local model, saved, and merged with near-duplicate "
    "axes. Scores blend prose and emote-name semantics.\n"
    "  ~top <trait> [n] [burst] -- burst ranks peak moments, not averages"
)


async def handle_top(bot, message, params):
    if not params:
        await message.channel.send(
            f"Usage: ~top <trait> [n] -- built-ins: {', '.join(sorted(pole_map()))} "
            "(or any word -- I'll build the axis)"
        )
        return
    args = [p.lower() for p in params]
    burst = "burst" in args
    args = [a for a in args if a != "burst"]
    trait = args[0] if args else ""
    n = 5
    if len(args) > 1 and args[1].isdigit():
        n = max(1, min(int(args[1]), 10))
    rows, note = await asyncio.to_thread(top, trait, n * 3, burst)
    if rows is None:
        await message.channel.send(
            f"Couldn't build a '{trait}' axis this time -- the local model flaked "
            "(it happens under load). Trying again usually works."
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

    parts = [f"{i}. {author} ({z:+.1f}σ)" for i, (author, z) in enumerate(shown, 1)]
    mode = " (peak moments)" if burst else ""
    msg = f"most {trait}{mode}: " + " | ".join(parts)
    if note:
        msg += f"  [{note}]"
    await message.channel.send(msg)
