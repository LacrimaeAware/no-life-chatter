import asyncio

from utils import chat_archive
from utils import comedic_influence as ci

description = (
    "Comedy ranking: how much MORE other people laugh in the ~30s AFTER you talk "
    "than the ~30s before — so a whole chat reacting to the stream cancels out and "
    "only laughter you actually spark counts (laughing at your own joke never "
    "counts). Index is roster-relative: 100 = average chatter, higher = you make "
    "people laugh; 'breadth' = how many DIFFERENT people you've cracked up. Runs on "
    "the conversational chats set in config by default; override with chat=.\n"
    "  ~funny <user>  ·  ~funny top [n]  ·  ~funny bottom [n]  ·  ~funny <user> chat=<channel>"
)


def _parse(params):
    channels = None
    rest = []
    for p in params:
        if p.lower().startswith("chat="):
            vals = [chat_archive.normalize_channel(c)
                    for c in p.split("=", 1)[1].split(",") if c.strip()]
            channels = tuple(vals) or None
        else:
            rest.append(p)
    return channels, rest


async def handle_funny(bot, message, params):
    channels, rest = _parse(params or [])
    chans = channels or ci.DEFAULT_CHANNELS
    if not rest:
        await message.channel.send("Usage: ~funny <user> | ~funny top [n] | ~funny bottom [n]")
        return

    sub = rest[0].lower().lstrip("@")
    if sub in ("top", "bottom"):
        n = 5
        if len(rest) > 1 and rest[1].isdigit():
            n = max(1, min(int(rest[1]), 10))
        rows = await asyncio.to_thread(ci.leaderboard, chans, n, sub == "bottom")
        if not rows:
            await message.channel.send("Not enough data to rank comedy yet.")
            return
        label = "least funny" if sub == "bottom" else "funniest"
        parts = [f"{i}. {a} ({d['index']})" for i, (a, d) in enumerate(rows, 1)]
        await message.channel.send(f"😂 {label} [{'+'.join(chans)}]: " + " · ".join(parts))
        return

    user = chat_archive.normalize_author(sub)
    d = await asyncio.to_thread(ci.user_score, sub, chans)
    if not d:
        await message.channel.send(
            f"Not enough data to score {user} in {'/'.join(chans)} "
            f"(needs {ci.MIN_SETUPS}+ lines with others around, and several "
            f"different people laughing — not just one clique)."
        )
        return
    best = d.get("best_text") or ""
    tail = f" · best landed line: \"{best[:140]}\"" if best else ""
    await message.channel.send(
        f"😂 {user}: comedy index {d['index']} · made {d['breadth']} different "
        f"people laugh · judged on {d['setups']} lines{tail}"
    )
