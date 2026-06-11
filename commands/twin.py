from utils import persona_embeddings
from utils.persona_classifier import most_like

description = (
    "A chatter's overall twin — vocabulary + emotes + meaning blended into one "
    "verdict, with the per-channel breakdown. (~like = words/emotes only, "
    "~vibes = meaning only; this is both.)\n"
    "  ~twin <user>"
)


def _zscores(d):
    if not d:
        return {}
    vals = list(d.values())
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
    return {k: (v - mu) / sd for k, v in d.items()}


async def handle_twin(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~twin <user>")
        return
    user = params[0].lstrip("@")
    lex = {a: s for a, s, _ in most_like(user, n=200)}
    sem = persona_embeddings.similarities(user)
    if not lex and not sem:
        await message.channel.send(f"Not enough archived data for {user}.")
        return
    # z-score each channel so neither scale dominates, blend over the overlap
    lz, sz = _zscores(lex), _zscores(sem)
    pool = set(lz) & set(sz) or set(lz) or set(sz)
    blend = {a: 0.5 * lz.get(a, 0.0) + 0.5 * sz.get(a, 0.0) for a in pool}
    ranked = sorted(blend.items(), key=lambda kv: -kv[1])[:3]
    top, _ = ranked[0]
    detail = f"voice {lex.get(top, 0):.2f} · meaning {sem.get(top, 0):.2f}"
    rest = " · ".join(a for a, _ in ranked[1:])
    msg = f"🪞 {user}'s twin: {top} ({detail})"
    if rest:
        msg += f" · then {rest}"
    await message.channel.send(msg)
