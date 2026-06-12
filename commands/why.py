import asyncio
import os

from utils import chat_archive

description = (
    "Interpretability: WHICH of a chatter's real messages most drive their "
    "score on an axis — the receipts behind ~top/~traits. Works for any "
    "built-in or saved axis.\n"
    "  ~why <user> <trait>     e.g. ~why someuser doomer"
)


def _explain(user, trait):
    import numpy as np
    from utils.persona_axes import _ortho_builtin, _all_axis_vectors, resolve_axis
    resolved = resolve_axis(trait)
    if not resolved:
        return None, f"no axis for '{trait}'"
    axis, sign, _note = resolved
    ortho = _ortho_builtin()
    av = ortho[axis] if axis in ortho else _all_axis_vectors()[axis][0]
    canon = chat_archive.normalize_author(user)
    path = os.path.join("data", "unsynced", "msg_index", f"{canon}.npz")
    if not os.path.exists(path):
        return None, f"no message index for {user}"
    d = np.load(path, allow_pickle=True)
    V = d["vectors"].astype("float32")
    proj = sign * (V @ np.asarray(av, dtype="float32"))
    top = proj.argsort()[::-1][:2]
    bottom = proj.argsort()[:1]
    pos = [str(d["texts"][i])[:160] for i in top]
    neg = [str(d["texts"][i])[:120] for i in bottom]
    return (pos, neg), None


async def handle_why(bot, message, params):
    if len(params) < 2:
        await message.channel.send("Usage: ~why <user> <trait>")
        return
    user, trait = params[0].lstrip("@"), params[1].lower()
    result, err = await asyncio.to_thread(_explain, user, trait)
    if err:
        await message.channel.send(err)
        return
    pos, neg = result
    msg = f"🔍 why {user} scores '{trait}' — most: \"{pos[0]}\""
    if len(pos) > 1:
        msg += f" · also: \"{pos[1][:100]}\""
    msg += f" · least: \"{neg[0][:80]}\""
    await message.channel.send(msg[:480])
