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
    from utils.persona_axes import (_ortho_builtin, _all_axis_vectors,
                                    axis_scores, resolve_axis)
    resolved = resolve_axis(trait)
    if not resolved:
        return None, f"no axis for '{trait}'"
    axis, sign, _note = resolved
    person_z = sign * axis_scores(axis).get(
        chat_archive.normalize_author(user), 0.0)
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
    return (pos, neg, person_z), None


async def handle_why(bot, message, params):
    if len(params) < 2:
        await message.channel.send("Usage: ~why <user> <trait>")
        return
    user, trait = params[0].lstrip("@"), params[1].lower()
    result, err = await asyncio.to_thread(_explain, user, trait)
    if err:
        await message.channel.send(err)
        return
    pos, neg, person_z = result
    # honesty about signal strength: a weak overall score means these
    # 'most/least' examples are basically noise — say so
    if abs(person_z) >= 1.0:
        strength = f"{person_z:+.1f}σ overall"
    elif abs(person_z) >= 0.4:
        strength = f"{person_z:+.1f}σ overall — weak, examples are shaky"
    else:
        strength = f"{person_z:+.1f}σ overall — basically uncorrelated, examples below are NOISE"
    msg = f"🔍 {user} on '{trait}' ({strength}) — most: \"{pos[0]}\""
    if len(pos) > 1 and abs(person_z) >= 0.4:
        msg += f" · also: \"{pos[1][:90]}\""
    msg += f" · least: \"{neg[0][:80]}\""
    await message.channel.send(msg[:480])
