import asyncio
import os

from utils import chat_archive

description = (
    "Interpretability: WHICH of a chatter's real messages most drive their "
    "score on an axis — the receipts behind ~top/~traits. Works for any "
    "built-in or saved axis.\n"
    "  ~why <user> <trait>     e.g. ~why someuser doomer"
)


_axis_cal = {}   # axis -> (mu, sd) of per-message projections, index-wide


def _calibration(axis, av):
    if axis not in _axis_cal:
        import numpy as np
        import random
        d = os.path.join("data", "unsynced", "msg_index")
        files = [f for f in os.listdir(d) if f.endswith(".npz")]
        random.Random(7).shuffle(files)
        projs = []
        for f in files[:10]:
            V = np.load(os.path.join(d, f), allow_pickle=True)["vectors"].astype("float32")
            projs.append(V @ av)
        allp = np.concatenate(projs)
        _axis_cal[axis] = (float(allp.mean()), float(allp.std()) or 1.0)
    return _axis_cal[axis]


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
    av32 = np.asarray(av, dtype="float32")
    proj = sign * (V @ av32)
    mu, sd = _calibration(axis, av32)
    # proj already carries the sign; the calibration mean was measured on the
    # unsigned axis, so flip it to match
    z = (proj - sign * mu) / sd
    top = proj.argsort()[::-1][:2]
    bottom = proj.argsort()[:1]
    pos = [(str(d["texts"][i])[:150], float(z[i])) for i in top]
    neg = [(str(d["texts"][i])[:110], float(z[i])) for i in bottom]
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
    msg = (f"🔍 {user} on '{trait}' ({strength}) — "
           f"most ({pos[0][1]:+.1f}): \"{pos[0][0]}\"")
    if len(pos) > 1 and abs(person_z) >= 0.4:
        msg += f" · also ({pos[1][1]:+.1f}): \"{pos[1][0][:80]}\""
    msg += f" · least ({neg[0][1]:+.1f}): \"{neg[0][0][:70]}\""
    await message.channel.send(msg[:480])
