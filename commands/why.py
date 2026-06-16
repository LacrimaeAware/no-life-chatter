import asyncio
import os

from utils import chat_archive

description = (
    "~why <user> <trait> [words] — the actual messages driving a user's trait "
    "score (the receipts behind ~top/~traits). 'words' = which words carry it. "
    "Near-zero σ means the lean is weak and the examples are noise."
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


def _word_attribution(text, axis, sign):
    """Occlusion: drop each word, re-embed, measure projection loss. The words
    whose removal hurts most are what the axis is actually reacting to."""
    import numpy as np
    from utils.persona_axes import _ortho_builtin, _all_axis_vectors
    from utils.persona_traits import _embed
    ortho = _ortho_builtin()
    av = ortho[axis] if axis in ortho else _all_axis_vectors()[axis][0]
    av = np.asarray(av, dtype="float32")
    words = text.split()[:24]
    variants = [" ".join(words)] + [
        " ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    E = np.asarray(_embed(variants), dtype="float32")
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    base = sign * float(E[0] @ av)
    drops = [(w, base - sign * float(E[i + 1] @ av)) for i, w in enumerate(words)]
    drops.sort(key=lambda kv: -kv[1])
    return [(w, d) for w, d in drops if d > 0][:4]


async def handle_why(bot, message, params):
    if len(params) < 2:
        await message.channel.send("Usage: ~why <user> <trait> [words]")
        return
    user, trait = params[0].lstrip("@"), params[1].lower()
    words_mode = len(params) > 2 and params[2].lower() == "words"
    result, err = await asyncio.to_thread(_explain, user, trait)
    if err:
        await message.channel.send(err)
        return
    pos, neg, person_z = result
    if words_mode:
        from utils.persona_axes import resolve_axis
        axis, sign, _n = resolve_axis(trait)
        attr = await asyncio.to_thread(_word_attribution, pos[0][0], axis, sign)
        parts = " · ".join(f"{w} (+{d:.3f})" for w, d in attr) or "no single word carries it"
        await message.channel.send(
            f"🔬 \"{pos[0][0][:120]}\" reads '{trait}' because of: {parts}")
        return
    msg = (f"🔍 {user} on '{trait}' ({person_z:+.1f}σ overall) — "
           f"most ({pos[0][1]:+.1f}): \"{pos[0][0]}\"")
    if len(pos) > 1 and abs(person_z) >= 0.4:
        msg += f" · also ({pos[1][1]:+.1f}): \"{pos[1][0][:80]}\""
    msg += f" · least ({neg[0][1]:+.1f}): \"{neg[0][0][:70]}\""
    await message.channel.send(msg[:480])
