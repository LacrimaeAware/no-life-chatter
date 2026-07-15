import asyncio
import os
import re

from utils import chat_archive
from utils import emote_explain

description = (
    "~why <user> <trait> [words] - actual messages driving a user's trait score. "
    "Shows the resolved axis basis when a trait maps to a saved axis. "
    "~why emote <emote> [raw] explains learned emote meaning from vector evidence."
)


_axis_cal = {}   # axis -> (mu, sd) of per-message projections, index-wide


def _squash_repeated_text(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    words = text.split()
    if len(words) >= 4 and len(words) % 2 == 0:
        half = len(words) // 2
        if words[:half] == words[half:]:
            return " ".join(words[:half])
    return text


def _clip(text, limit):
    text = _squash_repeated_text(text)
    return text if len(text) <= limit else text[:max(0, limit - 1)].rstrip() + "..."


def _axis_basis_note(trait, axis, sign, pos_label, neg_label):
    wanted = pos_label if sign > 0 else neg_label
    other = neg_label if sign > 0 else pos_label
    if trait == wanted and trait == axis:
        return f"basis {wanted} vs {other}"
    return f"basis {wanted} vs {other} (axis {axis})"


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
    from utils.persona_axes import (axis_labels, axis_scores, resolve_axis,
                                    scoring_axis_vector)
    resolved = resolve_axis(trait)
    if not resolved:
        return None, f"no axis for '{trait}'"
    axis, sign, _note = resolved
    person_z = sign * axis_scores(axis).get(
        chat_archive.normalize_author(user), 0.0)
    pos_label, neg_label = axis_labels(axis)
    av = scoring_axis_vector(axis)
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
    pos = [(_clip(d["texts"][i], 150), float(z[i])) for i in top]
    neg = [(_clip(d["texts"][i], 110), float(z[i])) for i in bottom]
    basis = _axis_basis_note(trait, axis, sign, pos_label, neg_label)
    return (pos, neg, person_z, axis, sign, basis), None


def _word_attribution(text, axis, sign):
    """Occlusion: drop each word, re-embed, measure projection loss."""
    import numpy as np
    from utils.persona_axes import scoring_axis_vector
    from utils.persona_traits import _embed
    av = scoring_axis_vector(axis)
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
    if params and params[0].lower() in {"emote", "emotes"}:
        if len(params) < 2:
            await message.channel.send("Usage: ~why emote <emote> [raw]")
            return
        raw = any(p.lower() in {"raw", "scores", "vector"} for p in params[2:])
        report = await asyncio.to_thread(emote_explain.analyze, params[1].lstrip("@"))
        await message.channel.send(await emote_explain.chat_response(report, detail=True, raw=raw))
        return
    if len(params) < 2:
        await message.channel.send("Usage: ~why <user> <trait> [words] OR ~why emote <emote> [raw]")
        return
    user, trait = params[0].lstrip("@"), params[1].lower()
    words_mode = len(params) > 2 and params[2].lower() == "words"
    result, err = await asyncio.to_thread(_explain, user, trait)
    if err:
        await message.channel.send(err)
        return
    pos, neg, person_z, axis, sign, basis = result
    if words_mode:
        attr = await asyncio.to_thread(_word_attribution, pos[0][0], axis, sign)
        parts = " | ".join(f"{w} (+{d:.3f})" for w, d in attr) or "no single word carries it"
        await message.channel.send(
            f'"{_clip(pos[0][0], 120)}" reads {trait!r} because of: {parts}')
        return
    msg = (f"{user} on '{trait}' ({person_z:+.1f} sigma; {basis}) - "
           f"most ({pos[0][1]:+.1f}): \"{pos[0][0]}\"")
    if len(pos) > 1 and abs(person_z) >= 0.4:
        msg += f" | also ({pos[1][1]:+.1f}): \"{_clip(pos[1][0], 80)}\""
    msg += f" | least ({neg[0][1]:+.1f}): \"{_clip(neg[0][0], 70)}\""
    await message.channel.send(msg[:480])
