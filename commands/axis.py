description = (
    "Inspect a trait/custom axis and show nearby axes.\n"
    "  ~axis <trait> [n]"
)


def _norm(vec):
    import numpy as np
    v = np.asarray(vec, dtype="float32")
    return v / (float((v * v).sum()) ** 0.5 + 1e-9)


def _axis_report(term, n=6):
    from utils import persona_axes

    resolved = persona_axes.resolve_axis(term)
    if not resolved:
        return f"no axis for '{term}'"
    axis, sign, note = resolved
    axes = persona_axes._all_axis_vectors()
    av, pos, neg = axes[axis]
    target = _norm(av) * sign
    rows = []
    for name, (vec, pos_label, neg_label) in axes.items():
        if name == axis:
            continue
        sim = float(target @ _norm(vec))
        label = pos_label if sim >= 0 else neg_label
        rows.append((abs(sim), sim, name, label))
    rows.sort(reverse=True)
    n = max(1, min(12, int(n)))
    parts = [f"{name}:{sim:+.2f} ({label})" for _abs, sim, name, label in rows[:n]]
    pole = f"{pos} vs {neg}" if sign > 0 else f"{neg} vs {pos}"
    msg = f"axis {term} -> {axis} [{pole}] | nearest: " + " · ".join(parts)
    if note:
        msg += f" | {note}"
    return msg[:480]


async def handle_axis(bot, message, params):
    if not params:
        await message.channel.send("Usage: ~axis <trait> [n]")
        return
    term = params[0].lower()
    n = 6
    if len(params) > 1:
        try:
            n = int(params[1])
        except ValueError:
            pass
    await message.channel.send(_axis_report(term, n))
