"""Trait axes: project person-vectors onto directions defined by example
sentences — the first slice of 'second-order personality traits' from
docs/CHAT_PERSONALITY_RESEARCH.md.

Each axis is two poles, each pole a handful of chat-register example
sentences. The axis direction = mean(embed(pole B)) - mean(embed(pole A)),
and a chatter's score is their (centered) person-vector projected onto it,
z-scored across the roster — so +2.0 means 'two standard deviations more
B-pole than the average chatter here', not an absolute judgment.

Axis quality is only as good as the pole sentences; treat scores as a fun
mirror, not a diagnosis.
"""

import json
import urllib.error
import urllib.request

import config
from utils import persona_embeddings

# (negative-pole label, positive-pole label, negative examples, positive examples)
AXES = {
    "menace": (
        "wholesome", "menace",
        ["hope you have a great stream today",
         "that was really nice of you",
         "glad everyone is having a good time",
         "congrats man, well deserved",
         "take care of yourself, see you tomorrow"],
        ["you absolute waste of oxygen",
         "i will ruin your whole day for fun",
         "everyone in this chat is beneath me",
         "cry about it, nobody is coming to save you",
         "i hope your team loses every game forever"],
    ),
    "ironic": (
        "sincere", "ironic",
        ["i genuinely loved that movie, it moved me",
         "honestly this means a lot to me",
         "i really do care about this community",
         "no joke, that was impressive",
         "i'm being serious, that hurt my feelings"],
        ["oh yeah totally, best stream of all time, surely",
         "wow what an amazing take, never heard that one before",
         "yes because that worked so well last time",
         "ah yes, the classic strategy of losing on purpose",
         "truly the chess grandmaster of saying nothing"],
    ),
    "unhinged": (
        "chill", "unhinged",
        ["yeah that's fair enough",
         "no worries, it happens",
         "i'll probably just relax tonight",
         "sounds good man",
         "eh, not a big deal either way"],
        ["I AM GOING TO SCREAM UNTIL THE SUN EXPLODES",
         "i havent slept in four days and i can taste colors",
         "WHO SAID THAT. WHO. SAID. THAT.",
         "i am one bad pull from total meltdown",
         "deleting my account and moving into the woods TONIGHT"],
    ),
    "professor": (
        "brainrot", "professor",
        ["skibidi gyatt rizz lmao fr fr no cap",
         "bro is NOT him lil bro got ratio'd",
         "lmaooo dead 💀 actual npc behavior",
         "gg ez clap noob diff",
         "huh lol idk lmao"],
        ["the underlying incentive structure explains most of this behavior",
         "historically, this pattern repeats in every speculative market",
         "the etymology of that word is actually quite interesting",
         "if you consider the base rates, the conclusion is obvious",
         "there's a well-documented cognitive bias behind that"],
    ),
    "doomer": (
        "optimist", "doomer",
        ["it'll work out, it usually does",
         "next year is going to be great",
         "honestly things keep getting better",
         "we'll figure it out, no stress",
         "good things are coming, trust"],
        ["nothing ever gets better, why pretend",
         "we are all cooked, it's over",
         "no point planning, everything collapses anyway",
         "every year is somehow worse than the last",
         "hope is a scam invented to sell you things"],
    ),
}

_AXIS_VECS = None


def _embed(texts):
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    body = json.dumps({"model": config.LLM_EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(base + "/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return [d["embedding"] for d in json.load(r)["data"]]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:240]
        raise RuntimeError(f"embedding HTTP {exc.code}: {body}") from exc


def _axis_vectors():
    """RAW axis directions (mean(pos) - mean(neg), normalized). These are the
    directions used for axis BUILDING and merge decisions in persona_axes
    (synonym detection needs the un-rotated direction). For SCORING people, use
    ortho_axis_vectors() — the raw directions are mutually correlated (the five
    poles share a 'negativity' component), so raw projections double-count."""
    global _AXIS_VECS
    if _AXIS_VECS is None:
        import numpy as np
        vecs = {}
        for name, (_neg, _pos, neg_s, pos_s) in AXES.items():
            embs = _embed(neg_s + pos_s)
            neg = np.asarray(embs[:len(neg_s)], dtype="float32").mean(axis=0)
            pos = np.asarray(embs[len(neg_s):], dtype="float32").mean(axis=0)
            v = pos - neg
            vecs[name] = v / (float((v ** 2).sum()) ** 0.5 + 1e-9)
        _AXIS_VECS = vecs
    return _AXIS_VECS


_ORTHO_VECS = None


def ortho_axis_vectors():
    """The SCORING directions: the five built-in axes symmetrically
    orthogonalized (Löwdin, A_orth = (A Aᵀ)^{-1/2} A) so that a person's five
    trait z-scores stop double-counting a shared component.

    Why this and not Gram-Schmidt: the raw poles are correlated (menace·doomer
    ≈ 0.64 on bge-m3), so projecting onto raw directions makes the per-person
    scores correlate ~0.48 across the roster — 'all the axes feel the same'.
    Gram-Schmidt removes that too, but it is ORDER-DEPENDENT: whichever axis is
    last keeps only its residual (doomer stayed just 0.73 aligned with its own
    name). Löwdin is the orthogonal matrix closest to the raw set, so it shares
    the distortion evenly — every axis stays 0.92+ aligned with its label and
    no axis is privileged. Measured: score-correlation 0.483 raw → 0.302 here.
    See scripts/eval_geometry.py for the numbers and docs/RESEARCH_TO_APPLIED.md
    for the structured-transform-discovery lineage."""
    global _ORTHO_VECS
    if _ORTHO_VECS is None:
        import numpy as np
        raw = _axis_vectors()
        names = list(raw)
        A = np.vstack([np.asarray(raw[n], dtype="float64") for n in names])
        G = A @ A.T
        w, U = np.linalg.eigh(G)
        w = np.clip(w, 1e-9, None)
        inv_sqrt = U @ np.diag(w ** -0.5) @ U.T
        R = inv_sqrt @ A
        R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
        _ORTHO_VECS = {n: R[i].astype("float32") for i, n in enumerate(names)}
    return _ORTHO_VECS


def pole_map():
    """{pole_label: (axis, sign)} — both ends of every axis are queryable."""
    out = {}
    for axis, (neg, pos, _a, _b) in AXES.items():
        out[pos] = (axis, +1)
        out[neg] = (axis, -1)
    return out


def leaderboard(pole, n=5):
    """Top-n chatters toward a pole label: [(author, z), ...] or None if the
    pole isn't one of the configured axis ends."""
    import numpy as np
    target = pole_map().get((pole or "").lower())
    if not target or not persona_embeddings.available():
        return None
    axis, sign = target
    vectors = persona_embeddings._centered()
    names = list(vectors)
    M = np.vstack([vectors[a] for a in names])
    scores = M @ ortho_axis_vectors()[axis]
    z = (scores - scores.mean()) / (scores.std() or 1.0)
    order = (sign * z).argsort()[::-1][:n]
    return [(names[i], float(sign * z[i])) for i in order]


def traits_for(author):
    """[(axis_label, z)] sorted by |z|, or [] if the author has no vector.
    z is relative to the roster: +2 = far toward the axis name, -2 = far
    toward its opposite pole."""
    import numpy as np
    if not persona_embeddings.available():
        return []
    vectors = persona_embeddings._centered()
    canon = persona_embeddings.chat_archive.normalize_author(author)
    if canon not in vectors:
        return []
    axes = ortho_axis_vectors()   # decorrelated scoring dials (see ortho docstring)
    names = list(vectors)
    M = np.vstack([vectors[a] for a in names])
    out = []
    for axis, av in axes.items():
        scores = M @ av
        mu, sd = float(scores.mean()), float(scores.std()) or 1.0
        z = (float(vectors[canon] @ av) - mu) / sd
        out.append((axis, z))
    out.sort(key=lambda kv: -abs(kv[1]))
    return out
