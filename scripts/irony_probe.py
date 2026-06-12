"""Second-order irony probe — operationalizing the user's theory
(docs/CHAT_PERSONALITY_RESEARCH.md "Second-order semantics").

First-order = what the words literally say (surface-sarcasm axis catches the
"almost like..." constructions). Second-order = literal content vs values:
a message whose LITERAL semantics are extreme on the harm/menace axis, made
casually in friendly chat, is usually a bit — deadpan irony that the surface
axis reads as 'sincere' BY DESIGN. Rule probed here:

    surface says ironic                 -> marked sarcasm (weak irony)
    surface says sincere + extreme harm -> deadpan irony candidate
    surface says sincere + mild content -> probably sincere

Also merges fragments into utterances first ("I wish" + the punchline sent
seconds later are one turn).

    python scripts/irony_probe.py            # demo on a built-in exchange
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from utils import chat_archive, persona_classifier as pc  # noqa: E402
from utils.persona_traits import _axis_vectors, _embed  # noqa: E402

# a real exchange, anonymized author keys (a/b), with timestamps for merging
DEMO = [
    ("2026-06-12 14:35:00", "a", "almost like you aren't competing against people who have been playing the same exact unchanged mode for 20 years THINKING"),
    ("2026-06-12 14:39:00", "b", "it is mutton busting day man"),
    ("2026-06-12 14:39:10", "b", "i will not be participating"),
    ("2026-06-12 14:39:30", "a", "I wish"),
    ("2026-06-12 14:39:40", "a", "It would be a great day to cause suffering to children"),
    ("2026-06-12 14:40:00", "b", "I will laugh at the children extra hard just for you"),
    ("2026-06-12 14:44:00", "a", "Thank you"),
    ("2026-06-12 14:44:10", "a", "I do feel like being a normie today"),
    ("2026-06-12 14:52:00", "b", "can't discuss the retarded internet shit with them because they just won't get it"),
    ("2026-06-12 14:59:00", "a", "That's good FeelsOkayMan"),
    ("2026-06-12 14:59:10", "a", "God loves them more FeelsOkayMan"),
]


def calibrate(ax, n=400):
    """mu/sd of an axis over a random archive sample, embedded fresh (so the
    calibration always matches the CURRENT embedder)."""
    conn = chat_archive.connect()
    msgs = [r[0] for r in conn.execute(
        "SELECT content FROM messages WHERE LENGTH(content) > 12 "
        "ORDER BY RANDOM() LIMIT ?", (n,))]
    embs = []
    for i in range(0, len(msgs), 64):
        embs.extend(_embed(msgs[i:i + 64]))
    E = np.asarray(embs, dtype="float32")
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    s = E @ ax
    return float(s.mean()), float(s.std())


def main():
    axes = _axis_vectors()
    iron, harm = np.asarray(axes["ironic"]), np.asarray(axes["menace"])
    mu_i, sd_i = calibrate(iron)
    mu_h, sd_h = calibrate(harm)

    merged = chat_archive.merge_utterances(DEMO)
    texts = [pc.strip_emote_tokens(c) for _s, _a, c in merged]
    embs = _embed(texts)
    print(f"{len(DEMO)} raw messages -> {len(merged)} merged utterances\n")
    print(f"{'surface':>8} {'harm':>6}  verdict")
    for (s_, a, c), e in zip(merged, embs):
        v = np.asarray(e, dtype="float32")
        v /= (np.linalg.norm(v) + 1e-9)
        zi = (float(v @ iron) - mu_i) / sd_i
        zh = (float(v @ harm) - mu_h) / sd_h
        if zi > 0.5:
            verdict = "marked sarcasm (weak irony)"
        elif zh > 1.2:
            verdict = "DEADPAN IRONY candidate (sincere surface, extreme content)"
        elif zi < -0.5:
            verdict = "probably sincere"
        else:
            verdict = "unclear"
        print(f"  {zi:+7.2f} {zh:+6.2f}  {verdict:<58} [{a}] {c[:70]}")


if __name__ == "__main__":
    main()
