"""Country-affinity probe: which country do a chatter's messages orbit?

The experiment: embed first-person residence anchors per country ("i live in
Germany", "im from Germany", ...), score every roster chatter's per-message
vectors against each country (mean of the top-K cosines = peak topical
affinity), then **z-score each country ACROSS the roster**. The z-scoring is
the important part: without it, a person can look "high on China" only
because nobody else ever mentions China, while genuinely-German chatters
drown in a sea of other Germans. Roster-relative z asks "does THIS person
talk near this country unusually much for this chat?"

Diagnostic only — this reads topical affinity, not residency. A person who
argues about French politics daily scores France without being French. Treat
it as a lead generator for the profile builder's location slot, and always
read the receipt lines before believing anything.

    python scripts/probe_location.py someuser
    python scripts/probe_location.py someuser --top 8 --receipts 3
"""

from __future__ import annotations

import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from utils import chat_archive, persona_iq, persona_msg_index as pmi  # noqa: E402

COUNTRIES = [
    "Germany", "the United States", "the United Kingdom", "France", "Poland",
    "Sweden", "Norway", "Denmark", "Finland", "the Netherlands", "Spain",
    "Italy", "Portugal", "Brazil", "Canada", "Australia", "Austria",
    "Switzerland", "the Czech Republic", "Russia", "Ukraine", "Turkey",
    "India", "Japan",
]

ANCHOR_TEMPLATES = [
    "i live in {c}",
    "i am from {c}",
    "here in {c} that is how it works",
    "{c} is my home country",
]


def _embed_chunked(texts, chunk=24):
    out = []
    for i in range(0, len(texts), chunk):
        out.extend(persona_iq._embed_batch(texts[i:i + chunk]))
    return out


def _country_centroids():
    texts, spans = [], []
    for country in COUNTRIES:
        start = len(texts)
        texts.extend(t.format(c=country) for t in ANCHOR_TEMPLATES)
        spans.append((country, start, len(texts)))
    vecs = np.asarray(_embed_chunked(texts), dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    out = {}
    for country, a, b in spans:
        c = vecs[a:b].mean(axis=0)
        out[country] = c / (np.linalg.norm(c) + 1e-9)
    return out


def _person_scores(author: str, centroids: dict, top_k: int):
    """{country: mean of top-K per-message cosines} for one chatter."""
    try:
        V, texts = pmi._load(author)
    except Exception:
        return None, None
    V = np.asarray(V, dtype="float32")
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    scores, receipts = {}, {}
    for country, c in centroids.items():
        sims = V @ c
        k = min(top_k, len(sims))
        idx = np.argsort(sims)[::-1][:k]
        scores[country] = float(sims[idx].mean())
        receipts[country] = [(float(sims[i]), texts[i]) for i in idx[:5]]
    return scores, receipts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("author")
    ap.add_argument("--top", type=int, default=6, help="countries to show")
    ap.add_argument("--top-k", type=int, default=20,
                    help="per-person: mean over this many best messages")
    ap.add_argument("--receipts", type=int, default=3)
    args = ap.parse_args()

    target = chat_archive.normalize_author(args.author)
    if not pmi.available(target):
        print(f"No message index for {target} (run scripts/build_message_index.py).")
        return 1

    print(f"Embedding {len(COUNTRIES)} country anchors...")
    centroids = _country_centroids()

    # roster pass for the base-rate correction
    roster = [a for a in os.listdir(pmi.DIR) if not a.endswith(".tmp")]
    roster = sorted({os.path.splitext(a)[0] for a in roster})
    per_person: dict[str, dict] = {}
    for author in roster:
        scores, _ = _person_scores(author, centroids, args.top_k)
        if scores:
            per_person[author] = scores
    if target not in per_person:
        print(f"{target} has an index but no scores — aborting.")
        return 1
    print(f"Scored {len(per_person)} roster chatters for the base rate.\n")

    rows = []
    _, target_receipts = _person_scores(target, centroids, args.top_k)
    for country in COUNTRIES:
        vals = np.asarray([per_person[a][country] for a in per_person])
        mu, sd = vals.mean(), vals.std() or 1e-9
        raw = per_person[target][country]
        rows.append((country, (raw - mu) / sd, raw))
    rows.sort(key=lambda r: -r[1])

    print(f"=== {target}: country affinity (z vs {len(per_person)}-chatter roster) ===")
    for country, z, raw in rows[:args.top]:
        print(f"  {country:22s} z={z:+.2f}  raw={raw:.3f}")
        for sim, text in target_receipts[country][:args.receipts]:
            print(f"      {sim:.2f}  {' '.join(text.split())[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
