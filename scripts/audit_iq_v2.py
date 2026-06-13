"""Audit the v2 `~iq` scorer with per-user receipts.

The output intentionally lives under data/unsynced because it includes raw chat
examples. Sorts by archive volume first, then shows each user's score,
component drivers, examples, and systemic failure flags.

    python scripts/audit_iq_v2.py --max-users 46 --per-author 180
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils import chat_archive, persona_iq  # noqa: E402

OUT = Path("data/unsynced/iq_audit.md")


def quote(text: str, max_len: int = 220) -> str:
    text = " ".join((text or "").split())
    if len(text) > max_len:
        text = text[:max_len - 1].rstrip() + "..."
    return text.replace("|", "\\|")


def score_lookup() -> dict:
    with open(persona_iq.CACHE, "rb") as fh:
        payload = pickle.load(fh)
    if "scores" in payload:
        return payload["scores"]
    return payload


def author_counts(authors: list[str]) -> dict[str, int]:
    conn = chat_archive.connect()
    out = {}
    prefix = getattr(config, "PREFIX", "~") + "%"
    for author in authors:
        keys = chat_archive.author_keys(author)
        ph = ",".join("?" for _ in keys)
        out[author] = conn.execute(
            f"SELECT COUNT(*) FROM messages WHERE author IN ({ph}) "
            "AND ltrim(content) NOT LIKE ?",
            [*keys, prefix],
        ).fetchone()[0]
    return out


def raw_metric_examples(rows, rarity, limit: int = 3):
    vocab, syntax, markers, questions = [], [], [], []
    for row in rows:
        toks = row["tokens"]
        text = row["clean"]
        if not toks:
            continue
        rarity_score = persona_iq._mean_rarity(toks, rarity)
        clause = len(persona_iq._CLAUSE_RE.findall(text))
        marker_count = sum(1 for t in toks if t in persona_iq._REASONING_MARKERS)
        marker_density = marker_count / math.sqrt(max(1, len(toks)))
        syntax_score = math.log1p(len(toks)) * (1.0 + clause + marker_density)
        q = 0.0
        if "?" in text:
            q += 0.35
        if persona_iq._QUESTION_RE.search(text):
            q += 1.0
        if q and len(toks) >= 7:
            q += min(0.6, math.log1p(len(toks)) / 6)
        if rarity_score is not None:
            vocab.append((rarity_score, row["raw"]))
        syntax.append((syntax_score, row["raw"]))
        markers.append((marker_density + clause, row["raw"]))
        questions.append((q, row["raw"]))
    return {
        "vocab": sorted(vocab, reverse=True)[:limit],
        "syntax": sorted(syntax, reverse=True)[:limit],
        "markers": sorted(markers, reverse=True)[:limit],
        "questions": sorted(questions, reverse=True)[:limit],
    }


def embed_examples(author_rows: dict[str, list[dict]], per_author: int, limit: int = 3):
    import numpy as np

    axes = persona_iq._axis_vectors()
    selected: dict[str, list[dict]] = {}
    mats = {}
    all_mats = []
    for author, rows in author_rows.items():
        eligible = [
            r for r in rows
            if 4 <= len(r["tokens"]) <= 80 and len(r["clean"]) <= 700
        ]
        eligible = persona_iq._sample(eligible, per_author, f"{author}:iq-audit")
        if len(eligible) < 10:
            continue
        vecs = []
        for i in range(0, len(eligible), 64):
            vecs.extend(persona_iq._embed_batch([r["clean"] for r in eligible[i:i + 64]]))
        mat = persona_iq._normalize_matrix(vecs)
        selected[author] = eligible
        mats[author] = mat
        all_mats.append(mat)
    if not all_mats:
        return {}
    global_mean = np.vstack(all_mats).mean(axis=0)
    out = {}
    for author, mat in mats.items():
        rows = selected[author]
        author_out = {}
        for axis in ("causal", "nuance", "connections", "problem_solving",
                     "metacognition", "abstraction", "technical"):
            vals = mat @ axes[axis]
            top = vals.argsort()[::-1][:limit]
            author_out[axis] = [(float(vals[i]), rows[i]["raw"]) for i in top]
        spec = np.linalg.norm(mat - global_mean, axis=1)
        top = spec.argsort()[::-1][:limit]
        author_out["specificity"] = [(float(spec[i]), rows[i]["raw"]) for i in top]
        out[author] = author_out
    return out


def flags_for(row: dict) -> list[str]:
    flags = []
    if row.get("confidence") == "low":
        flags.append("low_confidence")
    if row.get("split_delta", 0) and row["split_delta"] >= 0.9:
        flags.append("unstable_split")
    if row.get("vocab", 0) >= 1.0 and row.get("reasoning", 0) <= 0:
        flags.append("rare_vocab_over_reward")
    if row.get("depth", 0) >= 0.8 and row.get("reasoning", 0) <= 0:
        flags.append("niche_depth_without_reasoning")
    if row.get("syntax", 0) >= 1.0 and row.get("reasoning", 0) <= 0:
        flags.append("verbosity_marker_over_reward")
    if row.get("abstraction", 0) >= 1.0 and row.get("reasoning", 0) <= 0:
        flags.append("abstract_register_without_reasoning")
    if row.get("reasoning", 0) >= 1.0 and row.get("vocab", 0) <= -0.75:
        flags.append("plain_language_reasoning")
    return flags


def strongest_drivers(row: dict, n: int = 3) -> str:
    comps = row.get("components", {})
    ordered = sorted(comps.items(), key=lambda kv: -abs(kv[1]))[:n]
    return ", ".join(f"{k} {v:+.2f}" for k, v in ordered)


def render_examples(title: str, rows, max_items: int = 2) -> list[str]:
    out = [f"  - {title}:"]
    usable = [(score, text) for score, text in rows if score > 0]
    if not usable:
        return out + ["    - none found"]
    for score, text in usable[:max_items]:
        out.append(f"    - `{score:.3f}` {quote(text)}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-users", type=int, default=46)
    parser.add_argument("--per-author", type=int, default=180,
                        help="utterances to embed per author for audit receipts")
    parser.add_argument("--author-cap", type=int, default=12000,
                        help="utterances to inspect for lexical receipts")
    parser.add_argument("--word-freq-sample", type=int, default=200000)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    scores = score_lookup()
    authors = list(scores)
    counts = author_counts(authors)
    ordered = sorted(authors, key=lambda a: (-counts.get(a, 0), a))[:args.max_users]

    freqs, total = persona_iq._word_freqs(args.word_freq_sample)
    rarity = persona_iq._rarity_fn(freqs, total)
    author_rows = {}
    raw_examples = {}
    for author in ordered:
        rows = persona_iq._utterance_rows(author, author_cap=args.author_cap)
        rows = [row for row in rows if persona_iq._row_has_corpus_signal(row, rarity)]
        author_rows[author] = rows
        raw_examples[author] = raw_metric_examples(rows, rarity, limit=args.examples)
        print(f"loaded {author}: {len(rows)} utterances", flush=True)

    embed = embed_examples(author_rows, args.per_author, limit=args.examples)

    flag_counts = Counter()
    for author in ordered:
        flag_counts.update(flags_for(scores[author]))

    lines = [
        "# IQ v2 Audit",
        "",
        "Private raw-evidence report. Sorted by archive message volume, not score.",
        "",
        "## Summary",
        "",
        f"- Users audited: {len(ordered)}",
        f"- Embedded audit utterances per user: up to {args.per_author}",
        f"- Scorer cache: `{persona_iq.CACHE}`",
        "",
        "### Systemic Flags",
        "",
    ]
    if flag_counts:
        for name, count in flag_counts.most_common():
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- No flags triggered.")

    lines.extend([
        "",
        "## Users By Archive Volume",
        "",
    ])

    for idx, author in enumerate(ordered, 1):
        row = scores[author]
        flags = flags_for(row)
        lines.extend([
            f"### {idx}. {author}",
            "",
            (
                f"- messages: {counts.get(author, 0):,}; utterances inspected: "
                f"{len(author_rows.get(author, [])):,}"
            ),
            (
                f"- score: {row['iq']} (pct {row.get('percentile')}, "
                f"conf {row.get('confidence')}, z {row.get('z'):+.2f}, "
                f"split_delta {row.get('split_delta')})"
            ),
            f"- strongest drivers: {strongest_drivers(row)}",
            f"- flags: {', '.join(flags) if flags else 'none'}",
            "",
            "Examples:",
        ])
        rex = raw_examples.get(author, {})
        emb = embed.get(author, {})
        lines.extend(render_examples("vocab rarity", rex.get("vocab", []), args.examples))
        lines.extend(render_examples("syntax / markers", rex.get("syntax", []), args.examples))
        lines.extend(render_examples("causal-reasoning axis", emb.get("causal", []), args.examples))
        lines.extend(render_examples("nuance axis", emb.get("nuance", []), args.examples))
        lines.extend(render_examples("connections axis", emb.get("connections", []), args.examples))
        lines.extend(render_examples("niche specificity", emb.get("specificity", []), args.examples))
        lines.append("")

    lines.extend([
        "## Global Improvement Targets",
        "",
        "1. Add an evidence command/report to show which utterances drove each component.",
        "2. Penalize high niche-depth when the same utterances do not also score on reasoning axes.",
        "3. Split rare vocabulary into domain terminology vs usernames/emotes/lore leftovers.",
        "4. Add an optional LLM judge pass over embedding-selected candidates to reject jargon-only lines.",
        "5. Store audit receipts during the main build so inspection does not require re-embedding.",
        "6. Calibrate against split-half and channel-holdout stability before trusting rank movement.",
        "",
    ])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
