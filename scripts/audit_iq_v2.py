"""Audit the current `~iq` artifact from its stored receipts.

The filename is retained for compatibility. This script does not re-embed a
second sample or call the judge: it audits exactly what the live cache serves,
ordered by archive volume, and writes private raw examples under data/unsynced.

    python scripts/audit_iq_v2.py --max-users 40 --examples 2
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import chat_archive, message_quality, persona_iq  # noqa: E402

OUT = Path("data/unsynced/iq_audit.md")
DIMENSIONS = ("reasoning", "abstraction", "vocab", "syntax", "breadth", "depth")


def quote(text: str, max_len: int = 240) -> str:
    text = " ".join((text or "").split())
    if len(text) > max_len:
        text = text[:max_len - 1].rstrip() + "..."
    return text.replace("|", "\\|")


def load_payload() -> dict:
    with open(persona_iq.CACHE, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("scores"), dict):
        raise RuntimeError("IQ cache is missing or has a legacy shape")
    if not persona_iq._cache_current(payload):
        raise RuntimeError(
            f"IQ cache is not live-safe: {persona_iq.cache_problem() or 'stale metadata'}"
        )
    return payload


def author_counts(authors: list[str]) -> dict[str, int]:
    counts = dict(chat_archive.canonical_author_counts(include_bots=True))
    return {author: int(counts.get(author, 0)) for author in authors}


def receipt_rows(row: dict) -> list[tuple[str, dict]]:
    return [
        (dimension, receipt)
        for dimension, receipts in (row.get("receipts") or {}).items()
        for receipt in (receipts or [])
    ]


def flags_for(row: dict) -> list[str]:
    flags = []
    if row.get("confidence") == "low":
        flags.append("low_confidence")
    if float(row.get("split_delta") or 0.0) >= 0.9:
        flags.append("unstable_split")
    if int(row.get("n_utterances") or 0) < persona_iq.DEFAULT_MAX_UTTERANCES:
        flags.append("under_fixed_budget")
    if row.get("vocab", 0) >= 1.0 and row.get("reasoning", 0) <= 0:
        flags.append("rare_vocab_without_reasoning")
    if row.get("depth", 0) >= 0.8 and row.get("reasoning", 0) <= 0:
        flags.append("niche_depth_without_reasoning")
    if row.get("syntax", 0) >= 1.0 and row.get("reasoning", 0) <= 0:
        flags.append("syntax_without_reasoning")

    seen = set()
    duplicate = command = pasted = False
    for _dimension, receipt in receipt_rows(row):
        text = str(receipt.get("text") or "")
        key = chat_archive.line_match_key(text)
        if key and key in seen:
            duplicate = True
        if key:
            seen.add(key)
        command = command or message_quality.command_like(text)
        pasted = pasted or message_quality.likely_pasted_prose(text)
    if duplicate:
        flags.append("duplicate_receipts")
    if command:
        flags.append("command_receipt")
    if pasted:
        flags.append("pasted_prose_receipt")
    return flags


def strongest_drivers(row: dict, n: int = 3) -> str:
    ordered = sorted(
        (row.get("components") or {}).items(), key=lambda item: -abs(item[1])
    )[:n]
    return ", ".join(f"{name} {value:+.2f}" for name, value in ordered)


def render_receipts(row: dict, examples: int) -> list[str]:
    lines = []
    receipts = row.get("receipts") or {}
    for dimension in DIMENSIONS:
        stored = receipts.get(dimension) or []
        if not stored:
            lines.append(f"  - {dimension}: none stored")
            continue
        lines.append(f"  - {dimension} {row.get(dimension, 0.0):+.2f}:")
        for receipt in stored[:examples]:
            feature = receipt.get("feature", dimension)
            value = receipt.get("value")
            score = f" ({float(value):.3f})" if isinstance(value, (int, float)) else ""
            lines.append(f"    - `{feature}`{score} {quote(receipt.get('text', ''))}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-users", type=int, default=40)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    payload = load_payload()
    meta = payload.get("__meta__") or {}
    scores = persona_iq._canonicalized_scores(payload["scores"])
    counts = author_counts(list(scores))
    ordered = sorted(scores, key=lambda author: (-counts.get(author, 0), author))[
        :args.max_users
    ]
    flag_counts = Counter(
        flag for author in ordered for flag in flags_for(scores[author])
    )

    lines = [
        "# Current Text-IQ Receipt Audit",
        "",
        "Private raw-evidence report. Users are ordered by archive volume, not score.",
        "",
        "## Build",
        "",
        f"- scorer version: {meta.get('version')}",
        f"- built: {meta.get('built_at')}",
        f"- authors: {len(scores)}",
        f"- fixed scoring budget: {meta.get('fixed_scoring_budget')}",
        f"- embedding layer: {meta.get('embedding_features')}",
        f"- judge layer: {meta.get('llm_judge')}",
        f"- build quality: {meta.get('build_quality', 'legacy')}",
        "",
        "## Systemic Flags",
        "",
    ]
    if flag_counts:
        lines.extend(f"- `{name}`: {count}" for name, count in flag_counts.most_common())
    else:
        lines.append("- No flags triggered.")
    lines.extend(["", "## Users By Archive Volume", ""])

    for index, author in enumerate(ordered, 1):
        row = scores[author]
        flags = flags_for(row)
        lines.extend([
            f"### {index}. {author}",
            "",
            (
                f"- messages: {counts.get(author, 0):,}; scored utterances: "
                f"{row.get('n_utterances', 0):,}; available: "
                f"{row.get('available_utterances', 0):,}"
            ),
            (
                f"- score: {row['iq']} (pct {row.get('percentile')}, "
                f"conf {row.get('confidence')}, z {row.get('z'):+.2f}, "
                f"split delta {row.get('split_delta')})"
            ),
            f"- strongest drivers: {strongest_drivers(row)}",
            f"- flags: {', '.join(flags) if flags else 'none'}",
            "",
            "Stored receipts:",
        ])
        lines.extend(render_receipts(row, max(1, args.examples)))
        lines.append("")

    lines.extend([
        "## Global Improvement Targets",
        "",
        "1. Label contamination classes across this volume-ordered roster; never tune a person-specific exception.",
        "2. Add near-copy and quotation detection for unique or lightly edited pasted prose.",
        "3. Compare independent evidence samples and channel holdouts before trusting rank movement.",
        "4. Benchmark judge/heuristic disagreement against human-labeled reasoning moves.",
        "5. Keep reasoning-register embeddings weak unless that held-out benchmark supports more weight.",
        "",
    ])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(ordered)} users, {sum(flag_counts.values())} flags)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
