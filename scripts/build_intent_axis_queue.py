"""Build a v2 intent-axis oracle queue from archive messages.

The review tool expects one item = one question, so this does not bundle seven
questions into one card.  It emits one focused axis question per item and uses
the current intent probes only for sampling: high-probability positives and
near-boundary cases get prioritized for human review.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_intent_probes import row_text, try_embed  # noqa: E402
from utils import chat_archive, persona_classifier as pc  # noqa: E402


QUEUE_OUT = Path(
    "../ai-prompt-engineering/private_docs/review_queues/"
    "nolifechatter_intent_axes_v2.jsonl"
)
REPORT_OUT = Path("_private/INTENT_QUEUE_V2_BUILD.md")
MODEL_IN = Path("data/unsynced/intent_probes.pkl")


@dataclass(frozen=True)
class AxisQueueSpec:
    target: str
    axis_label: str
    select_label: str
    question: str
    options: list[str]
    include_realish_only: bool = True


AXES = [
    AxisQueueSpec(
        target="valid_utterance",
        axis_label="validity",
        select_label="not_valid",
        question="Is the MARKED message a real human utterance worth semantic labeling?",
        options=["valid", "not_valid", "unclear"],
        include_realish_only=False,
    ),
    AxisQueueSpec(
        target="literal_alignment",
        axis_label="literal_alignment",
        select_label="divergent",
        question="Do the literal words and intended stance align, or is there a reversal/gap?",
        options=["aligned", "divergent", "unclear", "not_applicable"],
    ),
    AxisQueueSpec(
        target="magnitude_distortion",
        axis_label="magnitude_distortion",
        select_label="overstated",
        question="Is the message normal/literal, overstated, or understated?",
        options=["literal_or_normal", "overstated", "understated", "unclear", "not_applicable"],
    ),
    AxisQueueSpec(
        target="play_frame",
        axis_label="play_frame",
        select_label="playful",
        question="Is the message framed as play/a bit, or mostly plain serious talk?",
        options=["low_or_none", "playful", "masking_play", "unclear", "not_applicable"],
    ),
    AxisQueueSpec(
        target="masking_facework",
        axis_label="masking_facework",
        select_label="present_or_possible",
        question="Is irony/play being used as cover for criticism, aggression, or status work?",
        options=["absent", "possible", "present", "unclear", "not_applicable"],
    ),
    AxisQueueSpec(
        target="hostility",
        axis_label="hostility",
        select_label="hostile_or_mock",
        question="How much hostile or mocking energy is present?",
        options=["low_or_none", "mild_or_mock", "present", "unclear", "not_applicable"],
    ),
    AxisQueueSpec(
        target="shock_attention",
        axis_label="shock_attention",
        select_label="present",
        question="Is this a shock-value / attention-bid message?",
        options=["low_or_none", "present", "unclear", "not_applicable"],
    ),
]


def is_realish_message(content: str) -> bool:
    stripped = (content or "").lstrip()
    if not stripped:
        return False
    if stripped.startswith("Replying to @"):
        return False
    if stripped[0] in {"<", "!"}:
        return False
    if stripped.startswith(("http://", "https://")):
        return False
    return pc._usable(content)


def load_candidates(limit: int, since: str) -> list[dict]:
    conn = chat_archive.connect()
    rows = conn.execute(
        "SELECT id, channel, author, content FROM messages "
        "WHERE LENGTH(content) BETWEEN 4 AND 260 AND sent_at >= ? "
        "ORDER BY RANDOM() LIMIT ?",
        (since, limit),
    ).fetchall()
    out = []
    for mid, channel, author, content in rows:
        window = chat_archive.context_window(mid, channel, before=4, after=2)
        context = "\n".join(f"{a}: {c[:140]}" for _i, a, c in window)
        out.append({
            "id": mid,
            "channel": channel,
            "author": author,
            "content": content,
            "context": context,
            "realish": is_realish_message(content),
        })
    return out


def _candidate_text(candidate: dict, bundle: dict) -> str:
    return row_text(
        {"subject": {"message": candidate["content"], "context": candidate["context"]}},
        bool(bundle.get("include_context", True)),
        bool(bundle.get("emote_tags", True)),
    )


def _classes(model) -> list[str]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps.get("clf"), "classes_", None)
    return [str(c) for c in classes]


def label_probabilities(model, features, label: str) -> list[float]:
    classes = _classes(model)
    if hasattr(model, "predict_proba") and label in classes:
        probs = model.predict_proba(features)
        idx = classes.index(label)
        return [float(row[idx]) for row in probs]
    # Last-resort ranking for classifiers that expose only a decision function.
    scores = model.decision_function(features)
    if getattr(scores, "ndim", 1) > 1 and label in classes:
        idx = classes.index(label)
        scores = scores[:, idx]
    return [1.0 / (1.0 + pow(2.718281828, -float(s))) for s in scores]


def score_candidates(bundle: dict, candidates: list[dict]) -> dict[str, list[float]]:
    texts = [_candidate_text(c, bundle) for c in candidates]
    feature_mode = bundle.get("feature_mode")
    if feature_mode == "embedding":
        vecs, note = try_embed(texts)
        if vecs is None:
            raise RuntimeError(f"Could not embed queue candidates: {note}")
        features = vecs
    else:
        features = texts

    out = {}
    for spec in AXES:
        target = bundle.get("targets", {}).get(spec.target)
        if not target:
            continue
        out[spec.target] = label_probabilities(target["model"], features, spec.select_label)
    return out


def select_axis_items(candidates: list[dict], scores: list[float], spec: AxisQueueSpec, n: int):
    pool = [(c, s) for c, s in zip(candidates, scores) if c["realish"] or not spec.include_realish_only]
    high_count = max(1, n // 2)
    boundary_count = max(0, n - high_count)
    selected = []
    seen = set()
    for reason, ordered in [
        ("high_selected_label_probability", sorted(pool, key=lambda cs: cs[1], reverse=True)),
        ("near_boundary", sorted(pool, key=lambda cs: abs(cs[1] - 0.5))),
    ]:
        quota = high_count if reason.startswith("high") else boundary_count
        for candidate, score in ordered:
            if len([x for x in selected if x[2] == reason]) >= quota:
                break
            if candidate["id"] in seen:
                continue
            seen.add(candidate["id"])
            selected.append((candidate, score, reason))
    return selected[:n]


def make_item(rank: int, spec: AxisQueueSpec, candidate: dict, score: float, reason: str) -> dict:
    return {
        "id": f"nlc-intent-v2-{spec.axis_label}-{rank:04d}",
        "source_project": "NoLifeChatter",
        "kind": "single-classification",
        "question": spec.question,
        "subject": {
            "title": candidate["content"][:80],
            "axis": spec.axis_label,
            "message": candidate["content"],
            "context": candidate["context"],
        },
        "evidence": {
            "selection_reason": reason,
            "selected_label": spec.select_label,
            "selected_label_probability": round(score, 3),
            "source_message_id": candidate["id"],
        },
        "options": spec.options,
        "allow_other": True,
        "answer": None,
        "answer_note": None,
        "answered_at": None,
    }


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_report(path: Path, args: argparse.Namespace, candidates: list[dict], items: list[dict]) -> None:
    counts = {}
    for item in items:
        axis = item["subject"]["axis"]
        counts[axis] = counts.get(axis, 0) + 1
    lines = [
        "# Intent Axis Queue v2 Build",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        f"Candidate rows sampled: {len(candidates)}",
        f"Real-ish candidates: {sum(1 for c in candidates if c['realish'])}",
        f"Items written: {len(items)}",
        f"Queue path: `{args.output}`",
        "",
        "Axis counts:",
    ]
    for axis, count in sorted(counts.items()):
        lines.append(f"- {axis}: {count}")
    lines.extend([
        "",
        "Sampling policy: for each axis, half high selected-label probability and",
        "half near the model boundary. The probe is only a sampler; the human",
        "review answer remains ground truth.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_IN)
    parser.add_argument("--output", type=Path, default=QUEUE_OUT)
    parser.add_argument("--report", type=Path, default=REPORT_OUT)
    parser.add_argument("--candidate-n", type=int, default=700)
    parser.add_argument("--per-axis", type=int, default=20)
    parser.add_argument("--since", default="2025-01-01")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        print(f"missing intent probe model: {args.model}")
        print("Run 12-train-intent-probes.bat first.")
        return 2
    with args.model.open("rb") as fh:
        bundle = pickle.load(fh)
    candidates = load_candidates(args.candidate_n, args.since)
    scores = score_candidates(bundle, candidates)
    items = []
    for spec in AXES:
        if spec.target not in scores:
            continue
        selected = select_axis_items(candidates, scores[spec.target], spec, args.per_axis)
        for candidate, score, reason in selected:
            items.append(make_item(len(items), spec, candidate, score, reason))
    write_jsonl(args.output, items)
    write_report(args.report, args, candidates, items)
    print(f"{len(items)} items -> {args.output}")
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
