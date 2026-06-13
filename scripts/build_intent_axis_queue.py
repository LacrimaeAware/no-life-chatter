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
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from scripts.train_intent_probes import row_text, try_embed  # noqa: E402
from utils import chat_archive, persona_classifier as pc  # noqa: E402


QUEUE_OUT = Path(
    "../ai-prompt-engineering/private_docs/review_queues/"
    "nolifechatter_intent_axes_v2.jsonl"
)
REPORT_OUT = Path("_private/INTENT_QUEUE_V2_BUILD.md")
MODEL_IN = Path("data/unsynced/intent_probes.pkl")
AUTO_INVALID_OUT = Path("data/unsynced/oracle/intent_v2_auto_invalid.jsonl")

BOT_AUTHORS = {
    "nightbot", "streamelements", "fossabot", "moobot", "wizebot", "streamlabs",
    "supibot", "potatbotat", "weirdfarts1av",
} | {str(u).lower() for u in getattr(config, "EXCLUDE_USERS", set())}
MOD_NOTICE_RE = re.compile(
    r"\b(has been timed out|has been timed-out|has been banned|timed out for|"
    r"no matching emotes found|removed \d+ emotes?|added \d+ emotes?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AxisQueueSpec:
    target: str
    axis_label: str
    select_label: str
    question: str
    guidance: str
    options: list[str]
    option_labels: dict[str, str]
    option_help: dict[str, str]
    include_realish_only: bool = True


AXES = [
    AxisQueueSpec(
        target="valid_utterance",
        axis_label="validity",
        select_label="not_valid",
        question="Validity",
        guidance=(
            "Should this row be eligible for semantic/intent training? Use not_valid for "
            "bot output, mod notices, pure art, pure links, command output, or fragments "
            "too opaque even with context. Short human messages can still be valid."
        ),
        options=["valid", "not_valid", "unclear"],
        option_labels={
            "valid": "valid: human/chat meaning",
            "not_valid": "not valid: bot/junk/noise",
            "unclear": "unclear",
        },
        option_help={
            "valid": "A human/chat utterance with interpretable communicative intent, even if short, emote-coded, or slang.",
            "not_valid": "Bot output, mod notice, pure link, pure ASCII/image text, command output, or unusable noise.",
            "unclear": "You cannot tell whether it is meaningful enough to train on.",
        },
        include_realish_only=False,
    ),
    AxisQueueSpec(
        target="literal_alignment",
        axis_label="literal_alignment",
        select_label="divergent",
        question="Literal alignment",
        guidance=(
            "Binary question: is this used straight, or as reversal/irony? Treat conventional "
            "chat meaning as literal enough: emotes, greetings, tucks, and reaction phrases are "
            "aligned when used in their normal way. Hyperbole is usually aligned because the "
            "stance direction is the same; magnitude distortion is a different axis."
        ),
        options=["aligned", "divergent", "unclear"],
        option_labels={
            "aligned": "straight / aligned",
            "divergent": "reversal / ironic gap",
            "unclear": "unclear / no stable meaning",
        },
        option_help={
            "aligned": "They mean the conventional chat meaning: normal slang, emote/action shorthand, greeting, reaction, or hyperbole with the same stance direction.",
            "divergent": "The message is used as irony/reversal: it says or performs one thing while intending the opposite or a sideways stance.",
            "unclear": "You cannot infer the conventional meaning or whether it is straight vs ironic.",
        },
    ),
    AxisQueueSpec(
        target="magnitude_distortion",
        axis_label="magnitude_distortion",
        select_label="overstated",
        question="Magnitude distortion",
        guidance=(
            "Literal/normal is ordinary strength. Overstated is hyperbole or exaggerated "
            "magnitude. Understated is deliberately downplayed. This axis is separate from irony."
        ),
        options=["literal_or_normal", "overstated", "understated", "unclear", "not_applicable"],
        option_labels={
            "literal_or_normal": "normal magnitude",
            "overstated": "overstated / hyperbolic",
            "understated": "understated",
            "unclear": "unclear",
            "not_applicable": "no magnitude to judge",
        },
        option_help={
            "literal_or_normal": "The strength is about what they mean.",
            "overstated": "The direction may be sincere, but the intensity is exaggerated.",
            "understated": "The wording downplays what is meant.",
            "unclear": "There is a magnitude claim, but you cannot judge distortion.",
            "not_applicable": "No claim/reaction intensity to compare.",
        },
    ),
    AxisQueueSpec(
        target="play_frame",
        axis_label="play_frame",
        select_label="playful",
        question="Play frame",
        guidance=(
            "Low/none is mostly plain serious talk. Playful is framed as a bit or joke. "
            "Masking-play is joke form that appears to cover criticism, aggression, or status work."
        ),
        options=["low_or_none", "playful", "masking_play", "unclear", "not_applicable"],
        option_labels={
            "low_or_none": "not playful",
            "playful": "playful / bit",
            "masking_play": "playful but masking",
            "unclear": "unclear",
            "not_applicable": "no play-frame signal",
        },
        option_help={
            "low_or_none": "Mostly plain serious or neutral talk.",
            "playful": "Clearly framed as a bit, joke, riff, or unserious performance.",
            "masking_play": "Joke form seems to cover criticism, aggression, status, or risky stance.",
            "unclear": "Cannot tell if it is a bit.",
            "not_applicable": "No usable framing signal.",
        },
    ),
    AxisQueueSpec(
        target="masking_facework",
        axis_label="masking_facework",
        select_label="present_or_possible",
        question="Masking / facework",
        guidance=(
            "Absent means no obvious cover. Possible/present means humor or irony seems to launder "
            "criticism, aggression, status, or a socially risky stance."
        ),
        options=["absent", "possible", "present", "unclear", "not_applicable"],
        option_labels={
            "absent": "not masking",
            "possible": "possibly masking",
            "present": "masking present",
            "unclear": "unclear",
            "not_applicable": "no masking signal",
        },
        option_help={
            "absent": "No obvious use of humor/irony as cover.",
            "possible": "Could be cover, but weak evidence.",
            "present": "Humor/irony is doing cover work for aggression, status, criticism, or risky stance.",
            "unclear": "Cannot tell whether cover is happening.",
            "not_applicable": "No relevant social/facework signal.",
        },
    ),
    AxisQueueSpec(
        target="hostility",
        axis_label="hostility",
        select_label="hostile_or_mock",
        question="Hostility",
        guidance=(
            "Low/none is not hostile. Mild/mock is teasing, mockery, or casual insult. "
            "Present is direct hostility or aggressive attack."
        ),
        options=["low_or_none", "mild_or_mock", "present", "unclear", "not_applicable"],
        option_labels={
            "low_or_none": "not hostile",
            "mild_or_mock": "mild/mock hostility",
            "present": "hostile",
            "unclear": "unclear",
            "not_applicable": "no hostility signal",
        },
        option_help={
            "low_or_none": "No meaningful hostile/mock energy.",
            "mild_or_mock": "Teasing, casual insult, mockery, or light aggression.",
            "present": "Direct hostility or attack.",
            "unclear": "Cannot judge the hostility.",
            "not_applicable": "No usable interpersonal signal.",
        },
    ),
    AxisQueueSpec(
        target="shock_attention",
        axis_label="shock_attention",
        select_label="present",
        question="Shock / attention",
        guidance=(
            "Present means shock value or attention-bid energy is the point. Low/none is ordinary "
            "chat, even if rude or dumb."
        ),
        options=["low_or_none", "present", "unclear", "not_applicable"],
        option_labels={
            "low_or_none": "not shock/attention",
            "present": "shock / attention bid",
            "unclear": "unclear",
            "not_applicable": "no shock signal",
        },
        option_help={
            "low_or_none": "Not primarily trying to shock or grab attention.",
            "present": "Shock value or attention seeking is the point.",
            "unclear": "Cannot tell if it is attention-bid behavior.",
            "not_applicable": "No usable signal.",
        },
    ),
]


def obvious_invalid_reason(author: str, content: str) -> str | None:
    content = content or ""
    stripped = content.lstrip()
    norm_author = chat_archive.normalize_author(author or "")
    if norm_author in BOT_AUTHORS:
        return "known_bot_author"
    if not stripped:
        return "empty"
    if stripped.startswith("Replying to @"):
        return "client_reply_header"
    if stripped.startswith(("http://", "https://")):
        return "pure_link"
    if MOD_NOTICE_RE.search(stripped):
        return "bot_or_mod_notice_text"
    if len(stripped) > 80:
        printable = sum(ch.isalnum() or ch.isspace() for ch in stripped)
        if printable / max(len(stripped), 1) < 0.45:
            return "ascii_or_symbol_art"
    return None


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
        window = chat_archive.context_window(mid, channel, before=10, after=5)
        context = "\n".join(f"{a}: {c[:140]}" for _i, a, c in window)
        invalid_reason = obvious_invalid_reason(author, content)
        out.append({
            "id": mid,
            "channel": channel,
            "author": author,
            "content": content,
            "context": context,
            "realish": invalid_reason is None and is_realish_message(content),
            "obvious_invalid": invalid_reason,
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
    pool = [
        (c, s) for c, s in zip(candidates, scores)
        if not c["obvious_invalid"] and (c["realish"] or not spec.include_realish_only)
    ]
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
        "guidance": spec.guidance,
        "option_labels": spec.option_labels,
        "option_help": spec.option_help,
        "subject": {
            "title": candidate["content"][:80],
            "axis": spec.axis_label,
            "channel": candidate["channel"],
            "author": candidate["author"],
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


def write_auto_invalid(path: Path, candidates: list[dict]) -> int:
    rows = [c for c in candidates if c["obvious_invalid"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({
                "source": "intent_axis_queue_v2_auto_filter",
                "label": "not_valid",
                "reason": row["obvious_invalid"],
                "source_message_id": row["id"],
                "channel": row["channel"],
                "author": row["author"],
                "message": row["content"],
            }, ensure_ascii=False) + "\n")
    return len(rows)


def write_report(
    path: Path, args: argparse.Namespace, candidates: list[dict], items: list[dict],
    auto_invalid_n: int,
) -> None:
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
        f"Obvious invalids auto-filtered: {auto_invalid_n}",
        f"Items written: {len(items)}",
        f"Queue path: `{args.output}`",
        f"Auto-invalid path: `{args.auto_invalid_out}`",
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
    parser.add_argument("--auto-invalid-out", type=Path, default=AUTO_INVALID_OUT)
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
    auto_invalid_n = write_auto_invalid(args.auto_invalid_out, candidates)
    write_jsonl(args.output, items)
    write_report(args.report, args, candidates, items, auto_invalid_n)
    print(f"{len(items)} items -> {args.output}")
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
