"""Train small intent probes from the private oracle label dataset.

This is the first supervised pass after the v1 irony review queue.  The
important change is that "irony" is no longer treated as one overloaded label:
we train separate heads for literal alignment, hyperbole, play frame, masking,
hostility, and shock/attention.

Inputs and outputs live under ignored paths by default:

    python scripts/train_intent_probes.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_INPUT = Path("data/unsynced/oracle/irony_v1_multi_axis.jsonl")
DEFAULT_MODEL_OUT = Path("data/unsynced/intent_probes.pkl")
DEFAULT_REPORT_OUT = Path("_private/INTENT_PROBES_REPORT.md")


@dataclass(frozen=True)
class TargetSpec:
    name: str
    title: str
    description: str
    positive: str | None
    labeler: Callable[[dict], str | None]


def _axis(row: dict, key: str):
    return (row.get("axes") or {}).get(key)


def _is_valid(row: dict) -> bool:
    return _axis(row, "valid_utterance") is True


TARGETS = [
    TargetSpec(
        name="valid_utterance",
        title="Valid utterance",
        description="Separates real human/chat utterances from bot commands, notices, and junk.",
        positive="valid",
        labeler=lambda row: "valid" if _is_valid(row) else "not_valid",
    ),
    TargetSpec(
        name="literal_alignment",
        title="Literal/intended alignment",
        description="Divergent means the literal words and intended stance point apart.",
        positive="divergent",
        labeler=lambda row: (
            None if not _is_valid(row) or _axis(row, "literal_intended_alignment") == "not_applicable"
            else _axis(row, "literal_intended_alignment")
        ),
    ),
    TargetSpec(
        name="magnitude_distortion",
        title="Magnitude distortion",
        description="Overstatement/hyperbole vs normal literal magnitude.",
        positive="overstated",
        labeler=lambda row: (
            None if not _is_valid(row) or _axis(row, "magnitude_distortion") == "not_applicable"
            else _axis(row, "magnitude_distortion")
        ),
    ),
    TargetSpec(
        name="play_frame",
        title="Play frame",
        description="Whether the utterance is framed as play/bit rather than plain serious talk.",
        positive="playful",
        labeler=lambda row: _play_label(row),
    ),
    TargetSpec(
        name="masking_facework",
        title="Masking / facework",
        description="Whether irony is being used as cover for status, criticism, or aggression.",
        positive="present_or_possible",
        labeler=lambda row: _masking_label(row),
    ),
    TargetSpec(
        name="hostility",
        title="Hostility",
        description="Mock/hostile energy vs low-hostility messages.",
        positive="hostile_or_mock",
        labeler=lambda row: _hostility_label(row),
    ),
    TargetSpec(
        name="shock_attention",
        title="Shock / attention bid",
        description="Shock-value attention seeking vs ordinary low-shock chat.",
        positive="present",
        labeler=lambda row: (
            None if not _is_valid(row) or _axis(row, "shock_attention") == "not_applicable"
            else _axis(row, "shock_attention")
        ),
    ),
]


def _play_label(row: dict) -> str | None:
    if not _is_valid(row):
        return None
    value = _axis(row, "play_frame")
    if value == "not_applicable":
        return None
    if value in {"playful", "masking_play"}:
        return "playful"
    if value == "low_or_none":
        return "low_or_none"
    return value


def _masking_label(row: dict) -> str | None:
    if not _is_valid(row):
        return None
    value = _axis(row, "masking_facework")
    if value == "not_applicable":
        return None
    if value in {"present", "possible"}:
        return "present_or_possible"
    if value == "absent":
        return "absent"
    return value


def _hostility_label(row: dict) -> str | None:
    if not _is_valid(row):
        return None
    value = _axis(row, "hostility")
    if value == "not_applicable":
        return None
    if value in {"present", "mild_or_mock"}:
        return "hostile_or_mock"
    if value == "low_or_none":
        return "low_or_none"
    return value


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_import_emotes():
    try:
        from utils import emote_meaning, persona_classifier
    except Exception:
        return None, None
    return emote_meaning, persona_classifier


def expand_emote_tags(text: str) -> str:
    """Append known emote meaning tags while keeping the original token."""

    emote_meaning, persona_classifier = _safe_import_emotes()
    if not emote_meaning or not persona_classifier:
        return text
    out = []
    for token in text.split():
        out.append(token)
        try:
            if persona_classifier._is_emote_token(token):
                tags = emote_meaning.meaning_tags(token, n=3)
                if tags:
                    out.append("(" + " ".join(tags) + ")")
        except Exception:
            continue
    return " ".join(out)


def row_text(row: dict, include_context: bool, emote_tags: bool) -> str:
    subject = row.get("subject") or {}
    message = str(subject.get("message") or "").strip()
    context = str(subject.get("context") or "").strip()
    if emote_tags:
        message = expand_emote_tags(message)
        context = expand_emote_tags(context)
    parts = [f"Message: {message}"]
    if include_context and context:
        parts.append("Context:\n" + context)
    return "\n\n".join(parts)


def try_embed(texts: list[str]):
    try:
        import config
        from scripts.build_persona_embeddings import embed_batch
    except Exception as exc:
        return None, f"embedding imports failed: {exc}"
    if not getattr(config, "LLM_EMBED_MODEL", ""):
        return None, "config [llm] embed_model is empty"
    try:
        vecs = []
        for i in range(0, len(texts), 32):
            vecs.extend(embed_batch(texts[i:i + 32]))
    except Exception as exc:
        return None, f"embedding endpoint failed: {exc}"
    return vecs, f"local embedding model: {config.LLM_EMBED_MODEL}"


def _stratified_folds(labels: list[str]) -> int:
    counts = Counter(labels)
    if len(counts) < 2:
        return 0
    return min(5, min(counts.values()))


def _metric_summary(y_true, y_pred, positive: str | None) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.metrics import precision_recall_fscore_support

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if positive and positive in set(y_true):
        p, r, f1, _support = precision_recall_fscore_support(
            y_true, y_pred, labels=[positive], zero_division=0
        )
        out.update({
            "positive_label": positive,
            "precision": float(p[0]),
            "recall": float(r[0]),
            "f1": float(f1[0]),
        })
    return out


def _dummy_cv_text(texts: list[str], labels: list[str], folds: int):
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=1)),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])
    return cross_val_predict(pipe, texts, labels, cv=cv)


def _dummy_cv_matrix(x, labels: list[str], folds: int):
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)
    return cross_val_predict(DummyClassifier(strategy="most_frequent"), x, labels, cv=cv)


def evaluate_tfidf(texts: list[str], labels: list[str], spec: TargetSpec) -> tuple[object, dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=7)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            max_features=5000,
            min_df=1,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", clf),
    ])
    folds = _stratified_folds(labels)
    metrics = {"counts": dict(Counter(labels)), "folds": folds}
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)
        pred = cross_val_predict(pipe, texts, labels, cv=cv)
        base = _dummy_cv_text(texts, labels, folds)
        metrics.update(_metric_summary(labels, pred, spec.positive))
        metrics["dummy_balanced_accuracy"] = _metric_summary(labels, base, spec.positive)[
            "balanced_accuracy"
        ]
    else:
        metrics["warning"] = "not enough examples per class for cross-validation"
    pipe.fit(texts, labels)
    return pipe, metrics


def evaluate_embeddings(x, labels: list[str], spec: TargetSpec) -> tuple[object, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=7)
    folds = _stratified_folds(labels)
    metrics = {"counts": dict(Counter(labels)), "folds": folds}
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)
        pred = cross_val_predict(clf, x, labels, cv=cv)
        base = _dummy_cv_matrix(x, labels, folds)
        metrics.update(_metric_summary(labels, pred, spec.positive))
        metrics["dummy_balanced_accuracy"] = _metric_summary(labels, base, spec.positive)[
            "balanced_accuracy"
        ]
    else:
        metrics["warning"] = "not enough examples per class for cross-validation"
    clf.fit(x, labels)
    return clf, metrics


def top_tfidf_features(pipe, positive: str | None, n: int = 8) -> dict:
    try:
        import numpy as np

        vectorizer = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["clf"]
        names = vectorizer.get_feature_names_out()
        classes = list(clf.classes_)
        coef = clf.coef_
        if len(classes) == 2:
            pos_index = classes.index(positive) if positive in classes else 1
            weights = coef[0]
            if pos_index == 0:
                weights = -weights
            order_pos = np.argsort(weights)[::-1][:n]
            order_neg = np.argsort(weights)[:n]
            return {
                "toward_positive": [str(names[i]) for i in order_pos],
                "toward_negative": [str(names[i]) for i in order_neg],
            }
    except Exception:
        pass
    return {}


def train(rows: list[dict], args: argparse.Namespace) -> tuple[dict, list[str]]:
    import numpy as np

    texts = [row_text(row, args.include_context, args.emote_tags) for row in rows]
    feature_mode = args.feature_mode
    feature_note = "tf-idf lexical baseline"
    embedded = None
    if feature_mode in {"auto", "embedding"}:
        embedded, note = try_embed(texts)
        if embedded is not None:
            feature_mode = "embedding"
            feature_note = note
        elif args.feature_mode == "embedding":
            raise RuntimeError(note)
        else:
            feature_mode = "tfidf"
            feature_note = "embedding unavailable; fell back to tf-idf (" + note + ")"
    if feature_mode == "tfidf":
        x = None
    else:
        x = np.asarray(embedded, dtype="float32")

    bundle = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_mode": feature_mode,
        "feature_note": feature_note,
        "include_context": bool(args.include_context),
        "emote_tags": bool(args.emote_tags),
        "n_rows": len(rows),
        "targets": {},
    }
    report = [
        "# Intent Probes Report",
        "",
        f"Created: {bundle['created_at']}",
        f"Rows: {len(rows)}",
        f"Feature mode: {feature_mode} ({feature_note})",
        f"Context included: {bool(args.include_context)}",
        f"Emote tags expanded: {bool(args.emote_tags)}",
        "",
        "This is a seed model, not a final judge. The current oracle set is only 60",
        "items, so balanced accuracy is more useful than plain accuracy and all",
        "numbers should be treated as directional.",
        "",
        "## Targets",
        "",
    ]

    for spec in TARGETS:
        indices = []
        labels = []
        for i, row in enumerate(rows):
            label = spec.labeler(row)
            if label is None:
                continue
            indices.append(i)
            labels.append(str(label))
        counts = Counter(labels)
        if len(counts) < 2:
            report.extend([
                f"### {spec.title}",
                "",
                f"Skipped: only one class present: {dict(counts)}",
                "",
            ])
            continue
        subtexts = [texts[i] for i in indices]
        if feature_mode == "tfidf":
            model, metrics = evaluate_tfidf(subtexts, labels, spec)
            features = top_tfidf_features(model, spec.positive)
        else:
            model, metrics = evaluate_embeddings(x[indices], labels, spec)
            features = {}
        bundle["targets"][spec.name] = {
            "title": spec.title,
            "description": spec.description,
            "positive": spec.positive,
            "classes": sorted(counts),
            "counts": dict(counts),
            "metrics": metrics,
            "features": features,
            "model": model,
        }
        report.extend(render_target_report(spec, metrics, features))
    return bundle, report


def _fmt_metric(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def render_target_report(spec: TargetSpec, metrics: dict, features: dict) -> list[str]:
    lines = [
        f"### {spec.title}",
        "",
        spec.description,
        "",
        f"Counts: {metrics.get('counts')}",
        f"CV folds: {metrics.get('folds')}",
    ]
    if "balanced_accuracy" in metrics:
        lines.extend([
            "Metrics:",
            f"- accuracy: {_fmt_metric(metrics.get('accuracy'))}",
            f"- balanced accuracy: {_fmt_metric(metrics.get('balanced_accuracy'))}",
            f"- dummy balanced accuracy: {_fmt_metric(metrics.get('dummy_balanced_accuracy'))}",
        ])
        if "f1" in metrics:
            lines.extend([
                f"- positive label: {metrics.get('positive_label')}",
                f"- positive precision: {_fmt_metric(metrics.get('precision'))}",
                f"- positive recall: {_fmt_metric(metrics.get('recall'))}",
                f"- positive f1: {_fmt_metric(metrics.get('f1'))}",
            ])
    else:
        lines.append(f"Warning: {metrics.get('warning')}")
    if features:
        lines.extend([
            "",
            f"Top lexical pulls toward `{spec.positive}`:",
            "- " + ", ".join(features.get("toward_positive") or []),
            "",
            "Top lexical pulls the other way:",
            "- " + ", ".join(features.get("toward_negative") or []),
        ])
    lines.append("")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument(
        "--feature-mode",
        choices=["auto", "embedding", "tfidf"],
        default="auto",
        help="auto tries the local embedding endpoint, then falls back to tf-idf",
    )
    parser.add_argument("--no-context", dest="include_context", action="store_false")
    parser.add_argument("--no-emote-tags", dest="emote_tags", action="store_false")
    parser.set_defaults(include_context=True, emote_tags=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"missing input: {args.input}")
        return 2
    rows = load_rows(args.input)
    bundle, report = train(rows, args)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with args.model_out.open("wb") as fh:
        pickle.dump(bundle, fh)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("\n".join(report).rstrip() + "\n", encoding="utf-8")
    print(f"trained {len(bundle['targets'])} probes using {bundle['feature_mode']}")
    print(f"model -> {args.model_out}")
    print(f"report -> {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
