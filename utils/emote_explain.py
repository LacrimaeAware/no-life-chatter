"""Structured emote-meaning explanations for chat commands.

The core path is artifact-only: registry metadata, blended usage/tag vectors,
nearest emotes, neighbor tag consensus, and cached built-in axis vectors. It
does not call the local LLM/embedding server during a live chat command.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from collections import Counter
from typing import Any

import config
from utils import emote_meaning

AXIS_CACHE = os.path.join("data", "unsynced", "eval", "axis_vecs_cache.json")
CUSTOM_AXES = os.path.join("data", "unsynced", "custom_axes.pkl")


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: max(0, n - 3)] + "..."


def _norm(vec):
    import numpy as np

    v = np.asarray(vec, dtype="float32")
    return v / (float((v * v).sum()) ** 0.5 + 1e-9)


def _lowdin(rows):
    import numpy as np

    A = np.vstack([_norm(row) for row in rows]).astype("float64")
    G = A @ A.T
    w, U = np.linalg.eigh(G)
    w = np.clip(w, 1e-9, None)
    inv_sqrt = U @ np.diag(w ** -0.5) @ U.T
    R = inv_sqrt @ A
    R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    return R.astype("float32")


def _cached_axis_rows(include_custom: bool = False) -> tuple[list[dict[str, Any]], str | None]:
    """Built-in scoring axes from the eval cache, optionally plus custom axes.

    Built-ins use the same Lowdin decorrelation as the live trait scorer. Custom
    axes are stored after creation, so they can be compared without asking the
    embedding server to rebuild anything.
    """
    rows: list[dict[str, Any]] = []
    note = None
    try:
        from utils.persona_traits import AXES

        if os.path.exists(AXIS_CACHE):
            cached = json.load(open(AXIS_CACHE, encoding="utf-8"))
            model = cached.get("model") or ""
            expected = getattr(config, "LLM_EMBED_MODEL", "") or model
            raw = cached.get("axes") or {}
            if model == expected and set(raw) >= set(AXES):
                names = [name for name in AXES if name in raw]
                ortho = _lowdin([raw[name] for name in names])
                for name, vec in zip(names, ortho):
                    neg, pos, _neg_s, _pos_s = AXES[name]
                    rows.append({
                        "name": name,
                        "vector": vec,
                        "positive": pos,
                        "negative": neg,
                        "kind": "builtin",
                    })
            elif raw:
                note = "axis cache model mismatch"
        else:
            note = "no axis cache"
    except Exception as exc:
        note = f"axis cache unavailable: {exc}"

    if include_custom and os.path.exists(CUSTOM_AXES):
        try:
            custom = pickle.load(open(CUSTOM_AXES, "rb"))
            for name, data in (custom or {}).items():
                vec = data.get("vector")
                if vec is None:
                    continue
                rows.append({
                    "name": name,
                    "vector": _norm(vec),
                    "positive": data.get("pos_label") or name,
                    "negative": data.get("neg_label") or f"non-{name}",
                    "kind": "custom",
                })
        except Exception:
            pass

    return rows, note


def _axis_neighbors(vec, n: int = 4,
                    include_custom: bool = False) -> tuple[list[dict[str, Any]], str | None]:
    rows, note = _cached_axis_rows(include_custom=include_custom)
    if vec is None or not rows:
        return [], note
    target = _norm(vec)
    scored = []
    for row in rows:
        sim = float(target @ _norm(row["vector"]))
        scored.append({
            "name": row["name"],
            "score": sim,
            "label": row["positive"] if sim >= 0 else row["negative"],
            "kind": row["kind"],
        })
    scored.sort(key=lambda item: -abs(item["score"]))
    return scored[: max(0, n)], note


def _registry_tags(info: dict[str, Any] | None) -> list[str]:
    return [str(tag).lower() for tag in ((info or {}).get("tags") or []) if str(tag).strip()]


def _neighbor_rows(neighbors: list[tuple[str, float]]) -> list[dict[str, Any]]:
    reg = emote_meaning.registry()
    rows = []
    for name, score in neighbors:
        rows.append({
            "name": name,
            "score": float(score),
            "tags": _registry_tags(reg.get(name)),
        })
    return rows


def _neighbor_tag_scores(neighbors: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in neighbors[:12]:
        for tag in row.get("tags") or []:
            counts[tag] += max(0.0, float(row.get("score") or 0.0))
    return [
        {"tag": tag, "score": float(score)}
        for tag, score in counts.most_common(limit)
    ]


def analyze(token: str, *, neighbors: int = 8, axes: int = 4,
            include_custom_axes: bool = False) -> dict[str, Any]:
    query = (token or "").strip().lstrip("@")
    name, info = emote_meaning.lookup(query)
    sem_key = emote_meaning.semantic_key(query)
    canonical = name or sem_key or query
    vec = emote_meaning.vector(query)
    near = emote_meaning.nearest_emotes(query, n=max(neighbors, 1)) if vec is not None else []
    neighbor_rows = _neighbor_rows(near)
    axis_rows, axis_note = _axis_neighbors(vec, n=axes, include_custom=include_custom_axes)

    signals = []
    if info:
        signals.append("registry")
    if vec is not None:
        signals.append("usage-vector")
    if neighbor_rows:
        signals.append("neighbors")
    if axis_rows:
        signals.append("axis-cache")
    usage_n = emote_meaning.usage_count(query)
    if usage_n >= 20 and neighbor_rows:
        confidence = "strong"
    elif vec is not None or info:
        confidence = "mixed"
    else:
        confidence = "thin"

    return {
        "query": query,
        "name": canonical,
        "registry_name": name,
        "semantic_key": sem_key,
        "registry": info or {},
        "registry_tags": _registry_tags(info),
        "has_vector": vec is not None,
        "usage_n": usage_n,
        "neighbors": neighbor_rows,
        "neighbor_tags": _neighbor_tag_scores(neighbor_rows),
        "axes": axis_rows,
        "axis_note": axis_note,
        "signals": signals,
        "confidence": confidence,
    }


def _meaning_phrase(report: dict[str, Any]) -> str:
    tags = [row["tag"] for row in report.get("neighbor_tags", [])[:3]]
    if not tags:
        tags = report.get("registry_tags", [])[:3]
    if tags:
        return "/".join(tags)
    neighbors = [row["name"] for row in report.get("neighbors", [])[:3]]
    if neighbors:
        return "near " + ", ".join(neighbors)
    if report.get("registry"):
        return "registry-only"
    return "no learned meaning yet"


def _basis(report: dict[str, Any]) -> str:
    bits = []
    usage_n = int(report.get("usage_n") or 0)
    if usage_n:
        bits.append(f"usage n={usage_n}")
    if report.get("registry"):
        origin = report["registry"].get("origin")
        channel = report["registry"].get("channel")
        label = "registry"
        if origin:
            label += f" {origin}"
        if channel:
            label += f"#{channel}"
        bits.append(label)
    if report.get("axis_note") and not report.get("axes"):
        bits.append(report["axis_note"])
    return ", ".join(bits) or "no registry/vector"


def _join_scored(rows: list[dict[str, Any]], key: str, *, n: int, scores: bool) -> str:
    parts = []
    for row in rows[:n]:
        label = str(row[key])
        if scores:
            label += f" {float(row.get('score') or 0.0):+.2f}"
        parts.append(label)
    return ", ".join(parts)


def _fit(prefix: str, segments: list[str], max_chars: int) -> str:
    msg = prefix.rstrip()
    for segment in segments:
        if not segment:
            continue
        sep = " " if msg.endswith(":") else " | "
        candidate = msg + (sep if msg else "") + segment
        if len(candidate) <= max_chars:
            msg = candidate
    return _clip(msg, max_chars)


def _segment(label: str, rows: list[dict[str, Any]], key: str, *,
             n: int, scores: bool) -> str:
    joined = _join_scored(rows, key, n=n, scores=scores)
    return f"{label} {joined}" if joined else ""


def format_chat(report: dict[str, Any], *, detail: bool = False,
                raw: bool = False, max_chars: int = 470) -> str:
    name = report.get("name") or report.get("query") or "emote"
    if not report.get("registry") and not report.get("has_vector"):
        return _clip(
            f"{name}: no registry entry or usage vector yet; probably rare, dead, "
            "or not recognized as an emote.",
            max_chars,
        )

    if raw:
        segments = [
            f"basis {report.get('confidence')} ({_basis(report)})",
            _segment("tags", report.get("neighbor_tags", []), "tag", n=5, scores=True),
            _segment("neighbors", report.get("neighbors", []), "name", n=5, scores=True),
            _segment("axes", report.get("axes", []), "name", n=4, scores=True),
        ]
        return _fit(f"{name} vector report", segments, max_chars)

    segments = [
        f"guess: {_meaning_phrase(report)}",
        f"basis: {report.get('confidence')} ({_basis(report)})",
    ]
    if report.get("registry_tags"):
        segments.append("own tags: " + ", ".join(report["registry_tags"][:4]))
    if report.get("neighbors"):
        segments.append("used like: " + _join_scored(report["neighbors"], "name", n=5, scores=detail))
    if detail and report.get("axes"):
        axis_bits = []
        for row in report["axes"][:3]:
            axis_bits.append(f"{row['name']} {row['score']:+.2f} ({row['label']})")
        segments.append("axes: " + ", ".join(axis_bits))
    return _fit(f"{name}:", segments, max_chars)
