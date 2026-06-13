"""Shared status checks for generated persona/archive artifacts.

This keeps commands from silently depending on stale pickles after aliases,
filters, embedding models, or semantic units change. It is intentionally
metadata-only: no raw chat lines are shown.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import config

EXPECTED_SEMANTIC_UNIT = "utterance"
PERSONA_EMBEDDINGS = Path("data/unsynced/persona_embeddings.pkl")
MESSAGE_INDEX_DIR = Path("data/unsynced/msg_index")
IQ_SCORES = Path("data/unsynced/iq_scores.pkl")
EMOTE_SEMANTICS = Path("data/unsynced/emote_semantics.pkl")
EMOTE_EMBEDDINGS = Path("data/unsynced/emote_embeddings.pkl")
CUSTOM_AXES = Path("data/unsynced/custom_axes.pkl")


def _mtime(path: Path) -> str:
    try:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "unknown"


def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _row(name: str, status: str, detail: str, commands: str) -> dict:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "commands": commands,
    }


def _classifier_status() -> dict:
    path = Path(config.CLASSIFIER_FILE)
    if not path.exists():
        return _row(
            "classifier/style profiles", "missing",
            "missing persona_classifier.pkl",
            "~whosaid, ~markers, ~like, ~twin, ~distinct emote layer",
        )
    try:
        data = _load_pickle(path)
        authors = len(data.get("authors") or [])
        profiles = len(data.get("profiles") or {})
        meta = data.get("__meta__") or data.get("meta") or {}
        classifier_meta = meta.get("classifier") if isinstance(meta, dict) else None
        style_meta = meta.get("style_profiles") if isinstance(meta, dict) else None
        status = "ok" if classifier_meta and style_meta else "warn"
        if classifier_meta or style_meta:
            note = (
                f"classifier={classifier_meta.get('built_at', '?') if classifier_meta else 'missing-meta'}; "
                f"style={style_meta.get('built_at', '?') if style_meta else 'missing-meta'}"
            )
        else:
            note = "missing build metadata (pre-status format)"
        return _row(
            "classifier/style profiles", status,
            f"{authors} classifier authors, {profiles} profiles; {note}; mtime {_mtime(path)}",
            "~whosaid, ~markers, ~like, ~twin, ~distinct emote layer",
        )
    except Exception as exc:
        return _row(
            "classifier/style profiles", "warn",
            f"could not read metadata: {exc}",
            "~whosaid, ~markers, ~like, ~twin",
        )


def _persona_embeddings_status() -> dict:
    path = PERSONA_EMBEDDINGS
    if not path.exists():
        return _row(
            "person semantic vectors", "missing",
            "missing persona_embeddings.pkl",
            "~distinct, ~traits, ~top averages, ~vibes, ~twin semantic",
        )
    try:
        data = _load_pickle(path)
        model = data.get("model") or "unknown"
        unit = data.get("unit")
        vectors = len(data.get("vectors") or {})
        notes = [f"{vectors} vectors", f"model={model}", f"mtime {_mtime(path)}"]
        status = "ok"
        if unit != EXPECTED_SEMANTIC_UNIT:
            status = "warn"
            found = unit or "missing-unit"
            notes.append(f"unit={found}; expected {EXPECTED_SEMANTIC_UNIT}")
        else:
            notes.append(f"unit={unit}")
        if getattr(config, "LLM_EMBED_MODEL", "") and model != config.LLM_EMBED_MODEL:
            status = "warn"
            notes.append(f"config embed_model={config.LLM_EMBED_MODEL}")
        return _row(
            "person semantic vectors", status,
            "; ".join(notes),
            "~distinct, ~traits, ~top averages, ~vibes, ~twin semantic",
        )
    except Exception as exc:
        return _row(
            "person semantic vectors", "warn",
            f"could not read metadata: {exc}",
            "~distinct, ~traits, ~top, ~vibes",
        )


def _message_index_status() -> dict:
    path = MESSAGE_INDEX_DIR
    if not path.is_dir():
        return _row(
            "semantic message index", "missing",
            "missing msg_index directory",
            "~why, ~top burst, semantic persona retrieval",
        )
    files = sorted(path.glob("*.npz"))
    if not files:
        return _row(
            "semantic message index", "missing",
            "0 per-author index files",
            "~why, ~top burst, semantic persona retrieval",
        )
    units = {}
    bad_reads = 0
    for file in files:
        try:
            import numpy as np
            with np.load(file, allow_pickle=True) as data:
                unit = str(data["unit"].item()) if "unit" in data.files else "missing-unit"
            units[unit] = units.get(unit, 0) + 1
        except Exception:
            bad_reads += 1
    status = "ok"
    notes = [f"{len(files)} authors", f"mtime {_mtime(max(files, key=lambda p: p.stat().st_mtime))}"]
    if units != {EXPECTED_SEMANTIC_UNIT: len(files)}:
        status = "warn"
        rendered = ", ".join(f"{unit}:{count}" for unit, count in sorted(units.items()))
        notes.append(f"units {rendered}; expected all {EXPECTED_SEMANTIC_UNIT}")
    else:
        notes.append(f"unit={EXPECTED_SEMANTIC_UNIT}")
    if bad_reads:
        status = "warn"
        notes.append(f"{bad_reads} unreadable")
    return _row(
        "semantic message index", status,
        "; ".join(notes),
        "~why, ~top burst, semantic persona retrieval",
    )


def _iq_status() -> dict:
    path = IQ_SCORES
    if not path.exists():
        return _row("text-IQ cache", "missing", "missing iq_scores.pkl", "~iq")
    try:
        data = _load_pickle(path)
        meta = data.get("__meta__") or {}
        status = "ok" if meta.get("version") == 2 else "warn"
        detail = (
            f"version={meta.get('version', 'legacy')}; authors={meta.get('authors', '?')}; "
            f"built={meta.get('built_at', _mtime(path))}; "
            f"embed={meta.get('embed_model', 'unknown')}"
        )
        return _row("text-IQ cache", status, detail, "~iq")
    except Exception as exc:
        return _row("text-IQ cache", "warn", f"could not read metadata: {exc}", "~iq")


def _simple_pickle_status(path: Path, name: str, commands: str) -> dict:
    if not path.exists():
        return _row(name, "missing", f"missing {path.name}", commands)
    try:
        data = _load_pickle(path)
        n = len(data) if hasattr(data, "__len__") else "?"
        return _row(name, "ok", f"{n} entries; mtime {_mtime(path)}", commands)
    except Exception as exc:
        return _row(name, "warn", f"could not read metadata: {exc}", commands)


def status_rows() -> list[dict]:
    return [
        _classifier_status(),
        _persona_embeddings_status(),
        _message_index_status(),
        _iq_status(),
        _simple_pickle_status(EMOTE_SEMANTICS, "emote usage semantics", "~emote, axes/emote meaning"),
        _simple_pickle_status(EMOTE_EMBEDDINGS, "emote fallback vectors", "~distinct, ~traits, ~top"),
        _simple_pickle_status(CUSTOM_AXES, "custom axes", "~top custom terms"),
    ]


def status_summary(max_rows: int = 4) -> str:
    rows = status_rows()
    counts = {key: sum(1 for row in rows if row["status"] == key)
              for key in ("ok", "warn", "missing")}
    problems = [row for row in rows if row["status"] != "ok"]
    head = (
        f"artifacts: {counts['ok']} ok, {counts['warn']} warn, "
        f"{counts['missing']} missing"
    )
    if not problems:
        return head
    bits = [
        f"{row['name']}={row['status']} ({row['detail']})"
        for row in problems[:max_rows]
    ]
    if len(problems) > max_rows:
        bits.append(f"+{len(problems) - max_rows} more")
    return head + " | " + " | ".join(bits)


def format_table() -> str:
    rows = status_rows()
    lines = []
    for row in rows:
        lines.append(
            f"{row['status'].upper():7} {row['name']}: {row['detail']} "
            f"[{row['commands']}]"
        )
    return "\n".join(lines)
