"""Shared status checks for generated persona/archive artifacts.

This keeps commands from silently depending on stale pickles after aliases,
filters, embedding models, or semantic units change. It is intentionally
metadata-only: no raw chat lines are shown.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import config
from utils import chat_archive, fact_bank, persona_iq, user_profiles

EXPECTED_SEMANTIC_UNIT = "utterance"
PERSONA_EMBEDDINGS = Path("data/unsynced/persona_embeddings.pkl")
MESSAGE_INDEX_DIR = Path("data/unsynced/msg_index")
IQ_SCORES = Path("data/unsynced/iq_scores.pkl")
EMOTE_SEMANTICS = Path("data/unsynced/emote_semantics.pkl")
EMOTE_EMBEDDINGS = Path("data/unsynced/emote_embeddings.pkl")
CUSTOM_AXES = Path("data/unsynced/custom_axes.pkl")
USER_PROFILES = Path("data/unsynced/user_profiles.json")
_STYLE_ROSTER_CACHE = None


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


def _alias_split_count(names) -> int:
    groups: dict[str, set[str]] = {}
    for raw in names or []:
        key = str(raw).casefold()
        groups.setdefault(chat_archive.normalize_author(key), set()).add(key)
    return sum(1 for members in groups.values() if len(members) > 1)


def _identity_health(signature, names) -> tuple[str, list[str]]:
    status = "ok"
    notes = []
    split = _alias_split_count(names)
    if split:
        status = "warn"
        notes.append(f"{split} split canonical identity group(s)")
    current = chat_archive.alias_signature()
    if not signature:
        status = "warn"
        notes.append("identity provenance missing")
    elif signature != current:
        status = "warn"
        notes.append("alias map changed since build")
    return status, notes


def _style_roster() -> set[str]:
    global _STYLE_ROSTER_CACHE
    try:
        path = Path(config.CLASSIFIER_FILE)
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size, chat_archive.alias_signature())
        if _STYLE_ROSTER_CACHE and _STYLE_ROSTER_CACHE[0] == stamp:
            return _STYLE_ROSTER_CACHE[1]
        data = _load_pickle(path)
        roster = {
            chat_archive.normalize_author(name)
            for name in (data.get("profiles") or {})
            if not chat_archive._is_noise_author(name)
        }
        _STYLE_ROSTER_CACHE = (stamp, roster)
        return roster
    except Exception:
        return set()


def _roster_note(names) -> str:
    expected = _style_roster()
    actual = {
        chat_archive.normalize_author(name)
        for name in names
        if not chat_archive._is_noise_author(name)
    }
    if not expected or actual == expected:
        return ""
    return (
        f"roster mismatch: {len(expected - actual)} missing, "
        f"{len(actual - expected)} extra"
    )


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
        author_names = set(data.get("authors") or [])
        profile_names = set(data.get("profiles") or {})
        authors = len(author_names)
        profiles = len(profile_names)
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
        identity_status, identity_notes = _identity_health(
            meta.get("alias_signature"),
            set(data.get("authors") or []) | set(data.get("profiles") or {}),
        )
        if identity_status == "warn":
            status = "warn"
            note += "; " + "; ".join(identity_notes)
        classifier_roster = {chat_archive.normalize_author(name) for name in author_names}
        style_roster = {chat_archive.normalize_author(name) for name in profile_names}
        if classifier_roster != style_roster:
            status = "warn"
            note += (
                f"; roster mismatch: {len(style_roster - classifier_roster)} style-only, "
                f"{len(classifier_roster - style_roster)} classifier-only"
            )
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
        status, identity_notes = _identity_health(
            data.get("alias_signature"), (data.get("vectors") or {}).keys()
        )
        notes.extend(identity_notes)
        roster_note = _roster_note((data.get("vectors") or {}).keys())
        if roster_note:
            status = "warn"
            notes.append(roster_note)
        if unit != EXPECTED_SEMANTIC_UNIT:
            status = "warn"
            found = unit or "missing-unit"
            notes.append(f"unit={found}; expected {EXPECTED_SEMANTIC_UNIT}")
        else:
            notes.append(f"unit={unit}")
        utterance_version = int(data.get("utterance_version") or 0)
        if utterance_version != chat_archive.UTTERANCE_VERSION:
            status = "warn"
            notes.append(
                f"utterance chunking v{utterance_version}; "
                f"expected v{chat_archive.UTTERANCE_VERSION}"
            )
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
    utterance_versions = {}
    identity_missing = 0
    identity_mismatch = 0
    names = []
    bad_reads = 0
    for file in files:
        try:
            import numpy as np
            with np.load(file, allow_pickle=True) as data:
                unit = str(data["unit"].item()) if "unit" in data.files else "missing-unit"
                utterance_version = (
                    int(data["utterance_version"].item())
                    if "utterance_version" in data.files else 0
                )
                sig = str(data["alias_signature"].item()) if "alias_signature" in data.files else ""
                if not sig:
                    identity_missing += 1
                elif sig != chat_archive.alias_signature():
                    identity_mismatch += 1
            units[unit] = units.get(unit, 0) + 1
            utterance_versions[utterance_version] = (
                utterance_versions.get(utterance_version, 0) + 1
            )
            names.append(file.stem)
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
    if utterance_versions != {chat_archive.UTTERANCE_VERSION: len(files)}:
        status = "warn"
        rendered = ", ".join(
            f"v{version}:{count}"
            for version, count in sorted(utterance_versions.items())
        )
        notes.append(
            f"utterance chunking {rendered}; expected v{chat_archive.UTTERANCE_VERSION}"
        )
    if bad_reads:
        status = "warn"
        notes.append(f"{bad_reads} unreadable")
    split = _alias_split_count(names)
    if split:
        status = "warn"
        notes.append(f"{split} split canonical identity group(s)")
    if identity_missing:
        status = "warn"
        notes.append(f"identity provenance missing in {identity_missing}")
    if identity_mismatch:
        status = "warn"
        notes.append(f"alias map changed for {identity_mismatch}")
    roster_note = _roster_note(names)
    if roster_note:
        status = "warn"
        notes.append(roster_note)
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
        status = "ok" if meta.get("version") == persona_iq.VERSION else "warn"
        identity_status, identity_notes = _identity_health(
            meta.get("alias_signature"), (data.get("scores") or {}).keys()
        )
        if identity_status == "warn":
            status = "warn"
        roster_note = _roster_note((data.get("scores") or {}).keys())
        if roster_note:
            status = "warn"
        detail = (
            f"version={meta.get('version', 'legacy')}; authors={meta.get('authors', '?')}; "
            f"built={meta.get('built_at', _mtime(path))}; "
            f"embed={meta.get('embed_model', 'unknown')}"
        )
        if meta.get("version") != persona_iq.VERSION:
            detail += f"; expected version={persona_iq.VERSION}"
        if meta.get("utterance_version") != chat_archive.UTTERANCE_VERSION:
            status = "warn"
            detail += (
                f"; utterance chunking v{meta.get('utterance_version', 0)}; "
                f"expected v{chat_archive.UTTERANCE_VERSION}"
            )
        quality_failures = list(meta.get("quality_failures") or [])
        embedding_note = str(meta.get("embedding_features", ""))
        judge_note = str(meta.get("llm_judge", ""))
        if embedding_note.startswith("failed:"):
            quality_failures.append(embedding_note)
        if meta.get("judge_requested") and not meta.get("judge_authors"):
            quality_failures.append(judge_note or "judge produced no results")
        if meta.get("build_quality") == "degraded" or quality_failures:
            status = "warn"
            detail += "; degraded build: " + "; ".join(dict.fromkeys(quality_failures))
        if identity_notes:
            detail += "; " + "; ".join(identity_notes)
        if roster_note:
            detail += "; " + roster_note
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


def _emote_semantics_status() -> dict:
    path = EMOTE_SEMANTICS
    if not path.exists():
        return _row(
            "emote usage semantics", "missing", "missing emote_semantics.pkl",
            "~emote, ~explain emote, ~why emote",
        )
    try:
        payload = _load_pickle(path)
        wrapped = isinstance(payload, dict) and isinstance(payload.get("emotes"), dict)
        vectors = payload.get("emotes") if wrapped else payload
        vectors = vectors if isinstance(vectors, dict) else {}
        meta = (payload.get("__meta__") or {}) if wrapped else {}
        counts = sorted(
            int((row or {}).get("n") or 0)
            for row in vectors.values()
            if isinstance(row, dict) and "vector" in row
        )
        median = counts[len(counts) // 2] if counts else 0
        target = int(meta.get("target_contexts") or 160)
        at_target = sum(value >= target for value in counts)
        status = "ok"
        notes = [
            f"{len(counts)} vectors",
            f"median contexts={median}",
            f"{at_target}/{len(counts)} at target {target}",
            f"mtime {_mtime(path)}",
        ]
        if meta.get("version") != 2:
            status = "warn"
            notes.append("legacy build metadata")
        if meta.get("model") != getattr(config, "LLM_EMBED_MODEL", ""):
            status = "warn"
            notes.append("embedding model missing or stale")
        if median < 80:
            status = "warn"
            notes.append("undersampled; top-up rebuild recommended")
        if meta and not meta.get("complete", False):
            status = "warn"
            notes.append("last rebuild checkpoint is incomplete")
        partial = path.with_name(f"{path.stem}.partial{path.suffix}")
        if partial.exists():
            status = "warn"
            notes.append("resumable top-up checkpoint is pending")
        return _row(
            "emote usage semantics", status, "; ".join(notes),
            "~emote, ~explain emote, ~why emote",
        )
    except Exception as exc:
        return _row(
            "emote usage semantics", "warn", f"could not read metadata: {exc}",
            "~emote, ~explain emote, ~why emote",
        )


def _user_profiles_status() -> dict:
    if not USER_PROFILES.exists():
        return _row(
            "verified user profiles", "missing", "missing user_profiles.json",
            "~askchat, ~persona",
        )
    try:
        data = json.loads(USER_PROFILES.read_text(encoding="utf-8"))
        profiles = data.get("profiles") or {}
        judged = data.get("judged") or {}
        meta = data.get("_meta") or {}
        populated = sum(bool(profile) for profile in profiles.values())
        status, notes = _identity_health(meta.get("alias_signature"), profiles.keys())
        if meta.get("version") != user_profiles.VERSION:
            status = "warn"
            notes.append(
                f"profile version={meta.get('version', 'missing')}; "
                f"expected {user_profiles.VERSION}"
            )
        if len(profiles) < 10:
            status = "warn"
            notes.append("roster coverage is partial")
        if profiles and populated == 0:
            status = "warn"
            notes.append("all profile records are empty; rebuild required")
        if profiles and not judged:
            status = "warn"
            notes.append("no candidate evidence was judged")
        if meta.get("build_complete") is False:
            status = "warn"
            notes.append("profile build is incomplete")
        partial_path = user_profiles._partial_path(USER_PROFILES)
        if partial_path.exists():
            status = "warn"
            notes.append("resumable profile checkpoint is pending")
        roster_note = _roster_note(profiles.keys())
        if roster_note:
            status = "warn"
            notes.append(roster_note)
        detail = (
            f"{len(profiles)} profiles ({populated} populated), "
            f"{len(judged)} judged candidates; "
            f"built={meta.get('built_at', _mtime(USER_PROFILES))}"
        )
        if notes:
            detail += "; " + "; ".join(notes)
        return _row("verified user profiles", status, detail, "~askchat, ~persona")
    except Exception as exc:
        return _row(
            "verified user profiles", "warn", f"could not read metadata: {exc}",
            "~askchat, ~persona",
        )


def _fact_bank_status() -> dict:
    path = fact_bank.DEFAULT_OUT
    if not path.exists():
        return _row("claim receipt bank", "missing", "missing fact_bank.jsonl", "~askchat")
    meta = fact_bank.load_metadata(path)
    status = "ok" if fact_bank.metadata_current(meta) else "warn"
    notes = [
        f"{meta.get('claims', '?')} claims",
        f"history cap={meta.get('max_utterances', '?')}/person",
        f"built={meta.get('built_at', _mtime(path))}",
    ]
    if meta.get("version") != fact_bank.VERSION:
        notes.append(f"version={meta.get('version', 'missing')}; expected {fact_bank.VERSION}")
    if meta.get("alias_signature") != chat_archive.alias_signature():
        notes.append("identity provenance missing or stale")
    if not fact_bank.content_current(path, meta):
        status = "warn"
        notes.append("content hash missing or mismatched")
    return _row("claim receipt bank", status, "; ".join(notes), "~askchat")


def status_rows() -> list[dict]:
    return [
        _classifier_status(),
        _persona_embeddings_status(),
        _message_index_status(),
        _iq_status(),
        _fact_bank_status(),
        _user_profiles_status(),
        _emote_semantics_status(),
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


def format_table(rows: list[dict] | None = None) -> str:
    rows = status_rows() if rows is None else rows
    lines = []
    for row in rows:
        lines.append(
            f"{row['status'].upper():7} {row['name']}: {row['detail']} "
            f"[{row['commands']}]"
        )
    return "\n".join(lines)
