"""Evidence-first memory/fact-bank helpers.

The first pass deliberately extracts *claims*, not verified truths. Each row
keeps the original evidence so later LLM summaries, chat commands, or human
review can stay grounded in receipts.
"""

from __future__ import annotations

import json
import hashlib
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import config
from utils import atomic_file, chat_archive, message_quality

DEFAULT_OUT = Path("data/unsynced/fact_bank.jsonl")
VERSION = 4
_LOAD_CACHE = {}
CLAIM_PATTERNS: list[tuple[str, float, re.Pattern]] = [
    ("self_identity", 0.72, re.compile(r"\b(?:i am|i'm|im)\s+(?!not\b)([^.?!]{3,120})", re.I)),
    ("self_negative_identity", 0.65, re.compile(r"\b(?:i am not|i'm not|im not)\s+([^.?!]{3,120})", re.I)),
    ("preference_positive", 0.62, re.compile(r"\bi\s+(?:like|love|enjoy|prefer)\s+([^.?!]{3,120})", re.I)),
    ("preference_negative", 0.62, re.compile(r"\bi\s+(?:hate|dislike|can't stand|cannot stand)\s+([^.?!]{3,120})", re.I)),
    ("belief", 0.55, re.compile(r"\bi\s+(?:think|believe|feel|guess|know)\s+(?:that\s+)?([^.?!]{4,150})", re.I)),
    ("possession", 0.70, re.compile(r"\bmy\s+([a-z][a-z0-9 _'-]{1,32})\s+(?:is|are|was|were)\s+([^.?!]{2,120})", re.I)),
    ("activity", 0.60, re.compile(r"\bi\s+(?:work|worked|study|studied|live|lived|play|played)\b\s*([^.?!]{2,120})", re.I)),
]
TAIL_STOP_RE = re.compile(
    r"\s+(?:lol|lmao|lmfao|xd|kekw|omegalul|imo|tbh|ngl|fr|bro|dude)\b.*$",
    re.I,
)
BAD_TAIL_RE = re.compile(
    r"^(?:not|just|so|like|the|a|an|it|this|that|you|he|she|they|we)\b$",
    re.I,
)
BAD_POSSESSION_ATTRS = {"i", "you", "he", "she", "they", "we", "anyways", "anyway"}


def _norm_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _clean_tail(text: str) -> str:
    text = _norm_spaces(text)
    text = re.split(
        r"\s+(?:but|though|because|unless|while|and i|so|when|if|which|who|where)\s+",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    text = TAIL_STOP_RE.sub("", text)
    text = re.sub(r"(?:\s+@[A-Za-z0-9_]+)+\s*$", "", text)
    return text.strip(" \t\r\n,;:-_'\"")


def _claim_key(kind: str, tail: str) -> str:
    low = tail.casefold()
    low = re.sub(r"https?://\S+|www\.\S+", " ", low)
    low = re.sub(r"[^a-z0-9']+", " ", low)
    return kind + ":" + _norm_spaces(low)


def _valid_tail(tail: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z']+", tail)
    if not (1 <= len(words) <= 24):
        return False
    if words[0].casefold() in {"just", "so", "really", "rlly", "very"}:
        return False
    if len(tail) < 3 or BAD_TAIL_RE.match(tail):
        return False
    if message_quality.repeated_token_spam(tail.split()):
        return False
    return True


def _valid_possession_value(value: str) -> bool:
    """Keep possession rows value-shaped, not predicates or arguments."""
    words = re.findall(r"[A-Za-z][A-Za-z']+", value)
    if not (1 <= len(words) <= 8):
        return False
    if re.search(
        r"\b(?:you|your|yours|he|she|they|them|their|we|our|people|someone)\b",
        value,
        re.I,
    ):
        return False
    if re.search(r"\b(?:better|worse|more|less|same)\s+than\b", value, re.I):
        return False
    return _valid_tail(value)


def extract_claims(author: str, text: str) -> list[dict]:
    """Extract candidate claims from one utterance.

    These are recall-oriented candidates. Confidence here means "pattern looked
    structurally like a useful memory row", not "this is objectively true".
    """
    raw = _norm_spaces(text)
    if (
        not raw
        or raw.startswith(getattr(config, "PREFIX", "~"))
        or raw.endswith("?")
        or message_quality.command_like(raw)
    ):
        return []
    clean = message_quality.clean_text(raw, strip_emotes=True, strip_urls=True)
    if not clean or message_quality.spam_like(clean):
        return []
    if len(clean) > 500 or len(clean.split()) > 90:
        return []

    out = []
    for kind, confidence, pattern in CLAIM_PATTERNS:
        for match in pattern.finditer(clean):
            if kind == "possession":
                attr = _clean_tail(match.group(1))
                attr_words = [w.casefold() for w in re.findall(r"[A-Za-z][A-Za-z']*", attr)]
                if (
                    not attr_words
                    or len(attr_words) > 4
                    or any(word in BAD_POSSESSION_ATTRS for word in attr_words)
                ):
                    continue
                value = _clean_tail(match.group(2))
                if not _valid_possession_value(value):
                    continue
                tail = f"{attr} = {value}"
            else:
                tail = _clean_tail(match.group(1))
            if not _valid_tail(tail):
                continue
            out.append({
                "author": chat_archive.normalize_author(author),
                "kind": kind,
                "claim": tail,
                "claim_key": _claim_key(kind, tail),
                "confidence": confidence,
                "evidence_text": raw,
            })
    return out


def _author_roster(limit: int, min_messages: int) -> list[str]:
    return chat_archive.canonical_author_roster(
        limit,
        min_messages=min_messages,
    )


def _author_utterance_rows(author: str, max_utterances: int, gap_seconds: int = 45) -> list[dict]:
    import datetime as dt

    def parse_ts(value: str):
        try:
            return dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    conn = chat_archive.connect()
    keys = chat_archive.author_keys(author)
    placeholders, params = chat_archive._in_clause(keys)
    rows = conn.execute(
        "SELECT sent_at, channel, author, content FROM messages "
        f"WHERE author IN ({placeholders}) ORDER BY sent_at DESC, id DESC LIMIT ?",
        [*params, max_utterances * 3],
    ).fetchall()
    rows = list(reversed(rows))
    out: list[dict] = []
    for sent_at, channel, row_author, content in rows:
        content = content or ""
        stripped = content.lstrip()
        if stripped.startswith(getattr(config, "PREFIX", "~")) or message_quality.command_like(content):
            continue
        canon = chat_archive.normalize_author(row_author)
        if out:
            prev = out[-1]
            t0, t1 = parse_ts(prev["last_seen"]), parse_ts(sent_at)
            if (
                gap_seconds > 0
                and prev["author"] == canon
                and prev["channel"] == chat_archive.normalize_channel(channel)
                and t0 and t1
                and (t1 - t0).total_seconds() <= gap_seconds
            ):
                prev["last_seen"] = sent_at
                prev["text"] = _norm_spaces(prev["text"] + " " + content)
                continue
        out.append({
            "author": canon,
            "channel": chat_archive.normalize_channel(channel),
            "first_seen": sent_at,
            "last_seen": sent_at,
            "text": content,
        })
    return out[-max_utterances:]


def build_fact_bank(
    authors: list[str] | None = None,
    *,
    max_authors: int = 40,
    min_messages: int = 500,
    max_utterances: int = 20000,
    evidence_limit: int = 5,
) -> list[dict]:
    if not authors:
        authors = _author_roster(max_authors, min_messages)
    else:
        authors = [chat_archive.normalize_author(a) for a in authors]

    grouped: dict[tuple[str, str], dict] = {}
    channel_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    line_authors: dict[str, set[str]] = defaultdict(set)
    for author in authors:
        # Regex claim extraction stays message-local. Merging bursts helps
        # semantic reasoning, but without boundary markers it makes a claim's
        # tail swallow the person's next two unrelated messages.
        for row in _author_utterance_rows(
            author, max_utterances=max_utterances, gap_seconds=0
        ):
            for claim in extract_claims(author, row["text"]):
                key = (claim["author"], claim["claim_key"])
                evidence_key = chat_archive.line_match_key(row["text"])
                if evidence_key:
                    line_authors[evidence_key].add(claim["author"])
                item = grouped.setdefault(key, {
                    "author": claim["author"],
                    "kind": claim["kind"],
                    "claim": claim["claim"],
                    "claim_key": claim["claim_key"],
                    "support_count": 0,
                    "confidence": claim["confidence"],
                    "extraction_confidence": claim["confidence"],
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                    "channels": [],
                    "evidence": [],
                    "_evidence_keys": set(),
                    "_support_days": set(),
                })
                item["last_seen"] = max(item["last_seen"], row["last_seen"])
                if evidence_key and evidence_key in item["_evidence_keys"]:
                    continue
                if evidence_key:
                    item["_evidence_keys"].add(evidence_key)
                item["support_count"] += 1
                item["_support_days"].add(row["first_seen"][:10])
                item["confidence"] = max(item["confidence"], claim["confidence"])
                item["extraction_confidence"] = item["confidence"]
                item["first_seen"] = min(item["first_seen"], row["first_seen"])
                channel_counts[key][row["channel"]] += 1
                if len(item["evidence"]) < evidence_limit:
                    item["evidence"].append({
                        "sent_at": row["first_seen"],
                        "channel": row["channel"],
                        "text": row["text"],
                        "clean_text": message_quality.clean_text(
                            row["text"], strip_emotes=True, strip_urls=True
                        ),
                    })

    # Positive and negative forms of the same normalized tail cannot both be
    # silently promoted. They remain receipts with an explicit contradiction.
    opposite = {
        "self_identity": "self_negative_identity",
        "self_negative_identity": "self_identity",
        "preference_positive": "preference_negative",
        "preference_negative": "preference_positive",
    }
    kinds_by_tail: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in grouped.values():
        tail = item["claim_key"].split(":", 1)[-1]
        kinds_by_tail[(item["author"], tail)].add(item["kind"])

    out = list(grouped.values())
    for item in out:
        counts = channel_counts[(item["author"], item["claim_key"])]
        item["channels"] = [name for name, _count in counts.most_common(5)]
        evidence_keys = item.pop("_evidence_keys")
        support_days = item.pop("_support_days")
        item["support_days"] = len(support_days)
        item["unique_phrasings"] = len(evidence_keys)
        item["echo_author_count"] = max(
            (len(line_authors[key]) for key in evidence_keys), default=1
        )
        tail = item["claim_key"].split(":", 1)[-1]
        kinds = kinds_by_tail[(item["author"], tail)]
        item["contradicted"] = opposite.get(item["kind"]) in kinds
        evidence_confidence = (
            0.20
            + 0.12 * min(item["support_days"], 3)
            + 0.08 * min(item["unique_phrasings"], 3)
        )
        if item["echo_author_count"] > 1:
            evidence_confidence -= 0.25
        if item["contradicted"]:
            evidence_confidence -= 0.25
        item["evidence_confidence"] = round(max(0.05, min(0.75, evidence_confidence)), 3)
        item["status"] = (
            "corroborated_claim"
            if item["support_days"] >= 2
            and item["unique_phrasings"] >= 2
            and item["echo_author_count"] == 1
            and not item["contradicted"]
            else "candidate"
        )
        item["confidence"] = round(item["confidence"], 3)
    out.sort(key=lambda x: (
        x["status"] != "corroborated_claim",
        -x["evidence_confidence"],
        -x["support_days"],
        x["author"],
        x["claim_key"],
    ))
    return out


def metadata_path(path: Path = DEFAULT_OUT) -> Path:
    return path.with_name(path.name + ".meta.json")


def metadata_current(meta: dict | None) -> bool:
    return bool(
        isinstance(meta, dict)
        and meta.get("version") == VERSION
        and meta.get("alias_signature") == chat_archive.alias_signature()
        and bool(meta.get("content_sha256"))
    )


def load_metadata(path: Path = DEFAULT_OUT) -> dict:
    try:
        return json.loads(metadata_path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def content_current(path: Path = DEFAULT_OUT, meta: dict | None = None) -> bool:
    meta = load_metadata(path) if meta is None else meta
    if not metadata_current(meta):
        return False
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == meta["content_sha256"]
    except Exception:
        return False


def write_jsonl(rows: list[dict], path: Path = DEFAULT_OUT,
                metadata: dict | None = None) -> None:
    with atomic_file.open_atomic(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    content_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {
        **(metadata or {}),
        "version": VERSION,
        "alias_signature": chat_archive.alias_signature(),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "claims": len(rows),
        "content_sha256": content_sha256,
    }
    with atomic_file.open_atomic(metadata_path(path), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=True, indent=2))
    _LOAD_CACHE.pop(str(path.resolve()), None)


def load_jsonl(path: Path = DEFAULT_OUT) -> list[dict]:
    meta_file = metadata_path(path)
    try:
        data_stat = path.stat()
        meta_stat = meta_file.stat()
    except OSError:
        return []
    stamp = (
        data_stat.st_mtime_ns,
        data_stat.st_size,
        meta_stat.st_mtime_ns,
        meta_stat.st_size,
    )
    cache_key = str(path.resolve())
    cached = _LOAD_CACHE.get(cache_key)
    if cached and cached[0] == stamp:
        return cached[1]
    before = load_metadata(path)
    if not path.exists() or not metadata_current(before):
        return []
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != before.get("content_sha256"):
            return []
        rows = [
            json.loads(line)
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        ]
    except Exception:
        return []
    if before != load_metadata(path):
        return []
    _LOAD_CACHE[cache_key] = (stamp, rows)
    return rows


def search(rows: list[dict], *, author: str | None = None, query: str = "", limit: int = 12) -> list[dict]:
    terms = [t.casefold() for t in re.findall(r"[A-Za-z0-9']+", query or "") if len(t) >= 2]
    canon = chat_archive.normalize_author(author) if author else None
    scored = []
    for row in rows:
        if canon and row.get("author") != canon:
            continue
        hay = f"{row.get('kind', '')} {row.get('claim', '')}".casefold()
        score = (
            row.get("support_days", 0) * 2
            + row.get("unique_phrasings", row.get("support_count", 0))
            + row.get("evidence_confidence", 0.0)
        )
        if terms:
            hits = sum(1 for term in terms if term in hay)
            if not hits:
                continue
            score += hits * 2
        scored.append((score, row))
    scored.sort(key=lambda item: -item[0])
    return [row for _score, row in scored[:limit]]
