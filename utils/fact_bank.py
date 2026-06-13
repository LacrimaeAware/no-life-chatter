"""Evidence-first memory/fact-bank helpers.

The first pass deliberately extracts *claims*, not verified truths. Each row
keeps the original evidence so later LLM summaries, chat commands, or human
review can stay grounded in receipts.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import config
from utils import chat_archive, message_quality

DEFAULT_OUT = Path("data/unsynced/fact_bank.jsonl")
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
        r"\s+(?:but|though|because|unless|while|and i)\s+",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    text = TAIL_STOP_RE.sub("", text)
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
                tail = _clean_tail(f"{attr} = {match.group(2)}")
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
    excluded = {chat_archive.normalize_author(u) for u in getattr(config, "EXCLUDE_USERS", set())}
    conn = chat_archive.connect()
    counts = Counter()
    for author, count in conn.execute(
        "SELECT author, COUNT(*) FROM messages GROUP BY author"
    ).fetchall():
        canon = chat_archive.normalize_author(author)
        if canon not in excluded:
            counts[canon] += int(count)
    return [a for a, c in counts.most_common() if c >= min_messages][:limit]


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
        canon = chat_archive.normalize_author(row_author)
        if out:
            prev = out[-1]
            t0, t1 = parse_ts(prev["last_seen"]), parse_ts(sent_at)
            if (
                prev["author"] == canon
                and prev["channel"] == chat_archive.normalize_channel(channel)
                and t0 and t1
                and (t1 - t0).total_seconds() <= gap_seconds
            ):
                prev["last_seen"] = sent_at
                prev["text"] = _norm_spaces(prev["text"] + " " + (content or ""))
                continue
        out.append({
            "author": canon,
            "channel": chat_archive.normalize_channel(channel),
            "first_seen": sent_at,
            "last_seen": sent_at,
            "text": content or "",
        })
    return out[-max_utterances:]


def build_fact_bank(
    authors: list[str] | None = None,
    *,
    max_authors: int = 40,
    min_messages: int = 500,
    max_utterances: int = 2000,
    evidence_limit: int = 5,
) -> list[dict]:
    if not authors:
        authors = _author_roster(max_authors, min_messages)
    else:
        authors = [chat_archive.normalize_author(a) for a in authors]

    grouped: dict[tuple[str, str], dict] = {}
    channel_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for author in authors:
        for row in _author_utterance_rows(author, max_utterances=max_utterances):
            for claim in extract_claims(author, row["text"]):
                key = (claim["author"], claim["claim_key"])
                item = grouped.setdefault(key, {
                    "author": claim["author"],
                    "kind": claim["kind"],
                    "claim": claim["claim"],
                    "claim_key": claim["claim_key"],
                    "support_count": 0,
                    "confidence": claim["confidence"],
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                    "channels": [],
                    "evidence": [],
                })
                item["support_count"] += 1
                item["confidence"] = min(0.95, max(item["confidence"], claim["confidence"])
                                         + min(0.20, 0.03 * (item["support_count"] - 1)))
                item["first_seen"] = min(item["first_seen"], row["first_seen"])
                item["last_seen"] = max(item["last_seen"], row["last_seen"])
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

    out = list(grouped.values())
    for item in out:
        counts = channel_counts[(item["author"], item["claim_key"])]
        item["channels"] = [name for name, _count in counts.most_common(5)]
        item["confidence"] = round(item["confidence"], 3)
    out.sort(key=lambda x: (-x["support_count"], -x["confidence"], x["author"], x["claim_key"]))
    return out


def write_jsonl(rows: list[dict], path: Path = DEFAULT_OUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path = DEFAULT_OUT) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def search(rows: list[dict], *, author: str | None = None, query: str = "", limit: int = 12) -> list[dict]:
    terms = [t.casefold() for t in re.findall(r"[A-Za-z0-9']+", query or "") if len(t) >= 2]
    canon = chat_archive.normalize_author(author) if author else None
    scored = []
    for row in rows:
        if canon and row.get("author") != canon:
            continue
        hay = f"{row.get('kind', '')} {row.get('claim', '')}".casefold()
        score = row.get("support_count", 0) + row.get("confidence", 0.0)
        if terms:
            hits = sum(1 for term in terms if term in hay)
            if not hits:
                continue
            score += hits * 2
        scored.append((score, row))
    scored.sort(key=lambda item: -item[0])
    return [row for _score, row in scored[:limit]]
