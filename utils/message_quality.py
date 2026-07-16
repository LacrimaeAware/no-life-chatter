"""Shared chat-message quality filters.

Most archive-derived features fail the same way when junk is treated as human
signal: bot commands, pasted logs, translation boilerplate, pure emote spam,
keyboard smash, links, and repeated copypasta. Keep those rules centralized so
RAG, embeddings, `~iq`, oracle queues, and future training exports do not drift
into separate local hacks.
"""

from __future__ import annotations

import re
import unicodedata
import hashlib
import math
from collections import Counter

import config

WORD_RE = re.compile(r"[\w']+", re.UNICODE)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
INVISIBLE_RE = re.compile("[\u200b-\u200f\u2060\ufeff\U000e0000-\U000e007f]")
COMMAND_TOKEN_RE = re.compile(r"^[!$^?<][\w-]+", re.UNICODE)
BOT_TEXT_RE = re.compile(
    r"(<groq|<gpt|\$gpt|\$ll|\$alias|!eval|\[translation\]|"
    r"here's the translation|i was unable to translate|i'?m not allowed to translate|"
    r"your message was not sent because|automod:|"
    r"\U0001f916\s*@)",
    re.I,
)
PASTED_LOG_RE = re.compile(r"(?:^|\s)\d{1,2}:\d{2}\s+[A-Za-z0-9_]{2,25}:")
MODEL_REQUEST_RE = re.compile(
    r"(?:^|\s)(?:<gemini\d*|<groq|<gpt|[$!^](?:gpt|llm|askai)\b)",
    re.I,
)
MODEL_PROSE_RE = re.compile(
    r"\b(?:there is no publicly available|based on (?:the |available )?"
    r"(?:evidence|information|context)|as of \w+ \d{1,2},? \d{4}|"
    r"the provided (?:evidence|information|context))\b",
    re.I,
)
SELECTION_VERSION = 1
_INFORMATION_TERMS = re.compile(
    r"\b(because|therefore|although|however|unless|assuming|implies|evidence|"
    r"reason|cause|effect|tradeoff|compared|relative|means|depends|why|how|what if)\b",
    re.I,
)


def _pc():
    from utils import persona_classifier
    return persona_classifier


def clean_text(text: str, *, strip_emotes: bool = True, strip_urls: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = INVISIBLE_RE.sub("", text)
    if strip_urls:
        text = URL_RE.sub(" ", text)
    if strip_emotes:
        text = _pc().strip_emote_tokens(text)
    text = re.sub(r"\s+", " ", text).strip()
    return collapse_repeated_spans(text)


def _span_key(words: list[str]) -> tuple[str, ...]:
    return tuple(
        re.sub(r"[^\w]+", "", word.casefold()).strip("_")
        for word in words
    )


def collapse_repeated_spans(text: str, *, max_span: int = 18) -> str:
    """Collapse adjacent repeated word spans from merged chat bursts.

    This targets archive artifacts like "because X because X" or copied bot
    output repeated twice inside one merged utterance. It only collapses exact
    adjacent spans of at least two tokens, so ordinary emphasis mostly survives.
    """
    words = (text or "").split()
    if len(words) < 8:
        return text or ""
    out = []
    i = 0
    while i < len(words):
        collapsed = False
        largest = min(max_span, (len(words) - i) // 2)
        for span in range(largest, 1, -1):
            key = _span_key(words[i:i + span])
            if not all(key):
                continue
            repeats = 1
            while i + (repeats + 1) * span <= len(words):
                nxt = _span_key(words[i + repeats * span:i + (repeats + 1) * span])
                if nxt != key:
                    break
                repeats += 1
            if repeats >= 2:
                out.extend(words[i:i + span])
                i += repeats * span
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return " ".join(out)


def letter_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def symbol_count(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch)[0] in {"M", "S", "C"})


def junk_count(text: str) -> int:
    return sum(
        1 for ch in text
        if unicodedata.category(ch)[0] in {"C", "M", "P", "S"} and ch not in "'?"
    )


def low_quality_token(tok: str) -> bool:
    if len(tok) < 3 or len(tok) > 28:
        return True
    if not any(ch.isalpha() for ch in tok):
        return True
    if any(ch.isdigit() for ch in tok):
        return True
    letters = [ch for ch in tok if ch.isalpha()]
    if not letters:
        return True
    if len(tok) >= 9 and len(set(tok)) / len(tok) < 0.35:
        return True
    latinish = all(ord(ch) < 128 for ch in letters)
    if latinish and len(tok) >= 7 and not re.search(r"[aeiouy]", tok):
        return True
    return False


def tokens(text: str) -> list[str]:
    out = []
    for tok in WORD_RE.findall((text or "").lower()):
        tok = tok.strip("'_")
        if not low_quality_token(tok):
            out.append(tok)
    return out


def command_like(text: str) -> bool:
    stripped = (text or "").lstrip()
    if not stripped:
        return True
    if stripped[0] in "!$^?<":
        return True
    if BOT_TEXT_RE.search(stripped) or PASTED_LOG_RE.search(stripped):
        return True
    raw_tokens = stripped.split()
    command_count = sum(1 for tok in raw_tokens if COMMAND_TOKEN_RE.match(tok))
    return command_count >= 2 or (raw_tokens and command_count / len(raw_tokens) > 0.12)


def model_request_like(text: str) -> bool:
    """Whether a chat line invokes a model-backed public command."""
    return bool(MODEL_REQUEST_RE.search(text or ""))


def generated_response_candidate(text: str) -> bool:
    """Cheap high-recall gate before checking a line's preceding context."""
    words = WORD_RE.findall(text or "")
    return (
        len(text or "") >= 80
        and len(words) >= 12
        and (bool(MODEL_PROSE_RE.search(text or "")) or bool(re.search(r"[.!?](?:\s|$)", text or "")))
    )


def likely_pasted_prose(text: str) -> bool:
    """Conservative detector for long formal single-message quotations.

    This is used only for cognitive scoring, where crediting a pasted article,
    model answer, or copypasta as the chatter's syntax is worse than omitting an
    unusually polished line. Persona/style systems intentionally keep these
    messages because quoting things can still be part of someone's voice.
    """
    text = (text or "").strip()
    words = WORD_RE.findall(text)
    if MODEL_PROSE_RE.search(text) and len(words) >= 18:
        return True
    if len(text) < 160 or len(words) < 24:
        return False
    sentences = re.findall(r"(?:^|[.!?]\s+)[\"'([]*[A-Z]", text)
    endings = re.findall(r"[.!?](?:[\"')\]]*)?(?:\s|$)", text)
    return len(endings) >= 3 or (len(endings) >= 2 and len(sentences) >= 2)


def repeated_token_spam(words) -> bool:
    norm_words = [
        re.sub(r"[^\w]+", " ", (word or "").casefold()).strip()
        for word in words
    ]
    norm_words = [word for word in norm_words if word]
    if len(norm_words) < 6:
        return False
    counts = Counter(norm_words)
    return len(counts) <= 4 and counts.most_common(1)[0][1] / len(norm_words) >= 0.45


def repeated_phrase_spam(toks: list[str]) -> bool:
    if len(toks) < 12:
        return False
    for n in (2, 3, 4, 5, 6):
        grams = [tuple(toks[i:i + n]) for i in range(0, len(toks) - n + 1)]
        if not grams:
            continue
        count = Counter(grams).most_common(1)[0][1]
        if count >= 3 and (count * n) / len(toks) >= 0.45:
            return True
    half = len(toks) // 2
    return half >= 6 and toks[:half] == toks[half:half * 2]


def spam_like(text: str, toks: list[str] | None = None) -> bool:
    toks = toks if toks is not None else tokens(text)
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if len(compact) > 650:
        return True
    if letter_count(compact) < 8:
        return True
    if symbol_count(compact) / max(1, len(compact)) > 0.22:
        return True
    if junk_count(compact) / max(1, len(compact)) > 0.28:
        return True
    if any(len(piece) > 42 for piece in (text or "").split()):
        return True
    if len(toks) >= 8:
        counts = Counter(toks)
        if counts.most_common(1)[0][1] / len(toks) >= 0.35:
            return True
    if len(toks) >= 18:
        bigrams = list(zip(toks, toks[1:]))
        if bigrams and Counter(bigrams).most_common(1)[0][1] / len(bigrams) >= 0.22:
            return True
    if repeated_phrase_spam(toks):
        return True
    return False


def usable_for_iq(raw: str, clean: str | None = None, toks: list[str] | None = None) -> bool:
    if not raw:
        return False
    clean = clean_text(raw) if clean is None else clean
    toks = tokens(clean) if toks is None else toks
    if not clean or command_like(clean) or len(toks) < 3:
        return False
    return not spam_like(clean, toks)


def usable_for_persona_exemplar(message: str, *, max_chars: int = 240) -> bool:
    if not message:
        return False
    stripped = message.lstrip()
    if stripped.startswith(config.PREFIX) or stripped.startswith("<"):
        return False
    if URL_RE.search(message):
        return False
    if command_like(message):
        return False
    words = message.split()
    if len(words) < 2 or len(message) > max_chars:
        return False
    if repeated_token_spam(words):
        return False
    return any(re.search(r"[a-z]{3}", word) for word in words)


def usable_for_snippet_context(message: str, *, max_chars: int = 240) -> bool:
    if not message or len(message) > max_chars:
        return False
    if URL_RE.search(message):
        return False
    return not repeated_token_spam(message.split())


def semantic_text(message: str, *, min_words: int = 4, max_words: int = 60) -> str | None:
    clean = clean_text(message)
    toks = tokens(clean)
    if not usable_for_iq(message, clean, toks):
        return None
    if not (min_words <= len(clean.split()) <= max_words):
        return None
    return clean


def _semantic_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"[^\w']+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def semantic_information_score(text: str) -> float:
    """Generic retrieval value, independent of any person or trait label."""
    words = tokens(text)
    if not words:
        return 0.0
    n = len(words)
    diversity = len(set(words)) / n
    structure = min(3, len(_INFORMATION_TERMS.findall(text or ""))) / 3
    question = 1.0 if "?" in (text or "") else 0.0
    # Length saturates quickly so verbose pasted prose does not dominate.
    length = min(1.0, math.log1p(n) / math.log(25))
    return (0.45 * length) + (0.30 * diversity) + (0.20 * structure) + (0.05 * question)


def _stable_rank(label: str, key: str) -> int:
    raw = f"{label}\0{key}".encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def select_semantic_units(
    messages,
    *,
    cap: int,
    label: str,
    min_words: int = 4,
    max_words: int = 70,
    coverage_share: float = 0.80,
) -> tuple[list[dict], dict]:
    """Select a deterministic coverage lane plus a high-information tail.

    Exact normalized repeats are collapsed first. The coverage lane preserves
    an unbiased view for person/topic averages; the smaller high-information
    lane improves RAG and peak-style analyses without distorting those means.
    """
    candidates = []
    seen = set()
    source_count = 0
    for original in messages:
        source_count += 1
        cleaned = semantic_text(original, min_words=min_words, max_words=max_words)
        if not cleaned:
            continue
        key = _semantic_key(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append({
            "text": original,
            "clean": cleaned,
            "key": key,
            "information": semantic_information_score(cleaned),
        })

    cap = max(0, int(cap))
    if not cap or len(candidates) <= cap:
        selected = sorted(candidates, key=lambda row: _stable_rank(label, row["key"]))
        for row in selected:
            row["kind"] = "coverage"
    else:
        coverage_n = max(1, min(cap, int(round(cap * coverage_share))))
        by_coverage = sorted(candidates, key=lambda row: _stable_rank(label, row["key"]))
        coverage = by_coverage[:coverage_n]
        coverage_keys = {row["key"] for row in coverage}
        remaining = [row for row in candidates if row["key"] not in coverage_keys]
        remaining.sort(
            key=lambda row: (-row["information"], _stable_rank(label + ":peak", row["key"]))
        )
        peak = remaining[:cap - coverage_n]
        for row in coverage:
            row["kind"] = "coverage"
        for row in peak:
            row["kind"] = "high_signal"
        selected = coverage + peak

    meta = {
        "version": SELECTION_VERSION,
        "source_count": source_count,
        "usable_unique": len(candidates),
        "selected": len(selected),
        "coverage": sum(row["kind"] == "coverage" for row in selected),
        "high_signal": sum(row["kind"] == "high_signal" for row in selected),
    }
    return selected, meta
