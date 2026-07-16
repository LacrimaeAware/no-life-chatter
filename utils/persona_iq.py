"""Text-IQ v5: peak expressed cognition from chat logs.

This keeps the chat UX as `~iq`, but the internals are deliberately an
ensemble rather than one brittle proxy. The score is roster-relative and built
offline from:

- peak lexical/syntax/reasoning markers over merged utterances
- embedding projections onto cognitive-expression axes
- topic breadth plus niche specificity from utterance embeddings
- optional local-LLM atomic judging of top candidate utterances

The key aggregation rule is "median of the top 10%" so normal chat filler does
not drown out a person's strongest expressed-cognition moments.
"""

from __future__ import annotations

import hashlib
import functools
import json
import logging
import math
import os
import pickle
import random
import re
import time
import urllib.request
from collections import Counter
from typing import Iterable

import config
from utils import atomic_file, chat_archive, message_quality as mq, persona_classifier as pc

CACHE = os.path.join("data", "unsynced", "iq_scores.pkl")
JUDGE_CACHE = os.path.join("data", "unsynced", "iq_judgments.pkl")
VERSION = 5
JUDGE_CACHE_VERSION = 1
JUDGE_PROMPT_VERSION = 1
_judge_cache_payload = None

DEFAULT_MIN_UTTERANCES = 80
DEFAULT_MAX_UTTERANCES = 3000
DEFAULT_AUTHOR_CAP = 15000
DEFAULT_WORD_FREQ_SAMPLE = 250000

_REASONING_MARKERS = {
    "because", "therefore", "whereas", "although", "though", "unless",
    "despite", "implies", "infer", "assuming", "assume", "hypothesis",
    "hypothetically", "specifically", "technically", "basically",
    "essentially", "counterexample", "base", "rates", "causal", "cause",
    "effect", "incentive", "incentives", "structure", "tradeoff",
    "evidence", "predict", "prediction", "model", "models", "pattern",
    "patterns", "relative", "context", "mechanism", "mechanisms",
}

_CLAUSE_RE = re.compile(
    r"\b(because|although|though|whereas|therefore|however|unless|despite|"
    r"implies|assuming|specifically|technically|hypothetically|relative to|"
    r"in terms of|on the other hand|the reason|it follows|as a result)\b",
    re.I,
)

_QUESTION_RE = re.compile(
    r"\b(why|how|what if|under what|in what sense|compared to|relative to|"
    r"what causes|what explains|how would|why would)\b",
    re.I,
)

# Axis examples are intentionally casual-chat register. They measure expressed
# reasoning moves, not school essay polish.
COG_AXES: dict[str, tuple[list[str], list[str]]] = {
    "abstraction": (
        [
            "lol that guy fell over",
            "this game looks bad",
            "i liked the old thing better",
            "chat is being weird again",
            "that was funny",
        ],
        [
            "the general pattern is incentives changing the behavior",
            "this is really a status game disguised as an argument",
            "the abstract version is scarcity creating fake value",
            "that example is one instance of a broader coordination problem",
            "the same structure shows up in politics and games",
        ],
    ),
    "causal": (
        [
            "it happened because it happened",
            "people are just dumb lol",
            "no reason, thats just how it is",
            "random stuff keeps happening",
            "it is bad because bad",
        ],
        [
            "the likely cause is that the reward system pushes everyone there",
            "if the first assumption is wrong then the conclusion collapses",
            "that outcome follows from the incentives, not from one person",
            "the missing variable is who benefits when the rule changes",
            "this predicts the opposite result once the sample changes",
        ],
    ),
    "nuance": (
        [
            "its obviously always true",
            "only an idiot would disagree",
            "there is literally no other explanation",
            "everyone who says otherwise is lying",
            "case closed, dont think about it",
        ],
        [
            "it depends which part of the claim you mean",
            "that is true in one sense but false under the other assumption",
            "the evidence points that way, but the sample is probably biased",
            "i think the weaker version is right and the strong version is not",
            "there are two different claims getting mixed together here",
        ],
    ),
    "technical": (
        [
            "thing does stuff and it breaks",
            "computer bad",
            "the game is laggy because vibes",
            "numbers went up somehow",
            "idk just click it",
        ],
        [
            "the bottleneck is probably memory bandwidth rather than compute",
            "that is a precision problem, not a recall problem",
            "the parser accepts the token but the downstream schema rejects it",
            "the rate limit changes the equilibrium once requests are batched",
            "the model is overfitting surface markers instead of the target",
        ],
    ),
    "connections": (
        [
            "that is unrelated",
            "new topic anyway",
            "i dont see any pattern",
            "every example is separate",
            "just a random thing",
        ],
        [
            "this is the same failure mode as the last system, just with names changed",
            "that connects to the earlier point about incentives",
            "the analogy is not perfect but the mechanism is similar",
            "both examples are really about information asymmetry",
            "this looks different on the surface but it is the same tradeoff",
        ],
    ),
    "problem_solving": (
        [
            "just give up then",
            "nothing can be done",
            "idk someone fix it",
            "complain until it works",
            "the solution is simply be better",
        ],
        [
            "first isolate whether the failure is input, retrieval, or generation",
            "try the simpler baseline, then add the complicated part back",
            "split it into a detection step and a scoring step",
            "measure the false positives before tuning the threshold",
            "make a held-out set so we can tell if the fix actually generalizes",
        ],
    ),
    "metacognition": (
        [
            "i know im right",
            "my first guess is definitely correct",
            "dont need evidence for this",
            "i never change my mind",
            "thinking about it more is pointless",
        ],
        [
            "my confidence is low because i only saw one example",
            "i might be anchoring on the funniest case instead of the common one",
            "the thing to check is whether my explanation predicts new cases",
            "i am separating what i know from what i am guessing",
            "if the counterexample holds then my model needs to change",
        ],
    ),
}

INTERPRETABLE_FEATURES = [
    "vocab_peak",
    "syntax_peak",
    "lexical_diversity",
    "reasoning_markers",
    "question_quality",
]

EMBED_FEATURES = [
    "abstraction",
    "causal",
    "nuance",
    "technical",
    "connections",
    "problem_solving",
    "metacognition",
    "topic_breadth",
    "niche_depth",
]

GROUPS = {
    "vocab": ("vocab_peak", "lexical_diversity"),
    "syntax": ("syntax_peak",),
    "reasoning": (
        "causal", "nuance", "connections", "problem_solving", "metacognition",
        "reasoning_markers", "question_quality", "llm_reasoning",
    ),
    "abstraction": ("abstraction", "technical"),
    "breadth": ("topic_breadth",),
    "depth": ("niche_depth",),
}

GROUP_WEIGHTS = {
    "reasoning": 0.30,
    "abstraction": 0.20,
    "vocab": 0.15,
    "syntax": 0.15,
    "breadth": 0.10,
    "depth": 0.10,
}


def _read_cache():
    if not os.path.exists(CACHE):
        return None
    with open(CACHE, "rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict) and payload.get("__meta__", {}).get("version") == VERSION:
        return payload
    return None


def _cache_current(payload: dict | None) -> bool:
    meta = (payload or {}).get("__meta__") or {}
    return (
        meta.get("version") == VERSION
        and meta.get("alias_signature") == chat_archive.alias_signature()
        and meta.get("utterance_version") == chat_archive.UTTERANCE_VERSION
        and meta.get("build_quality") != "degraded"
        and not meta.get("quality_failures")
        and not str(meta.get("embedding_features", "")).startswith("failed:")
    )


def cache_problem() -> str:
    payload = _read_cache()
    if not payload:
        return "missing"
    meta = payload.get("__meta__") or {}
    if meta.get("version") != VERSION:
        return "scoring method changed"
    if meta.get("alias_signature") != chat_archive.alias_signature():
        return "identity map changed"
    if meta.get("utterance_version") != chat_archive.UTTERANCE_VERSION:
        return "utterance data changed"
    if (
        meta.get("build_quality") == "degraded"
        or meta.get("quality_failures")
        or str(meta.get("embedding_features", "")).startswith("failed:")
    ):
        return "incomplete model coverage"
    return ""


def cache_info() -> dict:
    payload = _read_cache()
    return payload.get("__meta__", {}) if payload else {}


def _stable_rng(label: str, seed: int = 17) -> random.Random:
    h = hashlib.blake2b(str(label).encode("utf-8"), digest_size=8).hexdigest()
    return random.Random(seed + int(h, 16))


def _sample(items: list, cap: int, label: str) -> list:
    if cap <= 0 or len(items) <= cap:
        return list(items)
    items = list(items)
    _stable_rng(label).shuffle(items)
    return items[:cap]


def _clean(text: str) -> str:
    return mq.clean_text(text)


def _letter_count(text: str) -> int:
    return mq.letter_count(text)


def _symbol_count(text: str) -> int:
    return mq.symbol_count(text)


def _junk_count(text: str) -> int:
    return mq.junk_count(text)


def _is_low_quality_token(tok: str) -> bool:
    return mq.low_quality_token(tok)


def _tokens(text: str) -> list[str]:
    return mq.tokens(text)


def _command_like(text: str) -> bool:
    return mq.command_like(text)


def _spam_like(text: str, toks: list[str]) -> bool:
    return mq.spam_like(text, toks)


def _usable_iq_utterance(raw: str, clean: str, toks: list[str]) -> bool:
    return mq.usable_for_iq(raw, clean, toks)


def _top_tail_median(vals: Iterable[float], pct: float = 0.90) -> float:
    vals = sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not vals:
        return 0.0
    start = max(0, int(len(vals) * pct))
    tail = vals[start:] or vals[-1:]
    mid = len(tail) // 2
    if len(tail) % 2:
        return tail[mid]
    return (tail[mid - 1] + tail[mid]) / 2.0


def _receipt_text(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit - 3].rstrip() + "..."


def _tail_receipts(scored, feature: str, count: int = 2) -> list[dict]:
    """Return auditable examples from the same top tail used by the score.

    The median item comes first because `_top_tail_median` is the aggregation
    rule; the maximum follows as a useful upper-bound example. Exact duplicate
    lines are never shown twice.
    """
    usable = []
    seen = set()
    for value, text in scored:
        if value is None or not math.isfinite(float(value)):
            continue
        key = chat_archive.line_match_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        display = mq.collapse_repeated_spans(text)
        usable.append((float(value), _receipt_text(display)))
    if not usable:
        return []
    usable.sort(key=lambda item: -item[0])
    tail = usable[:max(1, int(math.ceil(len(usable) * 0.10)))]
    picks = [tail[len(tail) // 2], tail[0]]
    out = []
    used = set()
    for value, text in picks:
        key = chat_archive.line_match_key(text)
        if key in used:
            continue
        used.add(key)
        out.append({"feature": feature, "value": round(value, 3), "text": text})
        if len(out) >= count:
            break
    return out


def _mattr(tokens: list[str], window: int = 50) -> float:
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    vals = []
    for i in range(0, len(tokens) - window + 1, max(1, window // 3)):
        chunk = tokens[i:i + window]
        vals.append(len(set(chunk)) / window)
    return sum(vals) / len(vals)


def _word_freqs(sample: int = DEFAULT_WORD_FREQ_SAMPLE):
    conn = chat_archive.connect()
    counts = Counter()
    total_rows = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
    stride = max(1, total_rows // max(1, sample))
    offset = 17 % stride
    for (content,) in conn.execute(
            "SELECT content FROM messages WHERE id % ? = ? ORDER BY id LIMIT ?",
            (stride, offset, sample)):
        counts.update(_tokens(_clean(content or "")))
    total = sum(counts.values()) or 1
    return counts, total


def _rarity_exclusions() -> set[str]:
    """Tokens that must never count as 'rare vocabulary': emote names and
    chatter usernames. Both are frequent enough to clear the min-count floor
    yet absent from normal English, so they read as maximally rare and inflate
    vocab_peak for emote-spammers and people who @ their friends a lot
    (current receipt audit's global improvement targets)."""
    out: set[str] = set()
    try:
        from utils import emote_meaning
        out.update(name.casefold() for name in emote_meaning.registry())
    except Exception:
        pass
    try:
        conn = chat_archive.connect()
        for (author,) in conn.execute(
                "SELECT author FROM messages GROUP BY author HAVING COUNT(*) >= 50"):
            out.add((author or "").casefold())
        for (author,) in conn.execute("SELECT author FROM author_ids"):
            out.add((author or "").casefold())
    except Exception:
        pass
    return out


def _rarity_fn(freqs: Counter, total: int, exclusions: set[str] | None = None):
    exclusions = _rarity_exclusions() if exclusions is None else exclusions

    def rarity(word: str) -> float | None:
        if word.casefold() in exclusions:
            return None
        if freqs.get(word, 0) < 3:
            return None
        return -math.log((freqs.get(word, 0) + 1) / total)
    return rarity


def _mean_rarity(toks: list[str], rarity) -> float | None:
    # DISTINCT tokens with a >= 3-distinct floor: "TAP-IN TAP-IN TAP-IN" is
    # one rare token repeated, not three rare words.
    vals: dict[str, float] = {}
    for t in toks:
        if t not in vals:
            v = rarity(t)
            if v is not None:
                vals[t] = v
    if len(vals) < 3:
        return None
    return sum(vals.values()) / len(vals)


def _has_reasoning_shape(text: str, toks: list[str]) -> bool:
    if _QUESTION_RE.search(text or ""):
        return True
    if _CLAUSE_RE.search(text or ""):
        return True
    marker_count = sum(1 for t in toks if t in _REASONING_MARKERS)
    return marker_count >= 2


def _row_has_corpus_signal(row: dict, rarity) -> bool:
    if _mean_rarity(row["tokens"], rarity) is not None:
        return True
    return _has_reasoning_shape(row["clean"], row["tokens"])


@functools.lru_cache(maxsize=1)
def _model_generated_reply_ids() -> frozenset[int]:
    """Immediate answer-like replies to archived model commands.

    Building this set once is much cheaper than doing a predecessor query for
    every long line while visiting each person's history.
    """
    conn = chat_archive.connect()
    requests = conn.execute(
        "SELECT id, channel, sent_at, content FROM messages WHERE "
        "content LIKE '<gemini%' OR content LIKE '<groq%' OR "
        "content LIKE '<gpt%' OR content LIKE '$gpt%' OR "
        "content LIKE '!gpt%' OR content LIKE '^gpt%'"
    )
    next_sql = (
        "SELECT id, content, sent_at, "
        "(julianday(sent_at) - julianday(?)) * 86400.0 "
        "FROM messages INDEXED BY idx_msg_channel "
        "WHERE channel = ? AND sent_at >= ? "
        "AND sent_at <= datetime(?, '+45 seconds') "
        "ORDER BY sent_at, id LIMIT 8"
    )
    reply_ids = set()
    for request_id, channel, sent_at, content in requests:
        if not mq.model_request_like(content or ""):
            continue
        candidates = conn.execute(
            next_sql,
            (sent_at, channel, sent_at, sent_at),
        ).fetchall()
        row = next((candidate for candidate in candidates if (
            candidate[2] > sent_at
            or (candidate[2] == sent_at and int(candidate[0]) > int(request_id))
        )), None)
        if not row:
            continue
        response_id, response, _response_time, delta_seconds = row
        if (
            delta_seconds is not None
            and -0.5 <= float(delta_seconds) <= 45.0
            and mq.generated_response_candidate(response or "")
        ):
            reply_ids.add(int(response_id))
    return frozenset(reply_ids)


def _preceded_by_model_request(message_id: int) -> bool:
    return int(message_id) in _model_generated_reply_ids()


def _utterance_rows(author: str, author_cap: int) -> list[dict]:
    rows = []
    seen = set()
    for record in chat_archive.utterance_records_for(author):
        parts = record.get("parts") or [(record.get("id", 0), record.get("text", ""))]
        kept_parts = []
        for message_id, text in parts:
            if mq.command_like(text) or mq.likely_pasted_prose(text):
                continue
            if (
                mq.generated_response_candidate(text)
                and _preceded_by_model_request(message_id)
            ):
                continue
            kept_parts.append(text)
        if not kept_parts:
            continue
        raw = " ".join(kept_parts)
        clean = _clean(raw)
        toks = _tokens(clean)
        if not _usable_iq_utterance(raw, clean, toks):
            continue
        key = chat_archive.line_match_key(raw)
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append({"raw": raw, "clean": clean, "tokens": toks})
    if len(rows) > author_cap:
        rows = _sample(rows, author_cap, f"{author}:utterance-cap")
    return rows


def _drop_cross_author_copies(author_rows: dict[str, list[dict]]):
    """Remove long exact copypasta shared by different canonical people.

    This deliberately ignores short/common chat lines. Alias copies have
    already collapsed into one canonical author before this pass.
    """
    owner: dict[str, str | None] = {}
    for author, rows in author_rows.items():
        for row in rows:
            key = chat_archive.line_match_key(row["raw"])
            if len(key) < 32 or len(row["tokens"]) < 6:
                continue
            previous = owner.get(key)
            if previous is None and key not in owner:
                owner[key] = author
            elif previous != author:
                owner[key] = None

    duplicate_keys = {key for key, value in owner.items() if value is None}
    if not duplicate_keys:
        return author_rows, 0
    removed = 0
    filtered = {}
    for author, rows in author_rows.items():
        kept = []
        for row in rows:
            key = chat_archive.line_match_key(row["raw"])
            if key in duplicate_keys:
                removed += 1
            else:
                kept.append(row)
        filtered[author] = kept
    return filtered, removed


def _non_english(text: str) -> bool:
    """Confidently non-English utterances are excluded from the vocab-rarity
    signal: every Spanish/Portuguese word is corpus-rare, so code-switching
    chatters scored as erudite (audit 2026-07-09 — a top roster chatter's
    rarity drivers were Spanish insults and Portuguese meme lines)."""
    try:
        from utils.language_detect import detect_language
    except Exception:
        return False
    try:
        code, conf = detect_language(text)
    except Exception:
        return False
    return bool(code and code != "EN" and conf >= 0.85)


def _interpretable_scored(rows: list[dict], rarity):
    scored = {name: [] for name in (
        "vocab_peak", "syntax_peak", "reasoning_markers", "question_quality"
    )}
    all_tokens = []
    for row in rows:
        toks = row["tokens"]
        text = row["clean"]
        all_tokens.extend(toks)
        # quoting a bot command mid-message ("~whosaid xyz") is chat plumbing,
        # not vocabulary; command NAMES are corpus-rare and inflate rarity
        quotes_command = getattr(config, "PREFIX", "~") in row["raw"]
        r = None if (quotes_command or _non_english(text)) else _mean_rarity(toks, rarity)
        if r is not None:
            scored["vocab_peak"].append((r, row["raw"]))
        n = len(toks)
        clauses = {
            match.group(0).casefold()
            for match in _CLAUSE_RE.finditer(text)
        }
        # DISTINCT markers: a doubled line ("though he's close though he's
        # close") must not read as a reasoning chain
        marker_count = len({t for t in toks if t in _REASONING_MARKERS})
        marker_density = marker_count / math.sqrt(max(1, n))
        if n < 8:
            syntax = 0.0
        else:
            length = max(0.0, min(1.0, (min(n, 40) - 8) / 32.0))
            clause_structure = min(1.0, len(clauses) / 2.0)
            marker_structure = min(1.0, marker_count / 3.0)
            syntax = (
                0.40 * length
                + 0.45 * clause_structure
                + 0.15 * marker_structure
            )
        scored["syntax_peak"].append((syntax, row["raw"]))
        scored["reasoning_markers"].append((
            marker_density + len(clauses), row["raw"]
        ))
        q = 0.0
        if "?" in text:
            q += 0.35
        if _QUESTION_RE.search(text):
            q += 1.0
        if q and n >= 7:
            q += min(0.6, math.log1p(n) / 6)
        scored["question_quality"].append((q, row["raw"]))
    return scored, all_tokens


def _interpretable_features(rows: list[dict], rarity) -> dict[str, float]:
    if not rows:
        return {name: 0.0 for name in INTERPRETABLE_FEATURES}
    scored, all_tokens = _interpretable_scored(rows, rarity)
    return {
        "vocab_peak": _top_tail_median(v for v, _text in scored["vocab_peak"]),
        "syntax_peak": _top_tail_median(v for v, _text in scored["syntax_peak"]),
        "lexical_diversity": _mattr(all_tokens),
        "reasoning_markers": _top_tail_median(
            v for v, _text in scored["reasoning_markers"]
        ),
        "question_quality": _top_tail_median(
            v for v, _text in scored["question_quality"]
        ),
    }


def _interpretable_receipts(rows: list[dict], rarity) -> dict[str, list[dict]]:
    scored, _all_tokens = _interpretable_scored(rows, rarity)
    out = {}
    for feature, values in scored.items():
        if feature in {"reasoning_markers", "question_quality"}:
            values = [(value, text) for value, text in values if value > 0]
        if values:
            out[feature] = _tail_receipts(values, feature)
    return out


def _embed_batch(texts: list[str]):
    if not texts:
        return []
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    body = json.dumps({"model": config.LLM_EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(
        base + "/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return [d["embedding"] for d in data["data"]]


def _normalize_matrix(vecs):
    import numpy as np

    mat = np.asarray(vecs, dtype="float32")
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat


def _axis_vectors():
    import numpy as np

    labels = []
    spans = {}
    for axis, (neg, pos) in COG_AXES.items():
        start = len(labels)
        labels.extend(neg + pos)
        spans[axis] = (start, start + len(neg), start + len(neg) + len(pos))
    embs = _normalize_matrix(_embed_batch(labels))
    out = {}
    for axis, (start, split, end) in spans.items():
        neg = embs[start:split].mean(axis=0)
        pos = embs[split:end].mean(axis=0)
        vec = pos - neg
        out[axis] = vec / (np.linalg.norm(vec) + 1e-9)
    return out


def _indexed_embedding_rows(
    author: str,
    max_utterances: int,
    allowed_keys: set[str] | None = None,
):
    """Return coverage and peak matrices from the shared semantic index."""
    import numpy as np

    path = os.path.join("data", "unsynced", "msg_index", f"{author}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as data:
        if "model" not in data.files or "alias_signature" not in data.files:
            return None
        if str(data["model"].item()) != getattr(config, "LLM_EMBED_MODEL", ""):
            return None
        if str(data["alias_signature"].item()) != chat_archive.alias_signature():
            return None
        if (
            "utterance_version" not in data.files
            or int(data["utterance_version"].item()) != chat_archive.UTTERANCE_VERSION
        ):
            return None
        vectors = data["vectors"].astype("float32")
        texts = [str(text) for text in data["texts"]]
        kinds = (
            [str(kind) for kind in data["kinds"]]
            if "kinds" in data.files else ["coverage"] * len(texts)
        )
    if allowed_keys is not None:
        keep = [
            i for i, text in enumerate(texts)
            if chat_archive.line_match_key(text) in allowed_keys
        ]
        if not keep:
            return None
        vectors = vectors[keep]
        texts = [texts[i] for i in keep]
        kinds = [kinds[i] for i in keep]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    coverage_idx = [i for i, kind in enumerate(kinds) if kind == "coverage"] or list(range(len(texts)))
    axis_idx = list(range(len(texts)))
    if max_utterances > 0:
        coverage_idx = _sample(coverage_idx, max_utterances, f"{author}:iq-index-coverage")
        high = [i for i, kind in enumerate(kinds) if kind == "high_signal"]
        room = max(0, max_utterances - len(high))
        high_set = set(high)
        base = [i for i in coverage_idx if i not in high_set]
        axis_idx = high[:max_utterances] + _sample(base, room, f"{author}:iq-index-axis")
    return (
        vectors[coverage_idx],
        vectors[axis_idx],
        [texts[i] for i in coverage_idx],
        [texts[i] for i in axis_idx],
    )


def _embedding_features(author_rows: dict[str, list[dict]], max_utterances: int):
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    axis = _axis_vectors()
    coverage_matrices = {}
    axis_matrices = {}
    coverage_texts = {}
    axis_texts = {}
    all_vecs = []
    reused = embedded = 0
    for author, rows in author_rows.items():
        allowed_keys = {chat_archive.line_match_key(row["raw"]) for row in rows}
        indexed = _indexed_embedding_rows(author, max_utterances, allowed_keys)
        if indexed is not None and len(indexed[0]) >= 20:
            coverage_mat, axis_mat, coverage_lines, axis_lines = indexed
            reused += len(axis_mat)
        else:
            eligible = [
                row for row in rows
                if 4 <= len(row["tokens"]) <= 80 and len(row["clean"]) <= 700
            ]
            eligible = _sample(eligible, max_utterances, f"{author}:iq-embed")
            if len(eligible) < 20:
                continue
            vecs = []
            for i in range(0, len(eligible), 64):
                vecs.extend(_embed_batch([row["clean"] for row in eligible[i:i + 64]]))
            coverage_mat = axis_mat = _normalize_matrix(vecs)
            coverage_lines = axis_lines = [row["raw"] for row in eligible]
            embedded += len(axis_mat)
        coverage_matrices[author] = coverage_mat
        axis_matrices[author] = axis_mat
        coverage_texts[author] = coverage_lines
        axis_texts[author] = axis_lines
        all_vecs.append(coverage_mat)

    if not all_vecs:
        return {}, {}, "no utterances embedded or indexed"

    all_mat = np.vstack(all_vecs)
    global_mean = all_mat.mean(axis=0)
    specificity = np.linalg.norm(all_mat - global_mean, axis=1)
    k = min(24, max(6, int(math.sqrt(len(all_mat) / 80)) + 6))
    km = MiniBatchKMeans(n_clusters=k, random_state=23, batch_size=2048, n_init=5)
    labels = km.fit_predict(all_mat)

    out = {}
    receipts = {}
    offset = 0
    for author, coverage_mat in coverage_matrices.items():
        n = len(coverage_mat)
        lab = labels[offset:offset + n]
        spec = specificity[offset:offset + n]
        offset += n
        counts = np.bincount(lab, minlength=k).astype("float64")
        share = counts / max(1.0, counts.sum())
        nz = share[share > 0]
        feats = {
            "topic_breadth": float(-(nz * np.log(nz)).sum() / math.log(k)),
            "niche_depth": float(_top_tail_median(spec)),
        }
        author_receipts = {
            "niche_depth": _tail_receipts(
                zip(spec.tolist(), coverage_texts[author]), "niche_depth"
            )
        }
        cluster_receipts = []
        for cluster in np.argsort(counts)[::-1]:
            indices = np.flatnonzero(lab == cluster)
            if not len(indices):
                continue
            center = km.cluster_centers_[cluster]
            distances = np.linalg.norm(coverage_mat[indices] - center, axis=1)
            chosen = int(indices[int(np.argmin(distances))])
            cluster_receipts.append({
                "feature": "topic_cluster",
                "value": round(float(share[cluster]), 3),
                "text": _receipt_text(coverage_texts[author][chosen]),
            })
            if len(cluster_receipts) >= 3:
                break
        author_receipts["topic_breadth"] = cluster_receipts
        axis_mat = axis_matrices[author]
        axis_lines = axis_texts[author]
        for name, vec in axis.items():
            projections = axis_mat @ vec
            feats[name] = float(_top_tail_median(projections))
            author_receipts[name] = _tail_receipts(
                zip(projections.tolist(), axis_lines), name
            )
        out[author] = feats
        receipts[author] = author_receipts
    return (
        out,
        receipts,
        f"reused {reused} indexed utterances; embedded {embedded} fallback utterances",
    )


def _chat_sync(prompt: str, max_tokens: int = 500, model: str | None = None) -> str:
    body = json.dumps({
        "model": model or getattr(config, "LLM_MODEL", ""),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": False,
    }).encode()
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def require_model_dependencies(*, embeddings: bool, judge: bool) -> None:
    """Fail quickly before an expensive build when a requested backend is absent."""
    failures = []
    if embeddings:
        if not getattr(config, "LLM_EMBED_MODEL", ""):
            failures.append("no embedding model is configured")
        else:
            try:
                vectors = _embed_batch(["text IQ dependency check"])
                if len(vectors) != 1 or not vectors[0]:
                    raise RuntimeError("empty embedding response")
            except Exception as exc:
                failures.append(f"embedding backend unavailable: {exc}")
    if judge:
        if not getattr(config, "LLM_MODEL", ""):
            failures.append("no judge model is configured")
        else:
            try:
                if not _chat_sync("Reply with exactly OK.", max_tokens=3).strip():
                    raise RuntimeError("empty chat response")
            except Exception as exc:
                failures.append(f"judge backend unavailable: {exc}")
    if failures:
        raise RuntimeError("; ".join(failures))


def _quality_failures(
    *,
    embedding_requested: bool,
    embedding_authors: int,
    judge_requested: bool,
    judged_authors: int,
    total_authors: int,
    minimum_coverage: float = 0.90,
) -> list[str]:
    required = max(1, math.ceil(total_authors * minimum_coverage))
    failures = []
    if embedding_requested and embedding_authors < required:
        failures.append(
            f"embeddings covered {embedding_authors}/{total_authors} authors; "
            f"need at least {required}"
        )
    if judge_requested and judged_authors < required:
        failures.append(
            f"judge covered {judged_authors}/{total_authors} authors; "
            f"need at least {required}"
        )
    return failures


def _judge_cache() -> dict:
    global _judge_cache_payload
    if _judge_cache_payload is not None:
        return _judge_cache_payload
    payload = None
    try:
        with open(JUDGE_CACHE, "rb") as handle:
            candidate = pickle.load(handle)
        if candidate.get("__meta__", {}).get("version") == JUDGE_CACHE_VERSION:
            payload = candidate
    except (OSError, EOFError, pickle.PickleError, AttributeError):
        pass
    _judge_cache_payload = payload or {
        "__meta__": {"version": JUDGE_CACHE_VERSION},
        "rows": {},
    }
    return _judge_cache_payload


def _judge_evidence_key(author: str, examples: list[str]) -> str:
    payload = {
        "version": JUDGE_PROMPT_VERSION,
        "model": getattr(config, "LLM_MODEL", ""),
        "alias_signature": chat_archive.alias_signature(),
        "utterance_version": chat_archive.UTTERANCE_VERSION,
        "author": author,
        "examples": examples,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _save_judge_result(
    author: str,
    evidence_key: str,
    features: dict[str, float],
    receipts: dict[str, list[dict]],
) -> None:
    payload = _judge_cache()
    payload["rows"][author] = {
        "evidence_key": evidence_key,
        "features": features,
        "receipts": receipts,
    }
    payload["__meta__"].update({
        "version": JUDGE_CACHE_VERSION,
        "model": getattr(config, "LLM_MODEL", ""),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with atomic_file.open_atomic(JUDGE_CACHE, "wb") as handle:
        pickle.dump(payload, handle)


def _upper_half_median(values: list[float]) -> float | None:
    values = sorted((float(value) for value in values), reverse=True)
    if not values:
        return None
    upper = values[:max(2, math.ceil(len(values) / 2))]
    upper.sort()
    mid = len(upper) // 2
    if len(upper) % 2:
        return upper[mid]
    return (upper[mid - 1] + upper[mid]) / 2.0


def _judge_author(
    author: str,
    rows: list[dict],
    semantic: dict[str, list[dict]] | None = None,
    items: int = 8,
) -> tuple[dict[str, float], dict[str, list[dict]], bool] | None:
    if not getattr(config, "LLM_MODEL", ""):
        return None
    scored = []
    for row in rows:
        toks = row["tokens"]
        text = row["clean"]
        if len(toks) < 6 or len(text) > 500:
            continue
        markers = len({token for token in toks if token in _REASONING_MARKERS})
        clauses = len(_CLAUSE_RE.findall(text))
        questions = int(bool(_QUESTION_RE.search(text)))
        score = 0.8 * markers + 1.1 * clauses + 0.8 * questions
        score += 0.08 * min(30, len(toks))
        scored.append((score, text))
    scored.sort(key=lambda kv: -kv[0])
    direct_n = max(1, items // 2)
    examples = [text for _score, text in scored[:direct_n]]
    semantic_order = (
        "causal", "nuance", "connections", "problem_solving", "metacognition",
        "abstraction", "technical",
    )
    for feature in semantic_order:
        for receipt in (semantic or {}).get(feature, []):
            text = _clean(receipt.get("text", ""))
            if text:
                examples.append(text)
            if len(examples) >= items:
                break
        if len(examples) >= items:
            break
    deduped = []
    seen = set()
    for text in examples:
        key = chat_archive.line_match_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
        if len(deduped) >= items:
            break
    examples = deduped
    if not examples:
        return None
    evidence_key = _judge_evidence_key(author, examples)
    cached = (_judge_cache().get("rows") or {}).get(author) or {}
    if cached.get("evidence_key") == evidence_key:
        return cached.get("features") or {}, cached.get("receipts") or {}, True
    prompt = (
        "Score each Twitch-chat utterance independently. Reward an explicit reasoning "
        "move (premises, mechanism, comparison, inference, counterexample, or calibrated "
        "uncertainty), not merely a technical topic, a question, jargon, length, or correct "
        "facts. Abstraction requires connecting an example to a general structure. Set "
        "authored_chat low when the line looks quoted, copied, or model-generated. Ignore "
        "agreement, politics, spelling, and politeness. Use integers 0-4. Reply ONLY JSON: "
        "{\"items\":[{\"reasoning\":0,\"abstraction\":0,\"precision\":0,"
        "\"nuance\":0,\"authored_chat\":0}, ...]}.\n\n"
        + "\n".join(f"{i + 1}. {text}" for i, text in enumerate(examples))
    )
    raw = _chat_sync(prompt)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    data = json.loads(re.sub(r",\s*([\]}])", r"\1", match.group(0)))
    reasoning_scored = []
    abstraction_scored = []
    for index, item in enumerate(data.get("items", [])):
        if index >= len(examples):
            break
        if not isinstance(item, dict):
            continue
        def bounded(name):
            return max(0.0, min(4.0, float(item.get(name, 0))))
        authored = bounded("authored_chat") / 4.0
        reasoning = (
            bounded("reasoning") + bounded("precision") + bounded("nuance")
        ) / 12.0
        abstraction = bounded("abstraction") / 4.0
        reasoning_scored.append((reasoning * authored, examples[index]))
        abstraction_scored.append((abstraction * authored, examples[index]))
    reasoning_value = _upper_half_median([value for value, _text in reasoning_scored])
    abstraction_value = _upper_half_median([value for value, _text in abstraction_scored])
    if reasoning_value is None or abstraction_value is None:
        return None
    features = {
        "llm_reasoning": reasoning_value,
        "llm_abstraction": abstraction_value,
    }
    receipts = {
        "llm_reasoning": _tail_receipts(reasoning_scored, "llm_reasoning"),
        "llm_abstraction": _tail_receipts(abstraction_scored, "llm_abstraction"),
    }
    _save_judge_result(author, evidence_key, features, receipts)
    return features, receipts, False


def _canonical_roster(names: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for name in names:
        canon = chat_archive.normalize_author(name)
        if not canon or canon in seen or chat_archive._is_noise_author(canon):
            continue
        seen.add(canon)
        out.append(canon)
    return out


def _roster(max_authors: int | None = None) -> list[str]:
    try:
        profiles = (pc.load().get("profiles") or {})
        names = _canonical_roster(sorted(profiles))
    except Exception:
        names = []
    if not names:
        msg_dir = os.path.join("data", "unsynced", "msg_index")
        if os.path.isdir(msg_dir):
            names = _canonical_roster(sorted(f[:-4] for f in os.listdir(msg_dir) if f.endswith(".npz")))
    if max_authors:
        names = names[:max_authors]
    return names


def _z_table(raw: dict[str, dict[str, float]], features: list[str]):
    import numpy as np

    names = list(raw)
    z = {a: {} for a in names}
    stats = {}
    for feat in features:
        vals = np.array([raw[a].get(feat, 0.0) for a in names], dtype="float64")
        mu = float(vals.mean())
        sd = float(vals.std()) or 1.0
        stats[feat] = (mu, sd)
        vals = np.clip((vals - mu) / sd, -3.5, 3.5)
        for i, a in enumerate(names):
            z[a][feat] = float(vals[i])
    return z, stats


def _group_scores(z_feats: dict[str, float]) -> dict[str, float]:
    groups = {}
    for group, feats in GROUPS.items():
        if group == "reasoning":
            semantic = [
                z_feats[name]
                for name in (
                    "causal", "nuance", "connections", "problem_solving",
                    "metacognition",
                )
                if name in z_feats
            ]
            parts = []
            if semantic:
                parts.append((0.35, sum(semantic) / len(semantic)))
            if "reasoning_markers" in z_feats:
                parts.append((0.40, z_feats["reasoning_markers"]))
            if "question_quality" in z_feats:
                parts.append((0.25, z_feats["question_quality"]))
            if parts:
                total = sum(weight for weight, _value in parts)
                value = sum(weight * value for weight, value in parts) / total
                if "llm_reasoning" in z_feats:
                    value = 0.55 * value + 0.45 * z_feats["llm_reasoning"]
                groups[group] = value
            continue
        if group == "abstraction":
            vals = [z_feats[f] for f in ("abstraction", "technical") if f in z_feats]
            if vals:
                value = sum(vals) / len(vals)
                if "llm_abstraction" in z_feats:
                    value = 0.60 * value + 0.40 * z_feats["llm_abstraction"]
                groups[group] = value
            continue
        vals = [z_feats[f] for f in feats if f in z_feats]
        if vals:
            groups[group] = sum(vals) / len(vals)
    return groups


RECEIPT_FEATURES = {
    "reasoning": (
        "llm_reasoning", "causal", "nuance", "connections", "problem_solving", "metacognition",
        "reasoning_markers",
    ),
    "abstraction": ("llm_abstraction", "abstraction", "technical"),
    "vocab": ("vocab_peak",),
    "syntax": ("syntax_peak", "reasoning_markers"),
    "breadth": ("topic_breadth",),
    "depth": ("niche_depth",),
}


def _group_receipts(
    lexical: dict[str, list[dict]],
    semantic: dict[str, list[dict]],
    limit: int = 3,
) -> dict[str, list[dict]]:
    available = {**lexical, **semantic}
    out = {}
    for group, features in RECEIPT_FEATURES.items():
        chosen = []
        seen = set()
        # Round-robin keeps one axis from consuming every receipt slot.
        for rank in range(3):
            for feature in features:
                rows = available.get(feature) or []
                if rank >= len(rows):
                    continue
                row = dict(rows[rank])
                key = chat_archive.line_match_key(row.get("text", ""))
                if not key or key in seen:
                    continue
                seen.add(key)
                chosen.append(row)
                if len(chosen) >= limit:
                    break
            if len(chosen) >= limit:
                break
        if chosen:
            out[group] = chosen
    return out


def _weighted(groups: dict[str, float]) -> float:
    parts = [(GROUP_WEIGHTS[g], v) for g, v in groups.items() if g in GROUP_WEIGHTS]
    total = sum(w for w, _v in parts) or 1.0
    return sum(w * v for w, v in parts) / total


def _percentiles(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    n = len(ordered)
    if n <= 1:
        return {a: 50 for a, _ in ordered}
    return {a: int(round(100 * i / (n - 1))) for i, (a, _v) in enumerate(ordered)}


def _half_delta(
    split_raw: tuple[dict[str, float], dict[str, float]],
    stats: dict[str, tuple[float, float]],
) -> float | None:
    left, right = split_raw
    vals = []
    for feat in INTERPRETABLE_FEATURES:
        if feat not in left or feat not in right or feat not in stats:
            continue
        mu, sd = stats[feat]
        vals.append(abs(((left[feat] - mu) / sd) - ((right[feat] - mu) / sd)))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _confidence(n_utterances: int, split_delta: float | None, used_embeddings: bool) -> str:
    if split_delta is None:
        return "low"
    if n_utterances >= 700 and split_delta <= 0.45 and used_embeddings:
        return "high"
    if n_utterances >= 200 and split_delta <= 0.90:
        return "medium"
    return "low"


def compute_all(
    force: bool = False,
    *,
    use_embeddings: bool = True,
    use_llm: bool = False,
    max_utterances: int = DEFAULT_MAX_UTTERANCES,
    min_utterances: int = DEFAULT_MIN_UTTERANCES,
    author_cap: int = DEFAULT_AUTHOR_CAP,
    sample_word_freq: int = DEFAULT_WORD_FREQ_SAMPLE,
    max_authors: int | None = None,
    judge_items: int = 8,
    require_complete: bool = False,
) -> dict[str, dict]:
    """Build or load the roster-relative text-IQ scores."""
    if not force:
        payload = _read_cache()
        if _cache_current(payload):
            return payload["scores"]

    if require_complete:
        require_model_dependencies(embeddings=use_embeddings, judge=use_llm)

    started = time.time()
    authors = _roster(max_authors=max_authors)
    freqs, total = _word_freqs(sample_word_freq)
    rarity = _rarity_fn(freqs, total)

    candidate_rows: dict[str, list[dict]] = {}
    for author in authors:
        rows = _utterance_rows(author, author_cap=author_cap)
        rows = [row for row in rows if _row_has_corpus_signal(row, rarity)]
        if len(rows) >= min_utterances:
            candidate_rows[author] = rows
    candidate_rows, copied_utterances_removed = _drop_cross_author_copies(candidate_rows)

    author_rows: dict[str, list[dict]] = {}
    scoring_rows: dict[str, list[dict]] = {}
    raw: dict[str, dict[str, float]] = {}
    split_raw: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    lexical_receipts: dict[str, dict[str, list[dict]]] = {}
    for author, rows in candidate_rows.items():
        if len(rows) < min_utterances:
            continue
        author_rows[author] = rows
        selected = (
            _sample(rows, max_utterances, f"{author}:iq-fixed-evidence")
            if max_utterances > 0 and len(rows) > max_utterances
            else list(rows)
        )
        scoring_rows[author] = selected
        raw[author] = _interpretable_features(selected, rarity)
        lexical_receipts[author] = _interpretable_receipts(selected, rarity)
        mid = max(1, len(selected) // 2)
        split_raw[author] = (
            _interpretable_features(selected[:mid], rarity),
            _interpretable_features(selected[mid:], rarity),
        )

    embed_note = "disabled"
    used_embeddings = False
    embedding_authors = 0
    semantic_receipts: dict[str, dict[str, list[dict]]] = {}
    if use_embeddings and raw and getattr(config, "LLM_EMBED_MODEL", ""):
        try:
            embed_raw, semantic_receipts, embed_note = _embedding_features(
                author_rows, max_utterances
            )
            for author, feats in embed_raw.items():
                if author in raw:
                    raw[author].update(feats)
            used_embeddings = bool(embed_raw)
            embedding_authors = len(embed_raw)
        except Exception as exc:
            embed_note = f"failed: {exc}"
            logging.warning("IQ embedding features failed: %s", exc)

    llm_note = "disabled"
    judged = 0
    judge_cache_hits = 0
    if use_llm and raw:
        for index, (author, rows) in enumerate(scoring_rows.items(), 1):
            try:
                result = _judge_author(
                    author,
                    rows,
                    semantic_receipts.get(author),
                    items=judge_items,
                )
            except Exception as exc:
                logging.warning("IQ judge failed for %s: %s", author, exc)
                result = None
            if result is not None:
                features, receipts, cached = result
                raw[author].update(features)
                lexical_receipts[author].update(receipts)
                judged += 1
                judge_cache_hits += int(cached)
            print(
                f"  IQ judge {index}/{len(scoring_rows)} {author}: "
                f"{'failed' if result is None else ('cache' if result[2] else 'model')}",
                flush=True,
            )
        llm_note = (
            f"judged {judged}/{len(raw)} authors (reasoning + abstraction; "
            f"{judge_cache_hits} cache hits)"
        )

    if not raw:
        return {}

    quality_failures = _quality_failures(
        embedding_requested=use_embeddings,
        embedding_authors=embedding_authors,
        judge_requested=use_llm,
        judged_authors=judged,
        total_authors=len(raw),
    )
    if require_complete and quality_failures:
        raise RuntimeError(
            "refusing to publish incomplete IQ artifact: " + "; ".join(quality_failures)
        )

    features = sorted({feat for feats in raw.values() for feat in feats})
    z_feats, stats = _z_table(raw, features)
    group_raw = {author: _group_scores(z_feats[author]) for author in raw}
    composite_raw = {author: _weighted(groups) for author, groups in group_raw.items()}

    import numpy as np

    vals = np.array(list(composite_raw.values()), dtype="float64")
    mu = float(vals.mean())
    sd = float(vals.std()) or 1.0
    percentiles = _percentiles(composite_raw)

    out = {}
    for author, groups in group_raw.items():
        final_z = (composite_raw[author] - mu) / sd
        delta = _half_delta(split_raw[author], stats)
        n = len(scoring_rows[author])
        row = {
            "iq": int(max(62, min(158, round(100 + 15 * final_z)))),
            "z": round(float(final_z), 2),
            "percentile": percentiles[author],
            "confidence": _confidence(n, delta, used_embeddings),
            "n_utterances": n,
            "available_utterances": len(author_rows[author]),
            "split_delta": None if delta is None else round(float(delta), 2),
            "components": {k: round(float(v), 2) for k, v in sorted(groups.items())},
            "receipts": _group_receipts(
                lexical_receipts.get(author, {}),
                semantic_receipts.get(author, {}),
            ),
        }
        # Back-compat and compact command output.
        for key in ("vocab", "syntax", "reasoning", "abstraction", "breadth", "depth"):
            if key in groups:
                row[key] = round(float(groups[key]), 2)
        out[author] = row

    payload = {
        "__meta__": {
            "version": VERSION,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "authors": len(out),
            "features": features,
            "embed_model": getattr(config, "LLM_EMBED_MODEL", ""),
            "embedding_features": embed_note,
            "embedding_requested": use_embeddings,
            "embedding_authors": embedding_authors,
            "llm_judge": llm_note,
            "judge_requested": use_llm,
            "judge_authors": judged,
            "judge_cache_hits": judge_cache_hits,
            "build_quality": "degraded" if quality_failures else "complete",
            "quality_failures": quality_failures,
            "alias_signature": chat_archive.alias_signature(),
            "utterance_version": chat_archive.UTTERANCE_VERSION,
            "word_frequency_sample": "deterministic-id-stride",
            "max_utterances": max_utterances,
            "fixed_scoring_budget": max_utterances,
            "min_utterances": min_utterances,
            "author_cap": author_cap,
            "cross_author_copies_removed": copied_utterances_removed,
            "elapsed_sec": round(time.time() - started, 1),
        },
        "scores": out,
    }
    with atomic_file.open_atomic(CACHE, "wb") as fh:
        pickle.dump(payload, fh)
    return out


def score(author: str):
    payload = _read_cache()
    if not _cache_current(payload):
        return None
    data = _canonicalized_scores(payload.get("scores") or {})
    return data.get(chat_archive.normalize_author(author))


def leaderboard(n: int = 5, reverse: bool = False):
    payload = _read_cache()
    if not _cache_current(payload):
        return []
    data = _canonicalized_scores(payload.get("scores") or {})
    ranked = sorted(data.items(), key=lambda kv: kv[1]["iq"], reverse=not reverse)
    return ranked[:n]


def _canonicalized_scores(data: dict) -> dict:
    """Collapse stale cached score labels through the current alias map.

    IQ scores are cached on disk, so after adding an alias the cache can still
    contain both `oldname` and `currentname`. Until the next rebuild, display the
    current canonical name once. If the current name already has a row, prefer
    it; otherwise keep the row with the most utterances.
    """
    grouped: dict[str, tuple[str, dict]] = {}
    for author, row in data.items():
        canon = chat_archive.normalize_author(author)
        if chat_archive._is_noise_author(canon):
            continue
        prev = grouped.get(canon)
        if prev is None:
            grouped[canon] = (author, row)
            continue
        prev_author, prev_row = prev
        prefer_new = (
            (author == canon and prev_author != canon)
            or (
                author != canon
                and prev_author != canon
                and row.get("n_utterances", 0) > prev_row.get("n_utterances", 0)
            )
        )
        if prefer_new:
            grouped[canon] = (author, row)
    return {canon: row for canon, (_source, row) in grouped.items()}
