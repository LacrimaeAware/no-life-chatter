"""Structured emote-meaning explanations for chat commands."""

from __future__ import annotations

import json
import hashlib
import os
import pickle
import re
from collections import Counter
from typing import Any

import config
from utils import chat_archive, emote_meaning, message_quality

AXIS_CACHE = os.path.join("data", "unsynced", "eval", "axis_vecs_cache.json")
CUSTOM_AXES = os.path.join("data", "unsynced", "custom_axes.pkl")
CO_EMOTE_STOP = {
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BUT", "BY", "CHAT",
    "DO", "FOR", "FROM", "GET", "GOT", "HAD", "HAS", "HAVE", "HE",
    "HER", "HIM", "HIS", "I", "IN", "IS", "IT", "ITS", "JUST", "LIVE",
    "MADE", "MAKE", "ME", "MY", "NOT", "OF", "ON", "OR", "SO", "THAT",
    "THE", "THEIR", "THEM", "THEY", "THIS", "TO", "WAS", "WATCH", "WE",
    "WHAT", "WHEN", "WITH", "YOU", "YOUR",
}


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
    seen = set()
    for name, score in neighbors:
        key = str(name).casefold()
        if key in seen:
            continue
        seen.add(key)
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


def _strip_target(text: str, token: str) -> str:
    return re.sub(rf"(?<!\w){re.escape(token)}(?!\w)", " ", text or "", flags=re.IGNORECASE)


def _clean_context_text(text: str, token: str) -> str:
    stripped = _strip_target(text, token)
    return message_quality.clean_text(stripped, strip_emotes=True, strip_urls=True)


def _co_emotes(text: str, token: str, registry: dict[str, Any]) -> list[str]:
    out = []
    target = token.lower()
    for raw in (text or "").split():
        cleaned = raw.strip(".,!?;:\"'()[]{}<>")
        if not cleaned or cleaned.lower() == target:
            continue
        if cleaned.upper() in CO_EMOTE_STOP:
            continue
        if cleaned.isupper() and len(cleaned) <= 3:
            continue
        if cleaned in registry:
            out.append(cleaned)
    return out


def _archive_rows(token: str, *, limit: int = 90) -> tuple[int, list[dict[str, Any]]]:
    """Stable, mixed recent/random-ish archive rows containing the emote."""
    if not token:
        return 0, []
    conn = chat_archive.connect()
    cmd_sql, cmd_params = chat_archive._command_filter(False, alias="m")
    q = chat_archive._fts_phrase(token)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM messages_fts f CROSS JOIN messages m ON m.id = f.rowid "
            f"WHERE f.messages_fts MATCH ? {cmd_sql}",
            [q, *cmd_params],
        ).fetchone()[0]
        recent = conn.execute(
            "SELECT m.id, m.sent_at, m.channel, m.author, m.content FROM messages_fts f "
            "CROSS JOIN messages m ON m.id = f.rowid "
            f"WHERE f.messages_fts MATCH ? {cmd_sql}"
            "ORDER BY m.sent_at DESC LIMIT ?",
            [q, *cmd_params, max(20, limit // 3)],
        ).fetchall()
        salt = int(hashlib.sha1(token.lower().encode("utf-8")).hexdigest()[:8], 16) % 1000003 or 17
        spread = conn.execute(
            "SELECT m.id, m.sent_at, m.channel, m.author, m.content FROM messages_fts f "
            "CROSS JOIN messages m ON m.id = f.rowid "
            f"WHERE f.messages_fts MATCH ? {cmd_sql}"
            f"ORDER BY ((m.id * {(salt * 2) + 1}) % 2147483647) LIMIT ?",
            [q, *cmd_params, limit * 2],
        ).fetchall()
    except Exception:
        like = "%" + token.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_") + "%"
        total = conn.execute(
            "SELECT COUNT(*) FROM messages m WHERE m.content LIKE ? ESCAPE '\\' "
            f"{cmd_sql}",
            [like, *cmd_params],
        ).fetchone()[0]
        recent = conn.execute(
            "SELECT m.id, m.sent_at, m.channel, m.author, m.content FROM messages m "
            "WHERE m.content LIKE ? ESCAPE '\\' "
            f"{cmd_sql}ORDER BY m.sent_at DESC LIMIT ?",
            [like, *cmd_params, max(20, limit // 3)],
        ).fetchall()
        spread = conn.execute(
            "SELECT m.id, m.sent_at, m.channel, m.author, m.content FROM messages m "
            "WHERE m.content LIKE ? ESCAPE '\\' "
            f"{cmd_sql}ORDER BY m.id LIMIT ?",
            [like, *cmd_params, limit * 2],
        ).fetchall()

    seen = set()
    out = []
    token_re = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)", re.IGNORECASE)
    for row_id, sent_at, channel, author, content in list(recent) + list(spread):
        if not token_re.search(content or ""):
            continue
        key = (
            chat_archive.normalize_author(author),
            chat_archive.normalize_channel(channel),
            chat_archive.line_match_key(content),
            (sent_at or "")[:16],
        )
        if not key[2] or key in seen:
            continue
        seen.add(key)
        out.append({
            "id": row_id,
            "sent_at": sent_at,
            "channel": channel,
            "author": chat_archive.normalize_author(author),
            "text": content,
        })
        if len(out) >= limit:
            break
    return int(total or 0), out


def _context_snippet(row: dict[str, Any], token: str, *, max_chars: int = 260) -> str:
    try:
        window = chat_archive.context_window(row["id"], row["channel"], before=2, after=1)
    except Exception:
        window = []
    if not window:
        return _clip(f"{row['author']}: {row['text']}", max_chars)
    pieces = []
    for msg_id, author, content in window:
        marker = ">" if msg_id == row["id"] else "-"
        pieces.append(f"{marker}{chat_archive.normalize_author(author)}: {_clip(content, 90)}")
    return _clip(" / ".join(pieces), max_chars)


def _archive_evidence(token: str, *, limit: int = 90, examples: int = 10) -> dict[str, Any]:
    registry = emote_meaning.registry()
    try:
        total, rows = _archive_rows(token, limit=limit)
    except Exception:
        return {"hits": 0, "sampled": 0, "terms": [], "co_emotes": [], "examples": []}
    term_counts: Counter[str] = Counter()
    emote_counts: Counter[str] = Counter()
    snippets = []
    for row in rows:
        clean = _clean_context_text(row["text"], token)
        if clean:
            for term in chat_archive.query_terms(clean, max_terms=10, exclude_terms={token.lower()}):
                term_counts[term] += 1
        for emote in _co_emotes(row["text"], token, registry):
            emote_counts[emote] += 1
        if len(snippets) < examples:
            snippets.append(_context_snippet(row, token))
    return {
        "hits": total,
        "sampled": len(rows),
        "terms": [{"term": term, "count": count} for term, count in term_counts.most_common(12)],
        "co_emotes": [{"emote": emote, "count": count} for emote, count in emote_counts.most_common(12)],
        "examples": snippets,
    }


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
    archive = _archive_evidence(canonical)

    signals = []
    if info:
        signals.append("registry")
    if vec is not None:
        signals.append("usage-vector")
    if archive.get("sampled"):
        signals.append("archive-context")
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
        "archive": archive,
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
    return " ".join(parts)


def emote_tokens(report: dict[str, Any], *, n: int = 8) -> list[str]:
    tokens = [report.get("name") or report.get("query") or ""]
    tokens.extend(str(row.get("name") or "") for row in report.get("neighbors", [])[:n])
    out = []
    seen = set()
    for token in tokens:
        key = str(token).casefold()
        if token and key not in seen:
            seen.add(key)
            out.append(token)
    return out


def _fit(prefix: str, segments: list[str], max_chars: int) -> str:
    msg = prefix.rstrip()
    for segment in segments:
        if not segment:
            continue
        sep = " "
        candidate = msg + (sep if msg else "") + segment
        if len(candidate) <= max_chars:
            msg = candidate
    return _clip(msg, max_chars)


def _segment(label: str, rows: list[dict[str, Any]], key: str, *,
             n: int, scores: bool) -> str:
    joined = _join_scored(rows, key, n=n, scores=scores)
    return f"{label} {joined}" if joined else ""


def _meaning_words(report: dict[str, Any]) -> list[str]:
    words = [row["tag"] for row in report.get("neighbor_tags", [])[:3]]
    if not words:
        words = report.get("registry_tags", [])[:3]
    return [word.replace("_", " ") for word in words if word]


def _archive_terms(report: dict[str, Any], n: int = 2) -> list[str]:
    archive = report.get("archive") or {}
    sampled = int(archive.get("sampled") or 0)
    min_count = max(3, round(sampled * 0.08)) if sampled else 1
    terms = []
    for row in archive.get("terms", []):
        term = str(row.get("term") or "").strip()
        count = int(row.get("count") or min_count)
        if count < min_count:
            continue
        if term:
            terms.append(term)
        if len(terms) >= n:
            break
    return terms


def _similar_clause(report: dict[str, Any], n: int = 4) -> str:
    neighbors = []
    seen = set()
    for row in report.get("neighbors", []):
        name = str(row.get("name") or "").strip("`*'\".,:;!?()[]{}<>")
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        neighbors.append(name)
        if len(neighbors) >= n:
            break
    return "Similar emotes " + " ".join(neighbors) if neighbors else ""


def _strip_similar_tail(text: str) -> str:
    return re.sub(r"\s*(?:[.;]\s*)?Similar emotes\b.*$", "", text or "", flags=re.IGNORECASE).strip()


def _append_similar_clause(report: dict[str, Any], text: str, *,
                           max_chars: int = 470, n: int = 4) -> str:
    text = _strip_similar_tail(text)
    similar = _similar_clause(report, n=n)
    if not similar:
        return _clip(text, max_chars)
    if len(text) + 2 + len(similar) <= max_chars:
        return f"{text}. {similar}"
    return _clip(text, max_chars)


def format_sentence(report: dict[str, Any], *, max_chars: int = 470) -> str:
    """Deterministic natural-language fallback, with emote tokens left bare."""
    name = report.get("name") or report.get("query") or "emote"
    if not report.get("registry") and not report.get("has_vector") and not report.get("archive", {}).get("sampled"):
        return _clip(
            f"{name} is not in the emote registry and has no learned usage vector yet.",
            max_chars,
        )

    words = _meaning_words(report)
    similar = _similar_clause(report)
    if words:
        sentence = f"{name} is used to mean " + " / ".join(words)
        if similar:
            sentence += f". {similar}"
    elif report.get("neighbors"):
        neighbor_text = " ".join(row["name"] for row in report["neighbors"][:3])
        sentence = (
            f"{name} seems to be used in the same chat-reaction cluster as "
            f"{neighbor_text}"
        )
    elif _archive_terms(report):
        sentence = (
            f"{name} appears in archive contexts around "
            + " / ".join(_archive_terms(report))
        )
    else:
        sentence = f"{name} is known from the registry, but this bot has little learned usage for it yet."
    return _clip(sentence, max_chars)


def raw_report(report: dict[str, Any], *, max_chars: int = 470) -> str:
    name = report.get("name") or report.get("query") or "emote"
    segments = []
    archive = report.get("archive") or {}
    if archive.get("hits"):
        segments.append(f"archive_hits {archive['hits']}")
    if archive.get("sampled"):
        segments.append(f"archive_sample {archive['sampled']}")
    if report.get("usage_n"):
        segments.append(f"vector_sample_contexts {int(report['usage_n'])}")
    if report.get("registry"):
        origin = report["registry"].get("origin")
        channel = report["registry"].get("channel")
        reg = "registry"
        if origin:
            reg += f" {origin}"
        if channel:
            reg += f" channel {channel}"
        segments.append(reg)
    segments.extend([
        _segment("tags", report.get("neighbor_tags", []), "tag", n=5, scores=True),
        _segment("context_terms", archive.get("terms", []), "term", n=5, scores=False),
        _segment("co_emotes", archive.get("co_emotes", []), "emote", n=5, scores=False),
        _segment("neighbors", report.get("neighbors", []), "name", n=5, scores=True),
        _segment("axes", report.get("axes", []), "name", n=4, scores=True),
    ])
    return _fit(f"{name} vector report", segments, max_chars)


def synthesis_messages(report: dict[str, Any]) -> list[dict[str, str]]:
    """Prompt for local LLM synthesis from the structured evidence."""
    name = report.get("name") or report.get("query") or "emote"
    neighbor_names = [row["name"] for row in report.get("neighbors", [])[:6]]
    neighbor_tags = [row["tag"] for row in report.get("neighbor_tags", [])[:6]]
    registry = report.get("registry") or {}
    archive = report.get("archive") or {}
    evidence = {
        "emote": name,
        "registry_origin": registry.get("origin"),
        "registry_channel": registry.get("channel"),
        "registry_tags": report.get("registry_tags", [])[:6],
        "neighbor_tag_consensus": neighbor_tags,
        "similar_emotes_by_usage": neighbor_names,
        "confidence": report.get("confidence"),
        "has_usage_vector": bool(report.get("has_vector")),
        "archive_total_hits": archive.get("hits") or 0,
        "archive_sampled_hits": archive.get("sampled") or 0,
        "weak_archive_context_terms": archive.get("terms", [])[:10],
        "weak_archive_co_emotes": archive.get("co_emotes", [])[:10],
        "representative_archive_contexts": archive.get("examples", [])[:8],
        "usage_vector_sampled_context_lines": report.get("usage_n") or 0,
    }
    system = (
        "You explain Twitch emote meaning from evidence. Write one compact chat "
        "message under 280 characters. Usually format it as '<EMOTE> is used to mean ...'. "
        "Do not include a Similar emotes section; the caller appends that. Keep emote "
        "tokens case-sensitive and standalone: no colon, comma, quote, period, "
        "or parenthesis attached to an emote token. Do not mention vectors, "
        "basis, confidence, n, or sampled contexts. Do not repeat the same "
        "emote token twice. Similar emotes are evidence, not the answer. "
        "Treat vector neighbors and registry tags as "
        "the primary meaning signal. Use representative archive contexts to "
        "understand the situations where it appears, but do not define the "
        "emote from topical nouns like movie, game, streamer names, or links "
        "unless that theme is strongly repeated. Context terms and co-emotes "
        "are weak aggregate hints and may contain noise. Do not use outside "
        "Twitch knowledge as a shortcut. If has_usage_vector is false and "
        "registry/neighbor tags are empty, do not infer emotional valence; say "
        "what it seems to be used with/around instead. If the evidence is weak "
        "or mixed, say what it seems to be used with/around instead of "
        "pretending certainty."
    )
    user = (
        "Evidence JSON:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"Explain what {name} is used to mean."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_synthesis(report: dict[str, Any], answer: str | None,
                    *, max_chars: int = 470) -> str:
    name = report.get("name") or report.get("query") or "emote"
    text = re.sub(r"\s+", " ", answer or "").strip().strip(" '`*\"")
    if not text:
        return ""
    text = text.replace("`", "").replace("*", "")
    for token in emote_tokens(report):
        text = re.sub(
            rf"<\s*{re.escape(token)}\s*>",
            token,
            text,
            flags=re.IGNORECASE,
        )
    text = text.replace("<", "").replace(">", "")
    text = _strip_similar_tail(text)
    text = re.sub(rf"^{re.escape(name)}\s*[:;,.-]\s*", f"{name} ", text)
    starts = re.match(r"^([^\s]+)", text)
    first_key = (starts.group(1).strip("`*'\".,:;!?()[]{}<>").casefold()
                 if starts else "")
    if first_key != name.casefold():
        text = f"{name} {text}"
    for _ in range(3):
        words = text.split()
        keys = [w.strip("`*'\".,:;!?()[]{}<>").casefold() for w in words[:2]]
        if len(keys) >= 2 and keys[0] == keys[1] == name.casefold():
            new = " ".join([name] + words[2:])
        else:
            new = re.sub(
                rf"^{re.escape(name)}\s+{re.escape(name)}(?=\s|$)",
                name,
                text,
                flags=re.IGNORECASE,
            )
        if new == text:
            break
        text = new
    text = re.sub(r"\bSimilar emotes\s*:\s*", "Similar emotes ", text, flags=re.IGNORECASE)
    for token in emote_tokens(report):
        text = re.sub(
            rf"(?<!\S)['\"]({re.escape(token)})['\"](?=\s|$)",
            token,
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"(?<!\S)({re.escape(token)})([:;,.'\"!?]+)(?=\s|$)",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"([('])({re.escape(token)})(?=\s|$)",
            r"\2",
            text,
            flags=re.IGNORECASE,
        )
    text = text.strip(" '\"")
    return _clip(text, max_chars)


def _should_use_llm_synthesis(report: dict[str, Any]) -> bool:
    if report.get("has_vector"):
        return True
    if report.get("registry_tags") or report.get("neighbor_tags"):
        return True
    # Archive-only/no-vector cases are useful as raw receipts, but too easy for
    # the model to overread into confident valence. Use the deterministic
    # archive-term fallback until a usage vector or tags exist.
    return False


async def chat_response(report: dict[str, Any], *, detail: bool = False,
                        raw: bool = False, max_chars: int = 470) -> str:
    if raw:
        return raw_report(report, max_chars=max_chars)
    if not _should_use_llm_synthesis(report):
        return format_chat(report, detail=detail, raw=False, max_chars=max_chars)
    try:
        from services import llm
        from utils.output_filter import is_clean

        if await llm.available():
            answer = await llm.chat(
                synthesis_messages(report),
                max_tokens=100,
                temperature=0.25,
            )
            text = clean_synthesis(report, answer, max_chars=max_chars)
            if text and is_clean(text):
                return _append_similar_clause(report, text, max_chars=max_chars)
    except Exception:
        pass
    return format_chat(report, detail=detail, raw=False, max_chars=max_chars)


def format_chat(report: dict[str, Any], *, detail: bool = False,
                raw: bool = False, max_chars: int = 470) -> str:
    if raw:
        return raw_report(report, max_chars=max_chars)
    sentence = format_sentence(report, max_chars=max_chars)
    return sentence
