"""Evidence-backed archive/lore question helpers.

This is retrieval first, answer second: return receipts from the archive,
fact-bank claims, and emote meaning data instead of inventing a polished answer
without sources.
"""

from __future__ import annotations

import re
from pathlib import Path

import config
from utils import chat_archive, emote_meaning, fact_bank, message_quality

# Cosine floors for the dense (semantic) retrieval lane. The dense lane exists to
# find PARAPHRASES (no shared keyword), so it is gated by similarity, not by
# lexical overlap: an "anchored" hit (shares a query term) passes at the lower
# floor; an "unanchored" paraphrase must clear a higher confidence bar.
_SEM_FLOOR = float(getattr(config, "LLM_SEMANTIC_MIN_SCORE", 0.50))
_SEM_FLOOR_UNANCHORED = float(getattr(config, "LLM_SEMANTIC_UNANCHORED_MIN_SCORE", 0.62))

CHAT_FACT_MIN_SUPPORT = 2
QUERY_INTENT_TERMS = {
    "love", "loves", "loved", "like", "likes", "liked", "enjoy", "enjoys",
    "enjoyed", "hate", "hates", "hated", "prefer", "prefers", "preferred",
}


def clip(text: str, n: int = 120) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "..."


def _author_exclude_terms(author: str | None) -> set[str]:
    if not author:
        return set()
    terms = set()
    try:
        keys = chat_archive.author_keys(author)
    except Exception:
        keys = [author]
    for key in keys:
        norm = chat_archive.normalize_author(key)
        terms.add(norm)
        terms.add(norm.replace("_", ""))
        terms.update(part for part in norm.split("_") if part)
    return {term.casefold() for term in terms if term}


def _strip_scoped_author_terms(query: str, author: str | None) -> str:
    """Remove repeated scoped usernames from a natural-language query."""
    excludes = _author_exclude_terms(author)
    if not excludes:
        return query.strip()

    def repl(match: re.Match) -> str:
        raw = match.group(0).lstrip("@")
        norm = chat_archive.normalize_author(raw)
        variants = {raw.casefold(), norm.casefold(), norm.replace("_", "").casefold()}
        variants.update(part.casefold() for part in norm.split("_") if part)
        return " " if variants & excludes else match.group(0)

    stripped = re.sub(r"@?[A-Za-z0-9_]{2,40}", repl, query or "")
    return re.sub(r"\s+", " ", stripped).strip()


def _focus_terms(query: str, author: str | None = None) -> list[str]:
    excludes = _author_exclude_terms(author)
    return [
        term for term in chat_archive.query_terms(query, exclude_terms=excludes)
        if term not in QUERY_INTENT_TERMS
    ]


def _matches_focus(text: str, focus_terms: list[str]) -> bool:
    if not focus_terms:
        return True
    hay_tokens = set(chat_archive.line_match_key(text).split())
    return any(term in hay_tokens for term in focus_terms)


def parse_params(params: list[str], current_channel: str | None = None) -> dict:
    """Parse chat command args for ~askchat.

    Supports explicit `user=`, `author=`, `about=`, and `chat=`. For ergonomic
    chat use, a first token that resolves to an archived author is also treated
    as the user scope: `~askchat fernardo minecraft`.
    """
    author = None
    channel = None
    rest = []
    for token in params:
        low = token.lower()
        if low.startswith(("user=", "author=", "about=")):
            author = token.split("=", 1)[1].strip().lstrip("@") or author
        elif low.startswith("chat="):
            value = token.split("=", 1)[1].strip().lstrip("#").lower()
            if value in {"", "all", "*"}:
                channel = None
            elif value in {"here", "this"}:
                channel = current_channel
            else:
                channel = value
        else:
            rest.append(token)

    if not author and len(rest) >= 2:
        candidate = rest[0].lstrip("@")
        if candidate.lower() not in {"has", "did", "does", "what", "who", "when", "where", "why"}:
            try:
                if chat_archive.stats(candidate):
                    author = candidate
                    rest = rest[1:]
            except Exception:
                pass

    author = chat_archive.normalize_author(author) if author else None
    return {
        "author": author,
        "channel": chat_archive.normalize_channel(channel) if channel else None,
        "query": _strip_scoped_author_terms(" ".join(rest).strip(), author),
    }


def _search_all_hits(query: str, *, channel: str | None = None, limit: int = 5) -> list[dict]:
    q = chat_archive._fts_query(query)
    if not q:
        return []
    chan_sql, chan_params = chat_archive._channel_filter(channel)
    cmd_sql, cmd_params = chat_archive._command_filter(False)
    conn = chat_archive.connect()
    rows = conn.execute(
        "SELECT m.id, m.sent_at, m.channel, m.author, m.content FROM messages_fts f "
        "CROSS JOIN messages m ON m.id = f.rowid "
        f"WHERE f.messages_fts MATCH ? {chan_sql}{cmd_sql}"
        "AND length(m.content) <= ? "
        "ORDER BY bm25(messages_fts), m.sent_at DESC LIMIT ?",
        [q, *chan_params, *cmd_params, 260, limit * 5],
    ).fetchall()
    out = []
    seen = set()
    focus = _focus_terms(query)
    for row_id, sent_at, channel, author, content in rows:
        if not _usable_hit(content):
            continue
        if not _matches_focus(content, focus):
            continue
        key = (chat_archive.normalize_author(author), chat_archive.line_match_key(content))
        if not key[1] or key in seen:
            continue
        seen.add(key)
        out.append({
            "id": row_id,
            "sent_at": sent_at,
            "channel": channel,
            "author": chat_archive.normalize_author(author),
            "text": content,
        })
    return out


def _usable_hit(text: str) -> bool:
    if not message_quality.usable_for_snippet_context(text, max_chars=320):
        return False
    clean = message_quality.clean_text(text, strip_emotes=True, strip_urls=True)
    toks = message_quality.tokens(clean)
    if len(clean) < 18 or len(toks) < 3:
        return False
    return not message_quality.spam_like(clean, toks)


def _author_hits(author: str, query: str, *, channel: str | None = None, limit: int = 5) -> list[dict]:
    rows = chat_archive.search_author_hits(author, query, limit=limit * 6, max_chars=300)
    out = []
    seen = set()
    focus = _focus_terms(query, author)
    for row_id, sent_at, row_channel, content in rows:
        if channel and chat_archive.normalize_channel(row_channel) != channel:
            continue
        if not _usable_hit(content):
            continue
        if not _matches_focus(content, focus):
            continue
        key = chat_archive.line_match_key(content)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({
            "id": row_id,
            "sent_at": sent_at,
            "channel": row_channel,
            "author": chat_archive.normalize_author(author),
            "text": content,
        })
        if len(out) >= limit:
            break
    return out


def _recover_author_meta(author: str, texts: list[str], *, channel: str | None = None) -> dict:
    """{line_match_key: row-dict} for dense-only hits, so a paraphrase line the
    embedder found (but bm25 never keyword-matched) still carries its real
    id/sent_at/channel. One batched query over the author's own rows."""
    texts = [t for t in texts if t]
    if not texts:
        return {}
    conn = chat_archive.connect()
    keys = chat_archive.author_keys(chat_archive.normalize_author(author))
    a_ph = ",".join("?" for _ in keys)
    t_ph = ",".join("?" for _ in texts)
    chan_sql, chan_params = chat_archive._channel_filter(channel, alias="m")
    rows = conn.execute(
        f"SELECT m.id, m.sent_at, m.channel, m.content FROM messages m "
        f"WHERE m.author IN ({a_ph}) AND m.content IN ({t_ph}){chan_sql}",
        [*keys, *texts, *chan_params],
    ).fetchall()
    out = {}
    for row_id, sent_at, row_channel, content in rows:
        key = chat_archive.line_match_key(content)
        if key and key not in out:
            out[key] = {"id": row_id, "sent_at": sent_at, "channel": row_channel,
                        "author": chat_archive.normalize_author(author), "text": content}
    return out


def _rrf_author_hits(author: str, query: str, *, channel: str | None = None,
                     limit: int = 5, rrf_k: int = 60) -> list[dict]:
    """Author archive hits fusing the bm25 lane (_author_hits) with the dense
    semantic lane (persona_msg_index) by Reciprocal Rank Fusion:

        score(line) = 1/(k + rank_bm25) + 1/(k + rank_dense)

    RRF needs no score calibration between bm25 and cosine (it is rank-only),
    so it is robust on the un-whitened embedding space and can never rank below
    the better single lane. bm25 stays the first lane so the strong keyword
    baseline is never silently lost; dense only ADDS paraphrase recall (the
    'does X like cars' -> 'my civic rips' case that shares no keywords). Falls
    back to pure bm25 when the index or embedder is unavailable.

    Each returned item keeps the archive schema (id/sent_at/channel/author/text)
    plus a 'lanes' tag for transparency."""
    from utils import persona_msg_index

    pool = max(limit * 4, 12)
    bm25 = _author_hits(author, query, channel=channel, limit=pool)

    dense_pairs = []
    if persona_msg_index.available(author):
        try:
            dense_pairs = persona_msg_index.semantic_hits(author, query, k=pool)
        except Exception:
            dense_pairs = []   # embedder down -> bm25-only, never break QA
    if not dense_pairs:
        return bm25[:limit]

    # dense lane, quality-gated with the same filters as bm25; deduped by key so
    # two near-identical stored vectors can't double-count one logical line.
    focus = _focus_terms(query, author)
    dense_ranked = []
    seen_keys = set()
    for score, text in dense_pairs:
        if not _usable_hit(text):
            continue
        # Gate by cosine, NOT by lexical overlap: the old `_matches_focus` hard
        # drop discarded ~75% of the top dense hits (incl. the literal best
        # paraphrase answers, e.g. "he plays apex" for "what video games").
        # Anchored hits pass at the low floor; pure paraphrases need more cosine.
        anchored = _matches_focus(text, focus)
        if score < (_SEM_FLOOR if anchored else _SEM_FLOOR_UNANCHORED):
            continue
        key = chat_archive.line_match_key(text)
        if key and key not in seen_keys:
            seen_keys.add(key)
            dense_ranked.append((key, text))

    # recover metadata (sent_at/channel) for dense lines bm25 didn't surface.
    # NOTE: exact-content recovery works for single-message utterances; a MERGED
    # multi-message utterance has no single matching row, so it drops here. Fixing
    # that fully needs the message index to store row ids (a rebuild) — until then
    # the dense lane adds single-message paraphrase recall, not merged-utterance.
    bm25_by_key = {chat_archive.line_match_key(h["text"]): h for h in bm25}
    bm25_by_key.pop(None, None)
    missing = [t for k, t in dense_ranked if k not in bm25_by_key]
    recovered = _recover_author_meta(author, missing, channel=channel)

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    lanes: dict[str, set] = {}

    for rank, hit in enumerate(bm25):
        key = chat_archive.line_match_key(hit["text"])
        if not key:
            continue
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        items.setdefault(key, hit)
        lanes.setdefault(key, set()).add("bm25")

    for rank, (key, text) in enumerate(dense_ranked):
        meta = bm25_by_key.get(key) or recovered.get(key)
        if not meta:
            continue   # dense-only paraphrase whose row we couldn't recover
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        items.setdefault(key, meta)
        lanes.setdefault(key, set()).add("dense")

    ordered = sorted(scores, key=lambda k: -scores[k])
    out = []
    for key in ordered[:limit]:
        item = dict(items[key])
        item["lanes"] = sorted(lanes[key])
        out.append(item)
    return out


def _chatworthy_fact(fact: dict) -> bool:
    support = int(fact.get("support_count") or 0)
    return support >= CHAT_FACT_MIN_SUPPORT


def _scope(report: dict) -> str:
    scope = report.get("author") or "archive"
    if report.get("channel"):
        scope += f"#{report['channel']}"
    return scope


def _near_author(author: str, query: str, *, channel: str | None = None, limit: int = 2) -> list[dict]:
    rows = chat_archive.nearest_author_lines(
        author, query, limit=limit, min_score=0.65, channel=channel
    )
    return [
        {
            "score": round(float(score), 3),
            "sent_at": sent_at,
            "channel": channel,
            "author": chat_archive.normalize_author(author),
            "text": content,
        }
        for score, sent_at, channel, content in rows
    ]


def _fact_hits(author: str | None, query: str, limit: int = 4) -> list[dict]:
    rows = fact_bank.load_jsonl()
    if not rows:
        return []
    terms = chat_archive.query_terms(query)
    required = terms[0] if len(terms) >= 2 else None
    out = []
    for row in fact_bank.search(rows, author=author, query=query, limit=limit * 4):
        if required:
            hay = f"{row.get('kind', '')} {row.get('claim', '')}".casefold()
            if required.casefold() not in hay:
                continue
        ev = (row.get("evidence") or [{}])[0]
        out.append({
            "author": row.get("author"),
            "kind": row.get("kind"),
            "claim": row.get("claim"),
            "support_count": row.get("support_count", 0),
            "confidence": row.get("confidence", 0.0),
            "sent_at": ev.get("sent_at"),
            "channel": ev.get("channel"),
            "evidence": ev.get("clean_text") or ev.get("text") or "",
        })
        if len(out) >= limit:
            break
    return out


def _emote_hits(query: str, limit: int = 2) -> list[dict]:
    out = []
    seen = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,40}", query or ""):
        if raw.lower() in seen:
            continue
        seen.add(raw.lower())
        name, info = emote_meaning.lookup(raw)
        near = emote_meaning.nearest_emotes(raw, n=4)
        if not info and not near:
            continue
        out.append({
            "name": name or raw,
            "tags": (info or {}).get("tags") or [],
            "original": (info or {}).get("original"),
            "near": [emote for emote, _score in near],
        })
        if len(out) >= limit:
            break
    return out


def build_report(query: str, *, author: str | None = None,
                 channel: str | None = None, limit: int = 5) -> dict:
    query = query.strip()
    author = chat_archive.normalize_author(author) if author else None
    query = _strip_scoped_author_terms(query, author)
    channel = chat_archive.normalize_channel(channel) if channel else None
    facts = _fact_hits(author, query, limit=4)
    archive = _rrf_author_hits(author, query, channel=channel, limit=limit) if author else (
        _search_all_hits(query, channel=channel, limit=limit)
    )
    near = [] if archive or not author else _near_author(author, query, channel=channel)
    emotes = _emote_hits(query)
    terms = chat_archive.query_terms(query)
    return {
        "query": query,
        "author": author,
        "channel": channel,
        "terms": terms,
        "facts": facts,
        "archive": archive,
        "near": near,
        "emotes": emotes,
    }


# Human-readable rendering of a fact's claim KIND. Without this the raw kind
# token (e.g. "self_negative_identity", a fact_bank.CLAIM_PATTERNS category for
# "I'm not ..." lines) leaks into chat and reads like a person/source — which is
# alarming and wrong. A claim is always attributed to its real chatter author.
_KIND_PHRASE = {
    "self_identity": "says they are",
    "self_negative_identity": "says they're not",
    "preference_positive": "likes",
    "preference_negative": "dislikes",
    "belief": "thinks",
    "possession": "has",
    "activity": "does",
}


def _fact_phrase(fact: dict, claim_len: int = 150) -> str:
    """Render a fact-bank claim as a human sentence attributed to its author."""
    who = fact.get("author") or "someone"
    verb = _KIND_PHRASE.get(fact.get("kind", ""), "said")
    count = fact.get("support_count", 1) or 1
    rep = f" ({count}x)" if count > 1 else ""
    return f"{who} {verb} \"{clip(fact['claim'], claim_len)}\"{rep}"


def evidence_items(report: dict, max_items: int = 6) -> list[dict]:
    """Evidence lines safe to give an answer model.

    One-off fact-bank claims stay out of synthesis; they are candidate memory
    rows, not verified facts. The model gets raw archive receipts first.
    """
    items = []
    for hit in report.get("archive", [])[:4]:
        items.append({
            "label": f"A{len([i for i in items if i['label'].startswith('A')]) + 1}",
            "text": (
                f"{hit['author']}#{hit['channel']} {hit['sent_at'][:10]}: "
                f"\"{clip(hit['text'], 180)}\""
            ),
        })
    for hit in report.get("near", [])[:1]:
        items.append({
            "label": f"N{len([i for i in items if i['label'].startswith('N')]) + 1}",
            "text": (
                f"near {hit['score']:.0%} {hit['author']}#{hit['channel']} "
                f"{hit['sent_at'][:10]}: \"{clip(hit['text'], 180)}\""
            ),
        })
    for fact in [f for f in report.get("facts", []) if _chatworthy_fact(f)][:2]:
        items.append({
            "label": f"F{len([i for i in items if i['label'].startswith('F')]) + 1}",
            "text": _fact_phrase(fact, 160),
        })
    for emote in report.get("emotes", [])[:1]:
        bits = []
        if emote.get("tags"):
            bits.append("tags=" + ",".join(emote["tags"][:4]))
        if emote.get("near"):
            bits.append("used_like=" + " ".join(emote["near"][:5]))
        if bits:
            items.append({
                "label": "E1",
                "text": f"emote {emote['name']}: " + "; ".join(bits),
            })
    return items[:max_items]


def has_strong_evidence(report: dict) -> bool:
    return bool(evidence_items(report))


def answer_messages(report: dict) -> list[dict]:
    evidence = evidence_items(report)
    evidence_text = "\n".join(f"[{item['label']}] {item['text']}" for item in evidence)
    system = (
        "You answer Twitch chat archive questions using only the provided evidence. "
        "Be cautious: a quote is evidence that someone said a line, not proof a broad "
        "claim is true. If evidence is weak, say that. If the question asks whether "
        "someone likes/loves/hates/prefers something, only call it a preference when "
        "a receipt directly says that; otherwise say the evidence only shows mentions "
        "or behavior. Do not invent facts, do not use outside knowledge, and cite "
        "receipt labels like [A1]. Keep it under 360 chars."
    )
    user = (
        f"Scope: {_scope(report)}\n"
        f"Question: {report.get('query', '')}\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Answer in one compact chat message."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_answer_chat(report: dict, answer: str, max_chars: int = 470) -> str:
    answer = re.sub(r"\s+", " ", answer or "").strip().strip('"')
    answer = re.sub(r"^(?:answer|final)\s*:\s*", "", answer, flags=re.I)
    if not answer:
        return ""
    if evidence_items(report) and not re.search(r"\[[A-Z]\d+\]", answer):
        labels = " ".join(f"[{item['label']}]" for item in evidence_items(report)[:2])
        answer = f"{answer} {labels}"
    prefix = f"{_scope(report)}: "
    return clip(prefix + answer, max_chars)


def format_chat(report: dict, max_chars: int = 470) -> str:
    parts = [f"{_scope(report)}: "]

    for hit in report.get("archive", [])[:2]:
        parts.append(
            f"{hit['author']}#{hit['channel']} {hit['sent_at'][:10]} "
            f"\"{clip(hit['text'], 75)}\""
        )

    for hit in report.get("near", [])[:1]:
        parts.append(
            f"near {hit['score']:.0%} {hit['sent_at'][:10]} "
            f"\"{clip(hit['text'], 80)}\""
        )

    chat_facts = [fact for fact in report.get("facts", []) if _chatworthy_fact(fact)]
    for fact in chat_facts[:2]:
        parts.append(_fact_phrase(fact, 70))

    for emote in report.get("emotes", [])[:1]:
        bits = []
        if emote.get("tags"):
            bits.append("tags " + ",".join(emote["tags"][:3]))
        if emote.get("near"):
            bits.append("used like " + " ".join(emote["near"][:3]))
        if bits:
            parts.append(f"emote {emote['name']}: " + "; ".join(bits))

    if len(parts) == 1:
        terms = ", ".join(report.get("terms") or [])
        suffix = f" terms={terms}" if terms else ""
        weak = " Weak one-off claim candidates were ignored." if report.get("facts") else ""
        return f"No strong archive evidence found for '{clip(report.get('query', ''), 90)}'.{suffix}{weak}"

    out = " | ".join(parts)
    return clip(out, max_chars)


def format_cli(report: dict) -> str:
    lines = [f"query: {report['query']}"]
    if report.get("author"):
        lines.append(f"author: {report['author']}")
    if report.get("channel"):
        lines.append(f"channel: {report['channel']}")
    if report.get("terms"):
        lines.append("terms: " + ", ".join(report["terms"]))
    if report.get("facts"):
        lines.append("\nclaims:")
        for fact in report["facts"]:
            lines.append(
                f"- {_fact_phrase(fact)} [{fact['kind']}] conf={fact['confidence']}"
            )
            if fact.get("evidence"):
                lines.append(f"  {fact.get('sent_at')} #{fact.get('channel')}: {fact['evidence']}")
    if report.get("archive"):
        lines.append("\narchive hits:")
        for hit in report["archive"]:
            lines.append(
                f"- {hit['sent_at']} #{hit['channel']} {hit['author']}: {hit['text']}"
            )
    if report.get("near"):
        lines.append("\nnear matches:")
        for hit in report["near"]:
            lines.append(
                f"- {hit['score']:.0%} {hit['sent_at']} #{hit['channel']} {hit['author']}: {hit['text']}"
            )
    if report.get("emotes"):
        lines.append("\nemotes:")
        for emote in report["emotes"]:
            bits = []
            if emote.get("original") and emote["original"] != emote["name"]:
                bits.append(f"alias={emote['original']}")
            if emote.get("tags"):
                bits.append("tags=" + ",".join(emote["tags"][:6]))
            if emote.get("near"):
                bits.append("near=" + " ".join(emote["near"][:6]))
            lines.append(f"- {emote['name']}: " + "; ".join(bits))
    if len(lines) <= 4:
        lines.append("\nNo evidence found.")
    return "\n".join(lines)
