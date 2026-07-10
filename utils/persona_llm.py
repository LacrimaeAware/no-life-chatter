"""LLM persona engine: speak as a real chatter, using their own messages.

Many-shot voice cloning (no training): the prompt blends a random signature
sample from the person's full history with messages retrieved from that same
author for the current chat topic. The model sees those examples plus the live
conversation, then writes their next line. Because the exemplars keep their
natural length distribution, output isn't forced terse — it lands where they
actually land. Two modes: 'normal' (natural, conversational) and 'hyper' (their
traits cranked up for comedy).

Runs against any OpenAI-compatible endpoint (services/llm.py) — LM Studio's
local server by default, so edgy content stays on the machine.
"""

import json
import logging
import os
import random
import re
import time
from collections import Counter

import config
from services import llm
from utils import chat_archive, message_quality
from utils.log_rotation import rotate_file


def log_event(event: dict) -> None:
    """Append one persona event to the private JSONL log (never raises).

    Every ~persona/~hyper/reaction generation gets recorded — the evidence
    that was fed, every candidate with its rejection reason, and the final
    line — so quality problems can be diagnosed from real usage instead of
    hand-built smoke cases. The file lives in gitignored data/unsynced/.
    """
    if not getattr(config, "PERSONA_LOG", True):
        return
    try:
        path = getattr(config, "PERSONA_LOG_FILE",
                       os.path.join("data", "unsynced", "persona_logs.jsonl"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rotate_file(path, max_bytes=25 * 1024 * 1024, keep=5)
        event.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.debug(f"persona log write failed: {e}")

_exemplar_cache = {}
_archive_line_cache = {}
_last_rejection = None


class _CopiedPersonaOutput(Exception):
    pass

MODE_INSTRUCTION = {
    "normal": (
        "Reply naturally, exactly as they would in chat — same length, slang, "
        "emotes, capitalization and energy. Don't force it short; match how "
        "they actually talk."
    ),
    "hyper": (
        "Crank their most recognizable habits up to eleven — wacky, chaotic, "
        "over-the-top and funny — but still unmistakably THEM. Go a bit longer "
        "and weirder than usual."
    ),
}

OUTPUT_CONTRACT = (
    "Output contract: return exactly one raw Twitch chat line only. "
    "No explanations, no analysis, no 'based on the chat history', no username "
    "or speaker label, no colon-prefix format, no quotes, and no line breaks."
)


def _usable_exemplar(message: str) -> bool:
    return message_quality.usable_for_persona_exemplar(message)


def _repeated_token_spam(words) -> bool:
    return message_quality.repeated_token_spam(words)


def _usable_snippet_context(message: str) -> bool:
    return message_quality.usable_for_snippet_context(message)


def _unique_messages(messages, n: int, seen=None):
    if n <= 0:
        return []
    seen = seen or set()
    out = []
    for message in messages:
        if not _usable_exemplar(message) or message in seen:
            continue
        seen.add(message)
        out.append(message)
        if len(out) >= n:
            break
    return out


def _content_terms(text: str) -> set[str]:
    """Clean topic terms from a candidate evidence line.

    Uses the same query hygiene as retrieval so emotes, pings, and scaffolding
    words do not make a line look more relevant than it is.
    """
    return set(chat_archive.query_terms(text, max_terms=24))


def _term_overlap_count(query_terms, evidence_terms) -> int:
    count = 0
    for query in {t.lower() for t in query_terms}:
        for evidence in evidence_terms:
            if query == evidence:
                count += 1
                break
            if len(query) >= 4 and evidence.startswith(query):
                count += 1
                break
            if len(evidence) >= 4 and query.startswith(evidence):
                count += 1
                break
    return count


def _term_overlap_weight(query_terms, evidence_terms) -> float:
    weight = 0.0
    seen_queries = list(dict.fromkeys(t.lower() for t in query_terms))
    for idx, query in enumerate(seen_queries):
        matched = False
        for evidence in evidence_terms:
            if query == evidence:
                matched = True
                break
            if len(query) >= 4 and evidence.startswith(query):
                matched = True
                break
            if len(evidence) >= 4 and query.startswith(evidence):
                matched = True
                break
        if matched:
            # query_terms() is ordered by count, then first mention. For direct
            # commands the user's actual prompt is repeated, so its terms land
            # first and should matter more than incidental recent-context terms.
            weight += 2.0 if idx < 4 else 1.0
    return weight


def _evidence_score(text: str, terms, semantic_score: float | None = None) -> float:
    term_set = {t.lower() for t in terms}
    words = (text or "").split()
    evidence_terms = _content_terms(text)
    overlap = _term_overlap_count(term_set, evidence_terms)
    overlap_weight = _term_overlap_weight(terms, evidence_terms)
    if 4 <= len(words) <= 30:
        shape = 2.0
    elif 2 <= len(words) <= 60:
        shape = 1.0
    else:
        shape = 0.0
    score = (overlap_weight * 4.0) + shape
    if semantic_score is not None:
        # Cosine scores are already filtered before this point. Use the score
        # as a tiebreaker, not as permission for unrelated evidence to dominate.
        score += max(0.0, min(2.0, (semantic_score - 0.45) * 10.0))
        if term_set and overlap == 0:
            score -= 1.0
    return score


def _semantic_text_allowed(text: str, score: float, terms) -> bool:
    if not _usable_exemplar(text):
        return False
    term_set = {t.lower() for t in terms}
    overlap = bool(_term_overlap_count(term_set, _content_terms(text)))
    anchored_floor = getattr(config, "LLM_SEMANTIC_MIN_SCORE", 0.50)
    unanchored_floor = getattr(config, "LLM_SEMANTIC_UNANCHORED_MIN_SCORE", 0.62)
    floor = anchored_floor if overlap else unanchored_floor
    return score >= floor


def exemplars(author: str, n: int = None, channel: str = None):
    """~n messages from the author across their whole history.

    Mostly a STABLE seeded sample (same core every restart) plus a small fresh
    random tail. The old fully-random sample re-rolled the persona's entire
    style evidence on every cache rebuild — one lucky/unlucky draw could make
    the same persona brilliant one day and mush the next. A stable core keeps
    the voice consistent; the tail keeps it from fossilizing.
    """
    n = n or config.LLM_EXEMPLARS
    author_key = chat_archive.normalize_author(author)
    channel_key = chat_archive.normalize_channel(channel) if channel else None
    key = (author_key, n, channel_key)
    if key not in _exemplar_cache:
        # Channel-scoped voice: a persona invoked in a chat speaks the way the
        # person talks THERE, falling back to full history when they barely
        # chat in that channel.
        pool = []
        if channel_key:
            pool = [m for m in chat_archive.messages_for(author, channel=channel_key)
                    if _usable_exemplar(m)]
        if len(pool) < max(200, n):
            pool = [m for m in chat_archive.messages_for(author) if _usable_exemplar(m)]
        core_n = max(1, int(n * 0.8))
        stable = random.Random(f"persona-core:{author_key}")
        core_pool = list(pool)
        stable.shuffle(core_pool)
        core = _unique_messages(core_pool, core_n)
        fresh_pool = list(pool)
        random.shuffle(fresh_pool)
        tail = _unique_messages(fresh_pool, n - len(core), seen=set(core))
        _exemplar_cache[key] = core + tail
    return _exemplar_cache[key]


def _rank_hits(hits, terms):
    """Order retrieval hits by usefulness as persona evidence.

    bm25 alone let one-word emote lines and query-word echoes win. Boost hits
    that share actual topic terms and are conversation-sized; drop junk.
    """
    scored = []
    for hit in hits:
        content = hit[3]
        if not _usable_exemplar(content):
            continue
        scored.append((_evidence_score(content, terms), hit))
    scored.sort(key=lambda x: -x[0])
    return [hit for _, hit in scored]


def _rank_relevant_texts(keyword_texts, semantic_rows, terms):
    """Rank flat relevant examples from keyword and semantic retrieval.

    Semantic hits used to be prepended wholesale. That made loose embedding
    neighbors look like strong evidence. Now both sources compete under the
    same topic/shape score, with semantic similarity only as a tiebreaker.
    """
    scored = []
    seen = set()
    for idx, text in enumerate(keyword_texts):
        key = _copy_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        scored.append((_evidence_score(text, terms), -idx * 0.001, text))
    for idx, score_text in enumerate(semantic_rows):
        sem_score, text = score_text
        key = _copy_key(text)
        if not key or key in seen:
            continue
        if not _semantic_text_allowed(text, sem_score, terms):
            continue
        seen.add(key)
        scored.append((
            _evidence_score(text, terms, semantic_score=sem_score),
            -idx * 0.001,
            text,
        ))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [text for _, _, text in scored]


def relevant_exemplars(author: str, query_text: str, n: int = None,
                       exclude_terms=None):
    """Author-only examples relevant to the current chat topic, ranked."""
    n = n if n is not None else getattr(config, "LLM_RELEVANT_EXEMPLARS", 0)
    if n <= 0 or not (query_text or "").strip():
        return []
    hits = chat_archive.search_author_hits(
        author, query_text, limit=max(n * 4, 40), exclude_terms=exclude_terms
    )
    terms = chat_archive.query_terms(query_text, exclude_terms=exclude_terms)
    ranked = _rank_hits(hits, terms)
    return _unique_messages((content for _, _, _, content in ranked), n)


def evidence_snippets(author: str, query_text: str, hits_n: int = None,
                      exclude_terms=None):
    """Top retrieval hits expanded into small chat moments (±2 lines).

    An isolated matching line teaches vocabulary; the moment around it teaches
    *behavior* — what this person says in response to what. Returns
    (snippet_texts, used_contents) where used_contents are the hit lines, so
    the flat lists can avoid duplicating them.
    """
    hits_n = hits_n if hits_n is not None else getattr(config, "LLM_SNIPPET_HITS", 8)
    if hits_n <= 0 or not (query_text or "").strip():
        return [], set()
    author_key = chat_archive.normalize_author(author)
    hits = chat_archive.search_author_hits(
        author, query_text, limit=max(hits_n * 5, 40), exclude_terms=exclude_terms
    )
    terms = chat_archive.query_terms(query_text, exclude_terms=exclude_terms)
    ranked = _rank_hits(hits, terms)

    snippets, used_ids, used_contents = [], set(), set()
    for hit_id, _, channel, content in ranked:
        if len(snippets) >= hits_n:
            break
        if hit_id in used_ids:
            continue  # already shown inside an earlier snippet's window
        window = chat_archive.context_window(hit_id, channel, before=2, after=2)
        lines = []
        for row_id, row_author, row_content in window:
            used_ids.add(row_id)
            if row_id != hit_id and not _usable_snippet_context(row_content):
                continue
            text = row_content if len(row_content) <= 160 else row_content[:159] + "…"
            marker = ">> " if row_id == hit_id else ""
            lines.append(f"{marker}{row_author}: {text}")
        used_contents.add(content)
        snippets.append("\n".join(lines))
    return snippets, used_contents


def select_exemplars(author: str, query_text: str, n: int = None,
                     relevant_n: int = None):
    """Blend stable voice samples with per-call retrieved examples.

    Back-compat shape for the export/smoke scripts: (signature, relevant).
    The live prompt path uses select_evidence(), which adds snippets.
    """
    n = n or config.LLM_EXEMPLARS
    relevant_budget = (
        getattr(config, "LLM_RELEVANT_EXEMPLARS", 0)
        if relevant_n is None else relevant_n
    )
    relevant_target = min(n, max(0, relevant_budget), max(0, int(n * 0.6)))
    relevant = relevant_exemplars(author, query_text, relevant_target)

    seen = set(relevant)
    signature = _unique_messages(exemplars(author, n), n - len(relevant), seen)
    return signature, relevant


def select_evidence(author: str, query_text: str, n: int = None,
                    exclude_terms=None, channel: str = None):
    """Everything the prompt needs: (signature, relevant_flat, snippets).

    Snippets (chat moments) take priority inside the relevant budget — each
    one costs ~5 lines — and the flat relevant list fills what's left, so the
    total prompt size stays at the configured scale.
    """
    n = n or config.LLM_EXEMPLARS
    relevant_budget = min(
        max(0, getattr(config, "LLM_RELEVANT_EXEMPLARS", 0)), max(0, int(n * 0.6))
    )
    snippets, used = evidence_snippets(author, query_text,
                                       exclude_terms=exclude_terms)
    flat_budget = max(0, relevant_budget - len(snippets) * 5)
    relevant = []
    if flat_budget:
        terms = chat_archive.query_terms(query_text, exclude_terms=exclude_terms)
        keyword_pool = list(relevant_exemplars(author, query_text, flat_budget * 3,
                                               exclude_terms=exclude_terms))
        semantic_rows = []
        # Semantic retrieval (config [llm] semantic_retrieval): messages near
        # the conversation in MEANING, which FTS keyword overlap can't find.
        # Filtered and ranked with keyword evidence so loose associations do
        # not overpower direct archive evidence.
        if getattr(config, "LLM_SEMANTIC_RETRIEVAL", False):
            try:
                from utils import persona_msg_index
                if persona_msg_index.available(author):
                    semantic_rows = persona_msg_index.semantic_hits(
                        author, query_text, k=max(flat_budget * 3, 24))
            except Exception as e:
                logging.debug(f"semantic retrieval skipped: {e}")
        pool = _rank_relevant_texts(keyword_pool, semantic_rows, terms)
        relevant = _unique_messages(pool, flat_budget, seen=set(used))
    seen = set(relevant) | used
    signature = _unique_messages(exemplars(author, n, channel=channel),
                                 n - len(relevant) - len(snippets) * 5, seen)
    return signature, relevant, snippets


def _conversation_rows(recent):
    out, seen = [], set()
    for row in recent:
        if (row[2] or "").lstrip().startswith(config.PREFIX):
            continue
        key_text = chat_archive.line_match_key(row[2])
        if key_text:
            key = (chat_archive.normalize_author(row[1]), key_text)
            if key in seen:
                continue
            seen.add(key)
        out.append(row)
    return out


def _retrieval_text(recent, user_message: str | None) -> str:
    parts = []
    if user_message:
        # A direct ~persona question is the strongest retrieval signal. Repeat
        # it so unrelated recent chat cannot drown out the actual prompt.
        parts.extend([user_message] * 3)
    # Use content only, not author labels; names and command words are noisy
    # retrieval anchors, while the actual message text carries the topic.
    tail = 4 if user_message else 12
    for _, _, content in _conversation_rows(recent)[-tail:]:
        parts.append(content)
    return "\n".join(parts)


_LEADING_LABEL_RE = re.compile(
    r"^\s*@?(?P<label>[A-Za-z][A-Za-z0-9_]{2,25})\s*[:：>~-]\s*(?P<body>.*)$"
)
_LEADING_SPEAKER_RE = re.compile(
    r"^\s*@?[A-Za-z][A-Za-z0-9_]{2,25}\s*[:：]\s+\S"
)
_META_OUTPUT_RE = re.compile(
    r"(?:"
    r"based on .*(?:chat|history|style)|"
    r"chat history|"
    r"style of [A-Za-z0-9_]+|"
    r"(?:their|the)?\s*next (?:chat )?(?:message|line) (?:could|would) be|"
    r"(?:would|could) (?:say|reply|respond)|"
    r"a possible (?:reply|response)|"
    r"as an? (?:ai|assistant|language model)|"
    r"^here(?:'s| is)\b"
    r")",
    re.IGNORECASE,
)


def _strip_wrappers(line: str) -> str:
    line = (line or "").strip()
    line = re.sub(r"^\s*(?:>+\s*)+", "", line)
    line = re.sub(r"^\s*[-*]\s+", "", line)
    line = line.strip()
    if len(line) >= 2 and line[0] in "\"'" and line[-1] == line[0]:
        line = line[1:-1].strip()
    return line


def _strip_target_label(line: str, author: str) -> tuple[str, bool]:
    line = _strip_wrappers(line)
    match = _LEADING_LABEL_RE.match(line)
    if not match:
        return line, False
    label = match.group("label")
    if chat_archive.normalize_author(label) != chat_archive.normalize_author(author):
        return line, False
    return _strip_wrappers(match.group("body")), True


def _meta_output_preamble(text: str, author: str = "") -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if _META_OUTPUT_RE.search(text):
        return True
    author = (author or "").strip()
    if author and re.search(
        rf"\b{re.escape(author)}(?:'s)?\s+(?:next|reply|response)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def _clean_output(text: str, author: str) -> str:
    text = (text or "").replace("\r", "\n").strip()
    if not text:
        return ""
    lines = [_strip_wrappers(line) for line in text.split("\n")]
    lines = [line for line in lines if line]

    # Prefer a target-labeled line if the model emitted a preamble followed by
    # "author: actual message". Do not salvage other users' labels; those are
    # rejected later as speaker bleed.
    for line in lines:
        stripped, had_label = _strip_target_label(line, author)
        if had_label and stripped:
            return stripped

    for line in lines:
        stripped, _had_label = _strip_target_label(line, author)
        if stripped and not _meta_output_preamble(stripped, author):
            return stripped

    stripped, _had_label = _strip_target_label(lines[0], author)
    return stripped.strip()


def _copy_key(text: str) -> str:
    return chat_archive.line_match_key(text)


def _filter_excluded_evidence(signature, relevant, snippets, exclude_examples):
    """Remove held-out/private target lines from prompt evidence."""
    exclude_keys = {_copy_key(text) for text in (exclude_examples or []) if _copy_key(text)}
    if not exclude_keys:
        return signature, relevant, snippets

    def keep_text(text: str) -> bool:
        key = _copy_key(text)
        return not key or not any(
            key == excluded
            or (len(excluded) >= 8 and excluded in key)
            or (len(key) >= 8 and key in excluded)
            or chat_archive.line_similarity(text, excluded) >= 0.94
            for excluded in exclude_keys
        )

    def keep_snippet(snippet: str) -> bool:
        key = _copy_key(snippet)
        return not any(excluded and excluded in key for excluded in exclude_keys)

    return (
        [text for text in signature if keep_text(text)],
        [text for text in relevant if keep_text(text)],
        [snippet for snippet in snippets if keep_snippet(snippet)],
    )


def last_rejection() -> str | None:
    return _last_rejection


def _set_rejection(reason: str | None) -> None:
    global _last_rejection
    _last_rejection = reason


_last_model_tag = None


def last_model_tag() -> str | None:
    """Short tag (#llama / #lora) of the model that produced the most recent
    generation — None when the live A/B roll is off. Posting sites prepend it
    so chat can judge the two models on real usage."""
    return _last_model_tag


def resolve_model(name: str) -> str | None:
    """Resolve a user-supplied model= value: a configured shortcut (llama/lora),
    or a real id passed through. None if blank."""
    if not name:
        return None
    name = name.strip()
    return getattr(config, "LLM_MODEL_SHORTCUTS", {}).get(name.lower(), name)


def _roll_ab_model(invoked_by: str | None, override: str = None) -> str | None:
    """Pick a model for this generation: an explicit override (model= in the
    command) wins; otherwise the live A/B roll when configured. The compare
    script drives models explicitly, so it never rolls."""
    global _last_model_tag
    if override:
        low = override.lower()
        _last_model_tag = ("lora" if "lora" in low
                           else "llama" if "llama" in low
                           else low.split("/")[-1][:10])
        return override
    pool = getattr(config, "LLM_AB_MODELS", None)
    if not pool or invoked_by == "compare":
        _last_model_tag = None
        return None
    mid = random.choice(pool)
    low = mid.lower()
    _last_model_tag = ("lora" if "lora" in low
                       else "llama" if "llama" in low
                       else low.split("/")[-1][:10])
    return mid


def is_exact_archived_line(author: str, text: str) -> bool:
    """True when generated text is a normalized exact old line from this author."""
    key = chat_archive.normalize_author(author)
    if key not in _archive_line_cache:
        _archive_line_cache[key] = {
            _copy_key(message)
            for message in chat_archive.messages_for(author)
            if _copy_key(message)
        }
    return _copy_key(text) in _archive_line_cache[key]


def _near_example_copy(text: str, examples) -> str | None:
    """Return the copied example when output is too close to a prompt line."""
    for example in examples:
        if chat_archive.line_similarity(text, example) >= 0.94:
            return example
    return None


def _copied_source(author: str, text: str, examples) -> str | None:
    if not text:
        return None
    if is_exact_archived_line(author, text):
        return text
    return _near_example_copy(text, examples)


async def _repair_copied_output(author: str, channel: str, user_message: str | None,
                                mode: str, copied_output: str, copied_source: str,
                                signature, relevant, ctx: str) -> str | None:
    examples = _unique_messages(
        [*relevant[:10], *signature[:10]],
        16,
        seen={copied_source, copied_output},
    )
    if not examples:
        return None
    system = (
        f"You are rewriting a Twitch persona line for '{author}'. The previous "
        f"draft copied an archived line too closely. Write ONE new chat message "
        f"in {author}'s voice. Do not quote, paraphrase, or reuse the copied "
        f"line. Stay in character and output only the message. {OUTPUT_CONTRACT}"
    )
    user = (
        f"Current chat in #{channel}:\n{ctx}\n\n"
        f"Message directed at the persona: {user_message or '(none)'}\n\n"
        f"Copied draft to avoid:\n{copied_output}\n\n"
        f"Archived line it was too close to:\n{copied_source}\n\n"
        f"Small style sample from {author}:\n" + "\n".join(examples)
        + f"\n\nWrite a new {author} chat line now. {OUTPUT_CONTRACT}"
    )
    raw = await llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=120,
        temperature=1.05 if mode == "hyper" else 0.95,
    )
    repaired = _clean_output(raw, author)
    if not repaired:
        return None
    issue = _candidate_issues(author, repaired, [])
    if issue:
        logging.info("Rejected persona repair for %s (%s): %r", author, issue, repaired)
        return None
    if _copied_source(author, repaired, examples):
        logging.info("Rejected copied persona repair for %s: %r", author, repaired)
        return None
    return repaired


_BROKEN_MENTION_RE = re.compile(r"@\S*@|@\w+\w@")


def _candidate_issues(author: str, text: str, ctx_rows) -> str | None:
    """Reason this candidate must be rejected outright, or None if usable."""
    if not text:
        return "empty"
    if "\n" in text:
        return "multi-line output"
    if _meta_output_preamble(text, author):
        return "assistant preamble"
    if _LEADING_SPEAKER_RE.match(text):
        return "speaker label"
    if "http://" in text or "https://" in text or "www." in text:
        return "contains a URL"
    # Real chatters spam bot commands ($gpt/$remind/~persona), so models learn
    # to emit them — but the bot posting a command line is noise (and can
    # trigger OTHER bots). Personas address people with @, not with commands.
    if re.match(r"^[~$!][A-Za-z]{2,}", text.lstrip()):
        return "bot command line"
    if _BROKEN_MENTION_RE.search(text):
        return "broken mention"
    author_key = chat_archive.normalize_author(author)
    for _, ctx_author, ctx_content in ctx_rows[-12:]:
        if chat_archive.normalize_author(ctx_author) == author_key:
            continue
        if chat_archive.line_similarity(text, ctx_content) >= 0.9:
            return f"echoes {ctx_author}'s line"  # chat-bleed guard
    return None


def _candidate_score(text: str, engage_terms) -> float:
    words = text.split()
    shape = 1.0 if 2 <= len(words) <= 60 else 0.0
    if not engage_terms:
        return shape
    content_terms = {w.strip(".,!?\"'").lower() for w in words}
    overlap = len(set(engage_terms) & content_terms)
    return shape + overlap


async def generate(author: str, channel: str, user_message: str = None,
                   mode: str = "normal", exemplar_count: int = None,
                   context_count: int = None,
                   copy_strategy: str = "drop",
                   candidates: int = None,
                   invoked_by: str = None,
                   model_override: str = None,
                   recent_override=None,
                   exclude_examples=None) -> str | None:
    t0 = time.time()
    exemplar_count = exemplar_count or config.LLM_EXEMPLARS
    context_count = context_count or config.LLM_CONTEXT
    recent = (
        list(recent_override)
        if recent_override is not None
        else chat_archive.latest(channel, context_count)
    )
    ctx_rows = _conversation_rows(recent)
    ctx = "\n".join(f"{a}: {c}" for _, a, c in ctx_rows) or "(quiet right now)"
    # Other chatters' names in the live context are addressing, not topic —
    # without this they dominate retrieval ranking (smoke-test finding).
    ctx_names = {a for _, a, _ in ctx_rows}
    retrieval_text = _retrieval_text(recent, user_message)
    signature, relevant, snippets = select_evidence(
        author, retrieval_text, n=exemplar_count,
        exclude_terms=ctx_names, channel=channel,
    )
    signature, relevant, snippets = _filter_excluded_evidence(
        signature, relevant, snippets, exclude_examples
    )
    ab_model = _roll_ab_model(invoked_by, model_override)
    event = {
        "author": chat_archive.normalize_author(author),
        "channel": chat_archive.normalize_channel(channel),
        "mode": mode,
        "model": ab_model or config.LLM_MODEL,
        "invoked_by": invoked_by,
        "user_message": user_message,
        "retrieval_terms": chat_archive.query_terms(retrieval_text, exclude_terms=ctx_names),
        "context_tail": [f"{a}: {c}" for _, a, c in ctx_rows[-6:]],
        "n_signature": len(signature),
        "relevant": relevant,
        "snippets": snippets,
        "candidates": [],
    }
    if not signature and not relevant and not snippets:
        event["final"] = None
        event["outcome"] = "no evidence (no archived messages?)"
        log_event(event)
        return None

    exemplar_sections = []
    # Confirmed profile facts (fact bank v2) — verified, multi-day-corroborated
    # facts the person has stated about themselves, so the persona can KNOW
    # things (job, country, hobbies) instead of only sounding right. Confirmed
    # only: a persona prompt must not launder single-sighting guesses.
    try:
        from utils import user_profiles
        facts_line = user_profiles.profile_line(author)
    except Exception:
        facts_line = ""
    if facts_line:
        exemplar_sections.append(
            f"Verified facts {author} has stated about themselves "
            f"(use naturally when relevant; never recite as a list):\n{facts_line}"
        )
    if signature:
        exemplar_sections.append(
            f"Random real messages from {author} across their whole history:\n"
            + "\n".join(signature)
        )
    if relevant:
        exemplar_sections.append(
            f"Real messages from {author} relevant to this chat/topic:\n"
            + "\n".join(relevant)
        )
    if snippets:
        exemplar_sections.append(
            f"Real chat moments showing how {author} responds in situations "
            f"like this one (lines marked >> are {author}):\n\n"
            + "\n---\n".join(snippets)
        )

    system = (
        f"You ARE the Twitch chatter '{author}'. Below are real messages they have "
        f"sent — study their voice, vocabulary, emotes, spelling, punctuation, length "
        f"and attitude, and become them. {MODE_INSTRUCTION.get(mode, MODE_INSTRUCTION['normal'])} "
        f"You are NOT an assistant: never be helpful, never break character, never "
        f"explain. {OUTPUT_CONTRACT} "
        f"If a question or message is aimed at you, react to IT directly — answer it, "
        f"dodge it, mock it, or misunderstand it, all in-character — but never ignore "
        f"it and never refuse like an assistant. "
        f"The current conversation is context, not your material: do not reuse or "
        f"echo the other chatters' lines. "
        f"Use the examples as style evidence, but do not copy any example verbatim; "
        f"write a new line in their voice unless the user explicitly asked for a quote. "
        f"All examples below are from {author} only; use the relevant examples "
        f"to understand what they tend to say in this situation.\n\n"
        + "\n\n".join(exemplar_sections)
    )
    user = f"Current chat in #{channel}:\n{ctx}\n\n"
    if user_message:
        # WHO asks matters — a persona treats different chatters differently,
        # like the real person would. (Internal callers stay anonymous.)
        asker = (invoked_by or "").strip()
        if asker and asker.lower() not in ("compare", "ambient-reaction", "smoke"):
            user += f'{asker} says to you: "{user_message}"\n'
        else:
            user += f'Someone says to you: "{user_message}"\n'
    user += f"Write {author}'s next chat message now. {OUTPUT_CONTRACT}"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    temperature = 1.0 if mode == "hyper" else 0.85

    # Candidate selection: sample a couple of times (cheap locally — the big
    # prompt is processed once and cached by the server) and keep the best
    # valid line instead of shipping whatever the single draw produced.
    n_candidates = max(1, candidates or getattr(config, "LLM_CANDIDATES", 1))
    engage_terms = chat_archive.query_terms(
        (user_message or "") + " " + (ctx_rows[-1][2] if ctx_rows else ""),
        exclude_terms=ctx_names,
    )
    all_examples = [*signature, *relevant]
    best, best_score = None, -1.0
    first_copy = None
    for _ in range(n_candidates):
        raw = await llm.chat(messages, max_tokens=160, temperature=temperature,
                             model=ab_model)
        if not raw:
            event["candidates"].append({"text": None, "status": "llm failed/timeout"})
            continue
        out = _clean_output(raw, author)
        copied_example = _copied_source(author, out, all_examples)
        if copied_example:
            logging.info(
                "Rejected copied persona output for %s: %r copied from %r",
                author, out, copied_example,
            )
            event["candidates"].append(
                {"text": out, "status": "rejected", "reason": f"copy of: {copied_example}"})
            if first_copy is None:
                first_copy = (out, copied_example)
            continue
        issue = _candidate_issues(author, out, ctx_rows)
        if issue:
            logging.info("Rejected persona candidate for %s (%s): %r",
                         author, issue, out)
            event["candidates"].append(
                {"text": out, "status": "rejected", "reason": issue})
            continue
        score = _candidate_score(out, engage_terms)
        event["candidates"].append({"text": out, "status": "valid", "score": score})
        if score > best_score:
            best, best_score = out, score

    event["elapsed_ms"] = int((time.time() - t0) * 1000)
    if best:
        event["final"] = best
        event["outcome"] = "ok"
        log_event(event)
        return best
    if first_copy:
        out, copied_example = first_copy
        if copy_strategy == "repair":
            repaired = await _repair_copied_output(
                author, channel, user_message, mode, out, copied_example,
                signature, relevant, ctx,
            )
            if repaired:
                event["final"] = repaired
                event["outcome"] = "ok (copy repaired)"
                log_event(event)
                return repaired
            event["final"] = None
            event["outcome"] = "copied line; repair failed"
            log_event(event)
            raise _CopiedPersonaOutput
    event["final"] = None
    event["outcome"] = "no valid candidate"
    log_event(event)
    return None


async def generate_with_retry(author: str, channel: str, user_message: str = None,
                              mode: str = "normal",
                              invoked_by: str = None,
                              model_override: str = None) -> str | None:
    """Generate once with the full prompt, then retry compactly on failure.

    Local LM Studio can time out on heavy prompts, especially when two commands
    land close together. The compact retry keeps commands responsive without
    disabling the richer default prompt for normal cases.
    """
    _set_rejection(None)
    try:
        out = await generate(
            author, channel, user_message, mode=mode, copy_strategy="repair",
            invoked_by=invoked_by, model_override=model_override,
        )
    except _CopiedPersonaOutput:
        _set_rejection("model copied an archived line and the cheap repair failed")
        return None
    if out:
        return out
    if not chat_archive.messages_for(author):
        return None
    retry_exemplars = getattr(config, "LLM_RETRY_EXEMPLARS", 0)
    retry_context = getattr(config, "LLM_RETRY_CONTEXT", 0)
    if retry_exemplars <= 0:
        return None
    if retry_exemplars >= config.LLM_EXEMPLARS and retry_context >= config.LLM_CONTEXT:
        return None
    return await generate(
        author,
        channel,
        user_message,
        mode=mode,
        exemplar_count=retry_exemplars,
        context_count=retry_context,
        copy_strategy="drop",
        invoked_by=invoked_by,
        model_override=model_override,
    )
