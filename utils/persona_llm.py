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

import config
from services import llm
from utils import chat_archive


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


def _usable_exemplar(message: str) -> bool:
    if not message:
        return False
    stripped = message.lstrip()
    if stripped.startswith(config.PREFIX) or stripped.startswith("<"):
        return False  # commands to this bot or to other bots (<grok, <news ...)
    if "http://" in message or "https://" in message or "www." in message:
        return False  # links teach nothing about voice and leak into output
    words = message.split()
    if len(words) < 2 or len(message) > 240:
        return False
    # require at least one real lowercase word — drops pure ping+emote lines
    return any(re.search(r"[a-z]{3}", w) for w in words)


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
    term_set = {t.lower() for t in terms}
    scored = []
    for hit in hits:
        content = hit[3]
        if not _usable_exemplar(content):
            continue
        words = content.split()
        content_terms = {w.strip(".,!?\"'").lower() for w in words}
        overlap = len(term_set & content_terms)
        shape = 1 if 4 <= len(words) <= 30 else 0
        scored.append((overlap * 3 + shape, hit))
    scored.sort(key=lambda x: -x[0])
    return [hit for _, hit in scored]


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
        relevant = _unique_messages(
            relevant_exemplars(author, query_text, flat_budget * 2,
                               exclude_terms=exclude_terms),
            flat_budget, seen=set(used),
        )
    seen = set(relevant) | used
    signature = _unique_messages(exemplars(author, n, channel=channel),
                                 n - len(relevant) - len(snippets) * 5, seen)
    return signature, relevant, snippets


def _conversation_rows(recent):
    return [
        row for row in recent
        if not (row[2] or "").lstrip().startswith(config.PREFIX)
    ]


def _retrieval_text(recent, user_message: str | None) -> str:
    parts = []
    if user_message:
        parts.append(user_message)
    # Use content only, not author labels; names and command words are noisy
    # retrieval anchors, while the actual message text carries the topic.
    for _, _, content in _conversation_rows(recent)[-12:]:
        parts.append(content)
    return "\n".join(parts)


def _clean_output(text: str, author: str) -> str:
    text = (text or "").strip()
    # models sometimes wrap the line in quotes, prepend "name:", or imitate
    # the snippet hit-marker (">> author: line") from the evidence section
    text = re.sub(r"^\s*(?:>+\s*)+", "", text)
    text = re.sub(rf"^{re.escape(author)}\s*[:>-]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:>+\s*)+", "", text)
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text.split("\n")[0].strip()  # one chat line only


def _copy_key(text: str) -> str:
    return chat_archive.line_match_key(text)


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
        f"line. Stay in character and output only the message."
    )
    user = (
        f"Current chat in #{channel}:\n{ctx}\n\n"
        f"Message directed at the persona: {user_message or '(none)'}\n\n"
        f"Copied draft to avoid:\n{copied_output}\n\n"
        f"Archived line it was too close to:\n{copied_source}\n\n"
        f"Small style sample from {author}:\n" + "\n".join(examples)
        + f"\n\nWrite a new {author} chat line now."
    )
    raw = await llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=120,
        temperature=1.05 if mode == "hyper" else 0.95,
    )
    repaired = _clean_output(raw, author)
    if not repaired:
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
                   model_override: str = None) -> str | None:
    t0 = time.time()
    exemplar_count = exemplar_count or config.LLM_EXEMPLARS
    context_count = context_count or config.LLM_CONTEXT
    recent = chat_archive.latest(channel, context_count)
    ctx_rows = _conversation_rows(recent)
    ctx = "\n".join(f"{a}: {c}" for _, a, c in ctx_rows) or "(quiet right now)"
    # Other chatters' names in the live context are addressing, not topic —
    # without this they dominate retrieval ranking (smoke-test finding).
    ctx_names = {a for _, a, _ in ctx_rows}
    signature, relevant, snippets = select_evidence(
        author, _retrieval_text(recent, user_message), n=exemplar_count,
        exclude_terms=ctx_names, channel=channel,
    )
    ab_model = _roll_ab_model(invoked_by, model_override)
    event = {
        "author": chat_archive.normalize_author(author),
        "channel": chat_archive.normalize_channel(channel),
        "mode": mode,
        "model": ab_model or config.LLM_MODEL,
        "invoked_by": invoked_by,
        "user_message": user_message,
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
        f"explain. Output ONE single chat message as {author} and nothing else. "
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
        user += f'Someone says to you: "{user_message}"\n'
    user += f"Write {author}'s next chat message now."
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
