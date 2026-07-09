"""Slot-based, LLM-verified user profiles — fact bank v2.

The v1 fact bank (utils/fact_bank.py) extracts regex *claims* and boosts
confidence by repetition, so repeated slang saturates to high confidence
("im crying" -> self_identity 0.95). This module inverts the design around a
fixed schema of profile slots (location, age, occupation, ...) and treats
truth as something to be EARNED per candidate:

1. RETRIEVAL (recall): targeted FTS anchor phrases per slot, scoped to the
   author's alias group. Cheap, over-generates.
2. JUDGMENT (precision): a local LLM sees the candidate line IN CONTEXT
   (surrounding chat) and answers: does the speaker sincerely assert this
   about THEMSELVES? Jokes, quotes, copypasta, hypotheticals -> rejected.
3. CORROBORATION: a value is only "confirmed" when sincerely asserted on
   >= 2 independent days. Single sightings stay "candidate". Conflicting
   confirmed values mark the slot "disputed" instead of guessing.

Every stored value keeps receipts. Judged candidates are cached by message id
so rebuilds are incremental — suitable for a dead-hours batch job.

Output lives in data/unsynced/user_profiles.json (gitignored: real people).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from utils import chat_archive, message_quality

DEFAULT_PATH = Path("data/unsynced/user_profiles.json")
# Bump when the judge prompt or filters change semantics — the judged-message
# cache is keyed on it, so old verdicts are re-judged instead of trusted.
VERSION = 2

# Each slot: retrieval anchor phrases (FTS, exact phrase match), the judge
# question (interpolated into the prompt), and whether multiple values can be
# simultaneously true (hobbies yes, age no).
SLOTS: dict[str, dict] = {
    "location": {
        "question": "where the speaker lives or is from (country, region, or city)",
        "anchors": ["i live in", "i'm from", "im from", "i am from", "here in",
                    "i moved to", "in my country", "my country"],
        "multi": False,
    },
    "age": {
        "question": "the speaker's age or birth year",
        "anchors": ["years old", "i was born in", "just turned", "my age is",
                    "when i was your age", "my birthday"],
        "multi": False,
    },
    "gender": {
        "question": "the speaker's gender",
        "anchors": ["i'm a guy", "im a guy", "i'm a girl", "im a girl",
                    "i'm a dude", "im a dude", "as a man", "as a woman",
                    "i'm male", "im male", "i'm female", "im female"],
        "multi": False,
    },
    "occupation": {
        "question": "the speaker's job, profession, or what they study",
        "anchors": ["i work as", "i work at", "i work in", "my job", "my boss",
                    "my shift", "i study", "i'm studying", "im studying",
                    "my degree", "my major", "in uni", "in college", "my thesis"],
        "multi": False,
    },
    "relationship": {
        "question": "the speaker's relationship status (partner, married, single)",
        "anchors": ["my gf", "my girlfriend", "my bf", "my boyfriend", "my wife",
                    "my husband", "i'm single", "im single", "my fiancee",
                    "we got married", "my ex"],
        "multi": False,
    },
    "pets": {
        "question": "a pet the speaker owns",
        "anchors": ["my dog", "my cat", "my pet", "my puppy", "my kitten"],
        "multi": True,
    },
    "hobbies": {
        "question": "a hobby, game, sport, or interest the speaker actively does",
        "anchors": ["my hobby", "i main", "i've been playing", "ive been playing",
                    "i started playing", "my favorite game", "i go to the gym",
                    "i lift", "i draw", "i produce", "i collect", "i practice"],
        "multi": True,
    },
    "languages": {
        "question": "a language the speaker speaks natively or is learning",
        "anchors": ["my native language", "my first language", "i speak",
                    "i'm learning", "im learning", "my english"],
        "multi": True,
    },
    "family": {
        "question": "a family member of the speaker (sibling, parent, child)",
        "anchors": ["my brother", "my sister", "my mom", "my dad", "my mother",
                    "my father", "my son", "my daughter", "my kids"],
        "multi": True,
    },
}

_JUDGE_SYSTEM = (
    "You are a careful information extractor working on Twitch chat logs. "
    "Twitch chat is dense with jokes, sarcasm, memes, copypasta, roleplay, and "
    "quoting other people. Your job is to reject all of that and only extract "
    "facts a speaker plainly and sincerely asserts about THEMSELVES. "
    "When in doubt, do not extract."
)

_JUDGE_USER_TEMPLATE = """Chat moment from #{channel} ('>>' marks the speaker, {author}):

{context}

Question: does the marked message sincerely assert {question} about the speaker themselves?

Respond with ONLY a JSON object, no other text:
{{"asserts": true/false, "value": "<short normalized value or null>", "sincerity": "sincere"/"joke"/"unclear"}}

Rules:
- asserts=false if the statement is about someone else, hypothetical, negated, a quote, song lyric, meme, or copypasta (overly dramatic, story-shaped, or too-perfect lines are usually copypasta).
- asserts=false unless the message states the actual value: an anecdote that merely involves the topic (e.g. mentions their boss without saying what the job is) does NOT assert it.
- value: 1-6 words, lowercase, the direct answer to the question (e.g. "poland", "software developer", "27").
- sincerity="joke" for obvious bits/irony even if phrased as fact."""


# ------------------------------ retrieval ---------------------------------

def _said_by_others(content: str, author: str) -> bool:
    """Archive-grounded copypasta check: if any OTHER chatter has posted the
    same line (normalized), it's a meme circulating in chat, not a personal
    fact — no LLM judgment needed. Uses the first words as an FTS anchor and
    line_match_key for the exact comparison."""
    words = re.findall(r"[A-Za-z0-9']+", content or "")
    if len(words) < 5:
        return False  # too short to anchor; leave it to the judge
    canonical = chat_archive.normalize_author(author)
    key = chat_archive.line_match_key(content)
    if not key:
        return False
    try:
        conn = chat_archive.connect()
        rows = conn.execute(
            "SELECT m.author, m.content FROM messages_fts f "
            "JOIN messages m ON m.id = f.rowid "
            "WHERE f.messages_fts MATCH ? LIMIT 40",
            (chat_archive._fts_phrase(" ".join(words[:8])),),
        ).fetchall()
    except Exception:
        return False
    for row_author, row_content in rows:
        if chat_archive.normalize_author(row_author) == canonical:
            continue
        if chat_archive.line_match_key(row_content or "") == key:
            return True
    return False


def candidate_rows(author: str, slot: str, per_anchor: int = 6,
                   cap: int = 14) -> list[dict]:
    """Recent-first, deduped candidate messages for one (author, slot)."""
    spec = SLOTS[slot]
    conn = chat_archive.connect()
    keys = chat_archive.author_keys(author)
    ph, params = chat_archive._in_clause(keys)
    seen_keys: set = set()
    out: list[dict] = []
    for anchor in spec["anchors"]:
        try:
            rows = conn.execute(
                "SELECT m.id, m.sent_at, m.channel, m.content "
                "FROM messages_fts f JOIN messages m ON m.id = f.rowid "
                f"WHERE f.messages_fts MATCH ? AND m.author IN ({ph}) "
                "ORDER BY m.sent_at DESC LIMIT ?",
                [chat_archive._fts_phrase(anchor), *params, per_anchor],
            ).fetchall()
        except Exception as e:
            logging.debug(f"candidate_rows({author},{slot},{anchor!r}) failed: {e}")
            continue
        for mid, sent_at, channel, content in rows:
            content = content or ""
            if content.lstrip().startswith("~") or message_quality.command_like(content):
                continue
            k = chat_archive.line_match_key(content)
            if not k or k in seen_keys:
                continue
            seen_keys.add(k)
            if _said_by_others(content, author):
                continue
            out.append({"id": mid, "sent_at": sent_at, "channel": channel,
                        "content": content})
    out.sort(key=lambda r: r["sent_at"], reverse=True)
    return out[:cap]


def _context_block(row: dict, author: str) -> str:
    """The candidate line ±2 surrounding lines, '>>'-marked like persona RAG."""
    lines = []
    try:
        window = chat_archive.context_window(row["id"], row["channel"],
                                             before=2, after=2)
    except Exception:
        window = []
    if not window:
        window = [(row["id"], author, row["content"])]
    for mid, row_author, content in window:
        mark = ">>" if mid == row["id"] else "  "
        lines.append(f"{mark} {row_author}: {content}")
    return "\n".join(lines)


# ------------------------------ judgment ----------------------------------

def _parse_judgment(text: str | None) -> dict | None:
    """Extract the judge's JSON verdict; tolerate chatter around it."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(data, dict) or "asserts" not in data:
        return None
    value = data.get("value")
    if isinstance(value, (int, float)):
        value = str(value)
    if value is not None and not isinstance(value, str):
        return None
    return {
        "asserts": bool(data["asserts"]),
        "value": (value or "").strip().casefold() or None,
        "sincerity": str(data.get("sincerity", "unclear")).casefold(),
    }


def default_llm_call(system: str, user: str) -> str | None:
    """Synchronous wrapper over the async local-LLM client, for batch scripts."""
    import asyncio
    from services import llm
    return asyncio.run(llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=160, temperature=0.1,
    ))


def judge_candidate(author: str, slot: str, row: dict, llm_call) -> dict | None:
    prompt = _JUDGE_USER_TEMPLATE.format(
        channel=row["channel"], author=author,
        context=_context_block(row, author),
        question=SLOTS[slot]["question"],
    )
    verdict = _parse_judgment(llm_call(_JUDGE_SYSTEM, prompt))
    if verdict is None:
        return None
    verdict.update({"id": row["id"], "sent_at": row["sent_at"],
                    "channel": row["channel"], "content": row["content"]})
    return verdict


# ---------------------------- reconciliation ------------------------------

_AGE_RE = re.compile(r"\b(19\d\d|20\d\d|\d{1,2})\b")

# Judge outputs that are grammatically values but semantically non-answers
# (compared AFTER _norm_value strips possessives/articles).
_GENERIC_VALUES = {
    "country", "here", "home", "abroad", "somewhere",
    "native language", "language", "job", "work", "boss",
    "unknown", "none", "n/a", "null", "true", "yes", "unclear",
}


def _valid_value(value: str) -> bool:
    """Reject placeholder and anecdote-shaped values. A slot value names a
    thing ("germany", "software developer", "27") — it is never a clause about
    an event ("still working on july 2nd") or a self-reference ("my country")."""
    words = value.split()
    if not words or len(words) > 4:
        return False
    if value in _GENERIC_VALUES:
        return False
    if words[0] in {"i", "im", "i'm"}:
        return False
    return True


def _norm_value(slot: str, value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "")).strip().casefold()
    if slot == "age":
        m = _AGE_RE.search(value)
        if m:
            return m.group(1)
    # "my father" / "has a brother" / "a dog" must group with "father" /
    # "brother" / "dog" — corroboration counts VALUES, not phrasings.
    value = re.sub(r"^(?:i\s+)?(?:has|have|had|owns?)\s+", "", value)
    value = re.sub(r"^(?:my|a|an|the)\s+", "", value)
    return value


def _ages_compatible(a: str, b: str) -> bool:
    """People age; adjacent ages (or birth-year vs age) shouldn't conflict."""
    try:
        ia, ib = int(a), int(b)
    except Exception:
        return a == b
    if ia > 1900 or ib > 1900:  # birth year vs age — don't call it a dispute
        return True
    return abs(ia - ib) <= 2


def _reconcile(slot: str, judged: list[dict]) -> dict | None:
    """Group sincere assertions by value; corroboration = independent days."""
    kept = [j for j in judged
            if j and j.get("asserts") and j.get("value")
            and j.get("sincerity") == "sincere"
            and _valid_value(_norm_value(slot, j["value"]))]
    # A personal running gag repeats VERBATIM across days (copypasta-of-one);
    # a real fact gets re-stated in fresh words. Identical lines count once,
    # so verbatim repetition alone can never reach "confirmed".
    deduped, seen_lines = [], set()
    for j in kept:
        line_key = chat_archive.line_match_key(j.get("content") or "") or j.get("content")
        if line_key in seen_lines:
            continue
        seen_lines.add(line_key)
        deduped.append(j)
    kept = deduped
    if not kept:
        return None
    groups: dict[str, list[dict]] = defaultdict(list)
    for j in kept:
        groups[_norm_value(slot, j["value"])].append(j)

    def days(rows):
        return len({r["sent_at"][:10] for r in rows})

    # most independent days first; ties broken by most recent assertion
    ranked = sorted(groups.items(),
                    key=lambda kv: max(r["sent_at"] for r in kv[1]), reverse=True)
    ranked.sort(key=lambda kv: -days(kv[1]))

    def entry(value, rows, status):
        rows = sorted(rows, key=lambda r: r["sent_at"])
        return {
            "value": value,
            "status": status,
            "supports": len(rows),
            "days": days(rows),
            "first_seen": rows[0]["sent_at"],
            "last_seen": rows[-1]["sent_at"],
            "evidence": [{"sent_at": r["sent_at"], "channel": r["channel"],
                          "text": r["content"][:200]} for r in rows[-4:]],
        }

    if SLOTS[slot]["multi"]:
        values = [entry(v, rows, "confirmed" if days(rows) >= 2 else "candidate")
                  for v, rows in ranked]
        return {"values": values}

    top_value, top_rows = ranked[0]
    status = "confirmed" if days(top_rows) >= 2 else "candidate"
    alternatives = []
    for v, rows in ranked[1:]:
        if slot == "age" and _ages_compatible(v, top_value):
            continue
        alt_status = "confirmed" if days(rows) >= 2 else "candidate"
        if alt_status == "confirmed" and status == "confirmed":
            status = "disputed"
        alternatives.append(entry(v, rows, alt_status))
    result = entry(top_value, top_rows, status)
    if alternatives:
        result["alternatives"] = alternatives[:3]
    return result


# ------------------------------- pipeline ---------------------------------

def build_profiles(authors: list[str], *, llm_call=None, per_slot_cap: int = 14,
                   path: Path = DEFAULT_PATH, progress=None) -> dict:
    """Build/refresh profiles for `authors`. Incremental: previously judged
    message ids are cached and skipped, so re-runs only pay for new evidence."""
    llm_call = llm_call or default_llm_call
    store = load(path) or {"_meta": {}, "profiles": {}, "judged": {}}
    judged_cache: dict = store.setdefault("judged", {})

    for author in [chat_archive.normalize_author(a) for a in authors]:
        profile: dict = {}
        for slot in SLOTS:
            rows = candidate_rows(author, slot, cap=per_slot_cap)
            judged = []
            for row in rows:
                cache_key = f"v{VERSION}|{author}|{slot}|{row['id']}"
                if cache_key in judged_cache:
                    judged.append(judged_cache[cache_key])
                    continue
                verdict = judge_candidate(author, slot, row, llm_call)
                if verdict is None:
                    continue  # LLM down/unparseable — retry next run
                judged_cache[cache_key] = verdict
                judged.append(verdict)
                if progress:
                    progress(author, slot, verdict)
            resolved = _reconcile(slot, judged)
            if resolved:
                profile[slot] = resolved
            save(store, path)  # judge calls are the expense — never lose them
        store["profiles"][author] = profile

    import time
    store["_meta"] = {"version": VERSION, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "authors": sorted(store["profiles"])}
    save(store, path)
    return store


# ------------------------------- read side --------------------------------

_store_cache = None
_store_mtime = None


def load(path: Path = DEFAULT_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning(f"user_profiles load failed: {e}")
        return None


def save(store: dict, path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=1),
                    encoding="utf-8")


def profile_for(author: str, path: Path = DEFAULT_PATH,
                min_status: str = "candidate") -> dict:
    global _store_cache, _store_mtime
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    if _store_cache is None or mtime != _store_mtime:
        _store_cache = load(path) or {}
        _store_mtime = mtime  # a rebuild reaches the live bot without restart
    profiles = _store_cache.get("profiles", {})
    prof = profiles.get(chat_archive.normalize_author(author), {})
    if min_status == "confirmed":
        out = {}
        for slot, data in prof.items():
            if "values" in data:
                vals = [v for v in data["values"] if v["status"] == "confirmed"]
                if vals:
                    out[slot] = {"values": vals}
            elif data.get("status") == "confirmed":
                out[slot] = data
        return out
    return prof


def profile_line(author: str, max_len: int = 220) -> str:
    """Compact one-line summary for prompt injection ('' when nothing known).
    Confirmed facts only — a persona prompt must not launder guesses."""
    prof = profile_for(author, min_status="confirmed")
    bits = []
    for slot in SLOTS:  # stable order
        data = prof.get(slot)
        if not data:
            continue
        if "values" in data:
            vals = ", ".join(v["value"] for v in data["values"][:3])
            bits.append(f"{slot}: {vals}")
        else:
            bits.append(f"{slot}: {data['value']}")
    line = " · ".join(bits)
    return line[:max_len]
