"""Slot-based, LLM-verified user profiles - profile v5.

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
3. CORROBORATION: ordinary values need >=2 independent days; unusual values
   need >=3 and impossible readings never auto-confirm. Single sightings stay
   "candidate". Conflicting confirmed values become "disputed".

Every stored value keeps receipts. Judged candidates are cached by message id
so rebuilds are incremental — suitable for a dead-hours batch job.

Output lives in data/unsynced/user_profiles.json (gitignored: real people).
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
from collections import defaultdict
from pathlib import Path

from utils import atomic_file, chat_archive, message_quality

DEFAULT_PATH = Path("data/unsynced/user_profiles.json")
# Bump when the judge prompt or filters change semantics — the judged-message
# cache is keyed on it, so old verdicts are re-judged instead of trusted.
VERSION = 5


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.partial{path.suffix}")

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

_LANGUAGE_NAMES = {
    "arabic", "chinese", "czech", "english", "french", "german", "greek",
    "hungarian", "italian", "japanese", "korean", "latin", "polish",
    "portuguese", "russian", "slovak", "spanish", "turkish", "ukrainian",
    "vietnamese",
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
{{"asserts": true/false, "value": "<short normalized value or null>", "sincerity": "sincere"/"joke"/"unclear", "plausibility": "ordinary"/"unusual"/"impossible"/"unclear"}}

Rules:
- asserts=false if the statement is about someone else, hypothetical, negated, a quote, song lyric, meme, or copypasta (overly dramatic, story-shaped, or too-perfect lines are usually copypasta).
- asserts=false unless the message states the actual value: an anecdote that merely involves the topic (e.g. mentions their boss without saying what the job is) does NOT assert it.
- value: 1-6 words, lowercase, the direct answer to the question (e.g. "poland", "software developer", "27").
- sincerity="joke" for obvious bits/irony even if phrased as fact.
- plausibility is only a base-rate warning. Use "unusual" for possible but surprising claims and "impossible" only for literal impossibilities; never reject a claim merely because it is unusual."""


# ------------------------------ retrieval ---------------------------------


def _candidate_has_explicit_value(slot: str, content: str) -> bool:
    """Cheap precision gate before paying the contextual judge.

    Slot anchors are intentionally broad for recall, but lines such as "my
    boss", "if I speak", or "25 years old" about a game do not contain a
    self-profile value. The model prompt already requires a value in the marked
    message, so removing these rows loses no evidence the judge should accept.
    """
    text = message_quality.clean_text(content or "").casefold().replace("’", "'")
    if not text:
        return False

    if slot == "location":
        match = re.search(
            r"\b(?:i live in|i(?:'m| am|m) from|i moved to|"
            r"i(?:'m| am|m) here in)\s+([^,.!?]{2,60})",
            text,
        )
        if not match:
            return False
        value = match.group(1).strip()
        placeholders = (
            "my country", "your country", "this country", "that country",
            "middle of nowhere", "the middle of nowhere", "here", "there",
            "somewhere", "nowhere", "the other side", "other side",
        )
        return not value.startswith(placeholders)

    if slot == "age":
        return bool(re.search(
            r"\b(?:i(?:'m| am|m)\s+\d{1,3}\s+years? old|"
            r"i was born in\s+(?:19|20)\d{2}|i just turned\s+\d{1,3}|"
            r"my age is\s+\d{1,3})",
            text,
        ))

    if slot == "gender":
        return bool(re.search(
            r"\bi(?:'m| am|m)\s+(?:a\s+)?(?:guy|girl|dude|man|woman|male|female)\b",
            text,
        ))

    if slot == "occupation":
        match = re.search(
            r"\b(?:i work (?:as|at|in|for)\s+\S+|"
            r"i (?:study|am studying|'m studying|m studying)\s+\S+|"
            r"my (?:degree|major|thesis)(?: is| in| on)?\s+\S+|"
            r"i(?:'m| am|m) in (?:uni|college))",
            text,
        )
        if not match:
            return False
        return not any(phrase in text for phrase in (
            "work in that field", "work in this field", "work at that place",
            "work at this place",
        ))

    if slot == "relationship":
        return bool(re.search(
            r"\b(?:my (?:gf|girlfriend|bf|boyfriend|wife|husband|fiancee|ex)\b|"
            r"i(?:'m| am|m) single\b|we got married\b)",
            text,
        ))

    if slot == "pets":
        return bool(re.search(r"\bmy (?:dog|cat|pet|puppy|kitten)\b", text))

    if slot == "hobbies":
        direct = re.search(
            r"\b(?:my hobby(?: is)?\s+\S+|i main\s+\S+|"
            r"i started playing\s+\S+|my favorite game(?: is)?\s+\S+|"
            r"i go to the gym\b|i (?:lift|draw|produce|collect|practice)\b)",
            text,
        )
        if direct:
            return True
        playing = re.search(r"\bi(?:'ve| have|ve) been playing\s+(.+)", text)
        if not playing:
            return False
        value = playing.group(1).strip()
        return not value.startswith(("for ", "since ", "this ", "it ", "a lot"))

    if slot == "languages":
        if re.search(r"\bmy english\b", text):
            return True
        match = re.search(
            r"\b(?:my (?:native|first) language(?: is)?|i speak|"
            r"i(?:'m| am|m) learning)\s+([^,.!?]+)",
            text,
        )
        if not match:
            return False
        value = match.group(1).strip()
        value_words = set(re.findall(r"[a-z]+", value)[:4])
        return bool(value_words & _LANGUAGE_NAMES)

    if slot == "family":
        return bool(re.search(
            r"\bmy (?:brother|sister|mom|dad|mother|father|son|daughter|kids)\b",
            text,
        ))
    return False


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
        author_keys = chat_archive.author_keys(canonical)
        placeholders, params = chat_archive._in_clause(author_keys)
        rows = conn.execute(
            "SELECT m.author, m.content FROM messages_fts f "
            "CROSS JOIN messages m ON m.id = f.rowid "
            f"WHERE f.messages_fts MATCH ? AND m.author NOT IN ({placeholders}) LIMIT 40",
            [chat_archive._fts_phrase(" ".join(words[:8])), *params],
        ).fetchall()
    except Exception:
        return False
    for row_author, row_content in rows:
        if chat_archive.normalize_author(row_author) == canonical:
            continue
        if chat_archive.line_match_key(row_content or "") == key:
            return True
    return False


def candidate_rows(author: str, slot: str, per_anchor: int = 12,
                   cap: int = 30) -> list[dict]:
    """Timeline-diverse, deduped candidate messages for one fact slot."""
    spec = SLOTS[slot]
    conn = chat_archive.connect()
    canonical = chat_archive.normalize_author(author)
    keys = chat_archive.author_keys(author)
    ph, params = chat_archive._in_clause(keys)
    bounds = conn.execute(
        f"SELECT MIN(id), MAX(id) FROM messages WHERE author IN ({ph})",
        params,
    ).fetchone()
    min_id, max_id = (int(bounds[0] or 0), int(bounds[1] or 0)) if bounds else (0, 0)
    seen_keys: set = set()
    out: list[dict] = []
    for anchor in spec["anchors"]:
        query = chat_archive._fts_phrase(anchor)
        recent_n = max(1, (per_anchor + 1) // 2)
        oldest_n = max(1, per_anchor // 4)
        spread_n = max(1, per_anchor - recent_n - oldest_n)
        try:
            recent = conn.execute(
                "SELECT m.id, m.sent_at, m.channel, m.content "
                "FROM messages_fts f CROSS JOIN messages m ON m.id = f.rowid "
                f"WHERE f.messages_fts MATCH ? AND m.author IN ({ph}) "
                "ORDER BY m.sent_at DESC LIMIT ?",
                [query, *params, recent_n],
            ).fetchall()
            oldest = conn.execute(
                "SELECT m.id, m.sent_at, m.channel, m.content "
                "FROM messages_fts f CROSS JOIN messages m ON m.id = f.rowid "
                f"WHERE f.messages_fts MATCH ? AND m.author IN ({ph}) "
                "ORDER BY m.sent_at, m.id LIMIT ?",
                [query, *params, oldest_n],
            ).fetchall()
            spread = []
            if max_id > min_id:
                digest = hashlib.blake2b(
                    f"{canonical}|{slot}|{anchor}".encode("utf-8"), digest_size=8
                ).digest()
                pivot = min_id + int.from_bytes(digest, "big") % (max_id - min_id + 1)
                spread = conn.execute(
                    "SELECT m.id, m.sent_at, m.channel, m.content "
                    "FROM messages_fts f CROSS JOIN messages m ON m.id=f.rowid "
                    f"WHERE f.messages_fts MATCH ? AND m.author IN ({ph}) "
                    "AND m.id>=? LIMIT ?",
                    [query, *params, pivot, spread_n],
                ).fetchall()
                if len(spread) < spread_n:
                    spread.extend(conn.execute(
                        "SELECT m.id, m.sent_at, m.channel, m.content "
                        "FROM messages_fts f CROSS JOIN messages m ON m.id=f.rowid "
                        f"WHERE f.messages_fts MATCH ? AND m.author IN ({ph}) "
                        "AND m.id<? LIMIT ?",
                        [query, *params, pivot, spread_n - len(spread)],
                    ).fetchall())
            rows = recent + oldest + spread
        except Exception as e:
            logging.debug(f"candidate_rows({author},{slot},{anchor!r}) failed: {e}")
            continue
        for mid, sent_at, channel, content in rows:
            content = content or ""
            if content.lstrip().startswith("~") or message_quality.command_like(content):
                continue
            if message_quality.likely_pasted_prose(content):
                continue
            if not _candidate_has_explicit_value(slot, content):
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
    # First pass spans independent days instead of spending the whole budget
    # on one recent conversation; second pass fills remaining capacity.
    selected = []
    selected_ids = set()
    seen_days = set()
    for row in out:
        day = row["sent_at"][:10]
        if day in seen_days:
            continue
        selected.append(row)
        selected_ids.add(row["id"])
        seen_days.add(day)
        if len(selected) >= cap:
            return selected
    for row in out:
        if row["id"] in selected_ids:
            continue
        selected.append(row)
        if len(selected) >= cap:
            break
    return selected


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

def _normalize_judgment(data) -> dict | None:
    if not isinstance(data, dict) or "asserts" not in data:
        return None
    value = data.get("value")
    if isinstance(value, (int, float)):
        value = str(value)
    if value is not None and not isinstance(value, str):
        return None
    plausibility = str(data.get("plausibility", "unclear")).casefold()
    if plausibility not in {"ordinary", "unusual", "impossible", "unclear"}:
        plausibility = "unclear"
    sincerity = str(data.get("sincerity", "unclear")).casefold()
    if sincerity not in {"sincere", "joke", "unclear"}:
        sincerity = "unclear"
    asserts_raw = data["asserts"]
    asserts = asserts_raw is True or (
        isinstance(asserts_raw, str) and asserts_raw.casefold() == "true"
    )
    return {
        "asserts": asserts,
        "value": (value or "").strip().casefold() or None,
        "sincerity": sincerity,
        "plausibility": plausibility,
    }


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
    return _normalize_judgment(data)


def default_llm_call(system: str, user: str) -> str | None:
    """Synchronous wrapper over the async local-LLM client, for batch scripts."""
    import asyncio
    from services import llm
    return asyncio.run(llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=700, temperature=0.1,
    ))


def require_model_dependency() -> None:
    """Fail before archive retrieval when the configured chat model is absent."""
    import asyncio
    from services import llm

    if not asyncio.run(llm.available()):
        raise RuntimeError("local model server is unavailable")
    response = asyncio.run(llm.chat(
        [{"role": "user", "content": "Reply with exactly OK."}],
        max_tokens=3,
        temperature=0.0,
    ))
    if not response:
        raise RuntimeError(f"local judge model is unavailable: {llm.last_error()}")


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


def judge_candidates_batch(
    author: str,
    slot: str,
    rows: list[dict],
    llm_call,
) -> dict[int, dict]:
    """Judge several candidates in one model call, keyed by message id."""
    if not rows:
        return {}
    blocks = []
    for index, row in enumerate(rows, 1):
        blocks.append(
            f"--- ITEM {index} ---\n" + _context_block(row, author)
        )
    prompt = (
        f"Each numbered chat moment is from #{rows[0]['channel']} or another listed channel; "
        f"'>>' marks {author}. For every item, decide whether the marked line sincerely "
        f"asserts {SLOTS[slot]['question']} about the speaker themselves.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn ONLY JSON with exactly one result per item: "
          '{"items":[{"index":1,"asserts":true,"value":"short value or null",'
          '"sincerity":"sincere|joke|unclear",'
          '"plausibility":"ordinary|unusual|impossible|unclear"}]}. '
          "Reject statements about others, hypotheticals, negations, quotes, memes, "
          "copypasta, and anecdotes that do not state the value. Unusual means possible "
          "but surprising; do not reject solely for being unusual."
    )
    raw = llm_call(_JUDGE_SYSTEM, prompt)
    if not raw:
        return {}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except Exception:
        return {}
    out = {}
    for item in data.get("items", []) if isinstance(data, dict) else []:
        try:
            index = int(item.get("index")) - 1
        except (AttributeError, TypeError, ValueError):
            continue
        if not 0 <= index < len(rows):
            continue
        verdict = _normalize_judgment(item)
        if verdict is None:
            continue
        row = rows[index]
        verdict.update({
            "id": row["id"],
            "sent_at": row["sent_at"],
            "channel": row["channel"],
            "content": row["content"],
        })
        out[row["id"]] = verdict
    return out


# ---------------------------- reconciliation ------------------------------

_AGE_RE = re.compile(r"\b(19\d\d|20\d\d|\d{1,2})\b")

# Judge outputs that are grammatically values but semantically non-answers
# (compared AFTER _norm_value strips possessives/articles).
_GENERIC_VALUES = {
    "country", "here", "home", "abroad", "somewhere",
    "native language", "language", "job", "work", "boss",
    "unknown", "none", "n/a", "null", "true", "yes", "unclear",
}

_OCCUPATION_WORDS = {
    "accountant", "analyst", "artist", "chef", "consultant", "contractor",
    "developer", "doctor", "driver", "engineer", "lawyer", "manager",
    "mechanic", "nurse", "operator", "programmer", "researcher", "student",
    "teacher", "technician", "trader", "writer",
}
_FAMILY_VALUES = {
    "brother", "sister", "mom", "dad", "mother", "father", "son", "daughter",
    "kid", "kids",
}
_PET_VALUES = {"cat", "dog", "pet", "puppy", "kitten"}


def _valid_value(value: str, *, slot: str | None = None) -> bool:
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
    if any(word in {"because", "when", "since"} for word in words):
        return False
    if slot == "location" and value in {
        "room next door", "next door", "countryside", "green region",
        "other side", "middle of nowhere",
    }:
        return False
    if slot == "gender" and value not in {"male", "female", "nonbinary"}:
        return False
    if slot == "relationship" and not re.search(
        r"\b(?:single|girlfriend|boyfriend|wife|wives|husband|husbands|fiancee|ex)\b",
        value,
    ):
        return False
    if slot == "pets" and value not in _PET_VALUES:
        return False
    if slot == "family" and value not in _FAMILY_VALUES:
        return False
    if slot == "languages" and value not in _LANGUAGE_NAMES:
        return False
    if slot == "occupation" and len(words) == 1 and words[0] not in _OCCUPATION_WORDS:
        return False
    return True


def _norm_value(slot: str, value: str) -> str:
    value = message_quality.clean_text(value or "").strip().casefold()
    try:
        from utils import emote_meaning, persona_classifier

        emotes = {
            name.casefold() for name in emote_meaning.registry()
            if persona_classifier._is_emote_token(name)
        }
        value = " ".join(word for word in value.split() if word.casefold() not in emotes)
    except Exception:
        pass
    if slot == "age":
        m = _AGE_RE.search(value)
        if m:
            return m.group(1)
    if slot == "gender":
        if re.search(r"\b(?:girl|woman|female)\b", value):
            return "female"
        if re.search(r"\b(?:guy|dude|man|male)\b", value):
            return "male"
        if re.search(r"\b(?:nonbinary|non-binary)\b", value):
            return "nonbinary"
    if slot == "relationship":
        multiple = re.search(r"\b(\d+|two|three)\s+(wives|husbands)\b", value)
        if multiple:
            number = {"two": "2", "three": "3"}.get(multiple.group(1), multiple.group(1))
            return f"{number} {multiple.group(2)}"
        for term in (
            "girlfriend", "boyfriend", "wife", "husband", "fiancee", "single", "ex",
        ):
            if re.search(rf"\b{term}\b", value):
                return term
    if slot == "pets":
        for term in ("cat", "dog", "puppy", "kitten", "pet"):
            if re.search(rf"\b{term}\b", value):
                return {"puppy": "dog", "kitten": "cat"}.get(term, term)
    if slot == "family":
        for term in sorted(_FAMILY_VALUES):
            if re.search(rf"\b{term}\b", value):
                return term
    if slot == "languages":
        for term in sorted(_LANGUAGE_NAMES):
            if re.search(rf"\b{term}\b", value):
                return term
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
            and _valid_value(_norm_value(slot, j["value"]), slot=slot)]
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

    def plausibility(rows):
        # Use the most cautious base-rate label across supporting receipts. A
        # single impossible/unusual read should demand more evidence, not be
        # voted away by two optimistic model outputs.
        rank = {"ordinary": 0, "unclear": 1, "unusual": 2, "impossible": 3}
        labels = [row.get("plausibility", "unclear") for row in rows]
        return max(labels, key=lambda label: rank.get(label, 1)) if labels else "unclear"

    def status_for(rows):
        prior = plausibility(rows)
        if prior == "impossible":
            return "candidate"
        required_days = 2 if prior == "ordinary" else 3
        return "confirmed" if days(rows) >= required_days else "candidate"

    # most independent days first; ties broken by most recent assertion
    ranked = sorted(groups.items(),
                    key=lambda kv: max(r["sent_at"] for r in kv[1]), reverse=True)
    ranked.sort(key=lambda kv: -days(kv[1]))

    def entry(value, rows, status):
        rows = sorted(rows, key=lambda r: r["sent_at"])
        return {
            "value": value,
            "status": status,
            "plausibility": plausibility(rows),
            "supports": len(rows),
            "days": days(rows),
            "first_seen": rows[0]["sent_at"],
            "last_seen": rows[-1]["sent_at"],
            "evidence": [{"sent_at": r["sent_at"], "channel": r["channel"],
                          "text": r["content"][:200]} for r in rows[-4:]],
        }

    if SLOTS[slot]["multi"]:
        values = [entry(v, rows, status_for(rows))
                  for v, rows in ranked]
        return {"values": values}

    top_value, top_rows = ranked[0]
    status = status_for(top_rows)
    alternatives = []
    for v, rows in ranked[1:]:
        if slot == "age" and _ages_compatible(v, top_value):
            continue
        alt_status = status_for(rows)
        if alt_status == "confirmed" and status == "confirmed":
            status = "disputed"
        alternatives.append(entry(v, rows, alt_status))
    result = entry(top_value, top_rows, status)
    if alternatives:
        result["alternatives"] = alternatives[:3]
    return result


# ------------------------------- pipeline ---------------------------------

def build_profiles(
    authors: list[str],
    *,
    llm_call=None,
    per_slot_cap: int = 30,
    batch_size: int = 6,
    path: Path = DEFAULT_PATH,
    progress=None,
) -> dict:
    """Build/refresh profiles for `authors`. Incremental: previously judged
    message ids are cached and skipped, so re-runs only pay for new evidence."""
    llm_call = llm_call or default_llm_call
    checkpoint_path = _partial_path(path)
    store = load(checkpoint_path) or load(path) or {
        "_meta": {}, "profiles": {}, "judged": {}
    }
    meta = store.get("_meta") or {}
    if (
        meta.get("version") != VERSION
        or meta.get("alias_signature") != chat_archive.alias_signature()
    ):
        store = {"_meta": {}, "profiles": {}, "judged": {}}
    judged_cache: dict = store.setdefault("judged", {})
    canonical_authors = []
    for author in authors:
        canonical = chat_archive.normalize_author(author)
        if canonical and canonical not in canonical_authors:
            canonical_authors.append(canonical)
    candidate_total = 0
    judged_total = 0
    completed_slots = 0
    run_meta = store.setdefault("_meta", {})
    run_meta.update({
        "build_complete": False,
        "requested_authors": canonical_authors,
        "requested_slots": len(canonical_authors) * len(SLOTS),
        "completed_slots": 0,
        "candidate_rows": 0,
        "judged_candidate_rows": 0,
    })
    _checkpoint(store, checkpoint_path)

    for author in canonical_authors:
        profile: dict = {}
        for slot in SLOTS:
            rows = candidate_rows(author, slot, cap=per_slot_cap)
            candidate_total += len(rows)
            judged = []
            pending = []
            for row in rows:
                cache_key = f"v{VERSION}|{author}|{slot}|{row['id']}"
                if cache_key in judged_cache:
                    judged.append(judged_cache[cache_key])
                    judged_total += 1
                    continue
                pending.append(row)
            chunk_size = max(1, int(batch_size))
            for start in range(0, len(pending), chunk_size):
                chunk = pending[start:start + chunk_size]
                verdicts = judge_candidates_batch(author, slot, chunk, llm_call)
                for row in chunk:
                    verdict = verdicts.get(row["id"])
                    if verdict is None:
                        # Retry only a result omitted from malformed batch JSON.
                        verdict = judge_candidate(author, slot, row, llm_call)
                    if verdict is None:
                        continue  # model down/unparseable; retry next run
                    cache_key = f"v{VERSION}|{author}|{slot}|{row['id']}"
                    judged_cache[cache_key] = verdict
                    judged.append(verdict)
                    judged_total += 1
                    if progress:
                        progress(author, slot, verdict)
            resolved = _reconcile(slot, judged)
            if resolved:
                profile[slot] = resolved
            completed_slots += int(len(judged) == len(rows))
            store["profiles"][author] = dict(profile)
            store.setdefault("_meta", {}).update({
                "candidate_rows": candidate_total,
                "judged_candidate_rows": judged_total,
                "completed_slots": completed_slots,
            })
            # Judge calls are the expense; checkpoint privately without
            # replacing the last complete live artifact.
            _checkpoint(store, checkpoint_path)

    store.setdefault("_meta", {}).update({
        "candidate_rows": candidate_total,
        "judged_candidate_rows": judged_total,
        "completed_slots": completed_slots,
    })
    if canonical_authors and candidate_total == 0:
        _checkpoint(store, checkpoint_path)
        raise RuntimeError("retrieval returned zero profile candidates")
    if judged_total != candidate_total:
        _checkpoint(store, checkpoint_path)
        raise RuntimeError(
            f"profile judge covered {judged_total}/{candidate_total} candidates; "
            "checkpoint preserved for retry"
        )
    store["_meta"]["build_complete"] = True
    _checkpoint(store, path)
    try:
        checkpoint_path.unlink()
    except FileNotFoundError:
        pass
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
    with atomic_file.open_atomic(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(store, ensure_ascii=False, indent=1))


def _checkpoint(store: dict, path: Path = DEFAULT_PATH) -> None:
    import time

    meta = dict(store.get("_meta") or {})
    meta.update({
        "version": VERSION,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "authors": sorted(store.get("profiles") or {}),
        "alias_signature": chat_archive.alias_signature(),
    })
    store["_meta"] = meta
    save(store, path)


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
    meta = _store_cache.get("_meta") or {}
    if (
        meta.get("version") != VERSION
        or meta.get("alias_signature") != chat_archive.alias_signature()
    ):
        return {}
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


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}


def claims_in_text(text: str) -> list[dict]:
    """Small high-precision claim parser used by the irony evidence layer."""
    out = []
    age = re.search(
        r"\b(?:i am|i'm|im)\s+(\d{1,3})(?:\s+years?\s+old)?\b",
        text or "", re.I,
    )
    if age:
        value = age.group(1)
        number = int(value)
        prior = "impossible" if number > 125 else ("unusual" if number < 13 or number > 70 else "ordinary")
        out.append({"slot": "age", "value": value, "plausibility": prior})

    relationship = re.search(r"\b(?:i am|i'm|im)\s+(single|married)\b", text or "", re.I)
    if relationship:
        out.append({
            "slot": "relationship",
            "value": relationship.group(1).casefold(),
            "plausibility": "ordinary",
        })
    spouses = re.search(
        r"\bi\s+(?:have|got)\s+(\d+|one|two|three|four|five|six|seven|eight|nine)\s+"
        r"(wives|husbands|spouses)\b",
        text or "", re.I,
    )
    if spouses:
        raw_n = spouses.group(1).casefold()
        number = int(raw_n) if raw_n.isdigit() else _NUMBER_WORDS[raw_n]
        out.append({
            "slot": "relationship",
            "value": f"{number} {spouses.group(2).casefold()}",
            "plausibility": "unusual" if number >= 2 else "ordinary",
        })
    return out


def claim_consistency(author: str, text: str) -> dict:
    claims = claims_in_text(text)
    confirmed = profile_for(author, min_status="confirmed") if author else {}
    conflicts = []
    agreements = []
    for claim in claims:
        known = confirmed.get(claim["slot"])
        if not known:
            continue
        values = (
            [entry["value"] for entry in known.get("values", [])]
            if "values" in known else [known["value"]]
        )
        value = claim["value"]
        if claim["slot"] == "age":
            matches = any(_ages_compatible(value, other) for other in values)
        elif value in {"single", "married"}:
            matches = any(value == other for other in values)
        else:
            # Counts such as "two wives" are not contradicted merely by a
            # profile saying "married"; only a known single status conflicts.
            matches = not any(other == "single" for other in values)
        (agreements if matches else conflicts).append({
            "slot": claim["slot"], "claim": value, "known": values[:3]
        })
    return {"claims": claims, "conflicts": conflicts, "agreements": agreements}


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
