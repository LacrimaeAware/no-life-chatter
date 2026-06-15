"""SQLite chat archive: historical Chatterino logs + live bot capture.

One database (config.ARCHIVE_DB) holds every archived chat message, with an
FTS5 full-text index so "did user X ever say Y?" is answered instantly and
locally — no API. See docs/CHAT_ARCHIVE.md for the design and the verified
Chatterino log-format spec this parser implements.
"""

import logging
import os
import re
import sqlite3
import threading
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

import config

# Name-spelling variants that should count as the same channel/user. The legacy
# [archive.aliases] map applies to both; the newer specific maps avoid turning
# a user alt-account merge into a channel rename.
ALIASES = {alias.lower(): real.lower() for alias, real in config.ARCHIVE_ALIASES.items()}
USER_ALIASES = {
    **ALIASES,
    **{alias.lower(): real.lower() for alias, real in config.ARCHIVE_USER_ALIASES.items()},
}
CHANNEL_ALIASES = {
    **ALIASES,
    **{alias.lower(): real.lower() for alias, real in config.ARCHIVE_CHANNEL_ALIASES.items()},
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id       INTEGER PRIMARY KEY,
    channel  TEXT NOT NULL,
    author   TEXT NOT NULL,
    sent_at  TEXT NOT NULL,            -- 'YYYY-MM-DD HH:MM:SS' local time
    content  TEXT NOT NULL,
    source   TEXT NOT NULL DEFAULT 'chatterino',
    src_path TEXT                      -- originating log file (NULL for live)
);
CREATE INDEX IF NOT EXISTS idx_msg_src ON messages(src_path);
CREATE INDEX IF NOT EXISTS idx_msg_author  ON messages(author, sent_at);
CREATE INDEX IF NOT EXISTS idx_msg_channel ON messages(channel, sent_at);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content='messages', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TABLE IF NOT EXISTS ingested_files (
    path  TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    rows  INTEGER NOT NULL
);
"""

_conn = None
_thread_state = threading.local()


def _base_name(name: str) -> str:
    name = (name or "").strip().lstrip("@").rstrip(",").lower()
    return name


def _resolve_alias(name: str, aliases: dict) -> str:
    """Resolve aliases, including simple chains like typo -> old -> current."""
    name = _base_name(name)
    seen = set()
    while name in aliases and name not in seen:
        seen.add(name)
        name = _base_name(aliases[name])
    return name


def normalize(name: str) -> str:
    """Lowercase, strip @ / whitespace, and resolve legacy aliases."""
    return _resolve_alias(name, ALIASES)


def normalize_author(name: str) -> str:
    """Normalize a chatter username, including user-only alt-account aliases."""
    return _resolve_alias(name, USER_ALIASES)


def normalize_channel(name: str) -> str:
    """Normalize a channel name, including channel-only aliases."""
    return _resolve_alias(name, CHANNEL_ALIASES)


def author_keys(author: str) -> list[str]:
    """All stored author keys that should count as this person.

    Existing rows may still be stored under pre-alias names, so reads query the
    canonical name plus any configured aliases that resolve to it.
    """
    canonical = normalize_author(author)
    keys = {canonical, _base_name(author)}
    for alias in USER_ALIASES:
        if _resolve_alias(alias, USER_ALIASES) == canonical:
            keys.add(alias)
    return sorted(keys)


def _in_clause(values: list[str]) -> tuple[str, list[str]]:
    placeholders = ",".join("?" for _ in values)
    return placeholders, values


_MATCH_TRANSLATION = str.maketrans({
    "’": "'",
    "‘": "'",
    "´": "'",
    "`": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "—": "-",
    "–": "-",
    "‐": "-",
    "\u00a0": " ",
})
_EXACT_DEDUPE_SCAN_LIMIT = 5000


def line_match_key(text: str) -> str:
    """Normalize chat text for copy/near-copy checks.

    This intentionally treats curly vs straight apostrophes, punctuation, case,
    and spacing as irrelevant. It is stricter than "same meaning", but looser
    than byte-for-byte equality, which is what we need for copied chat lines.
    """
    text = unicodedata.normalize("NFKC", text or "").translate(_MATCH_TRANSLATION)
    text = text.casefold()
    text = text.replace("'", "")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_time_bucket(sent_at: str) -> str:
    """Minute bucket for import-duplicate suppression.

    The archive can contain the same raw line twice a few seconds apart from
    overlapping imports. Keep repeated memes on different minutes/days, but
    collapse same author/channel/text inside one minute.
    """
    return (sent_at or "")[:16]


def _dedupe_search_rows(rows, *, author_index: int | None,
                        channel_index: int, content_index: int):
    seen = set()
    out = []
    for row in rows:
        content_key = line_match_key(row[content_index])
        if not content_key:
            continue
        author_key = normalize_author(row[author_index]) if author_index is not None else ""
        key = (
            author_key,
            normalize_channel(row[channel_index]),
            content_key,
            _dedupe_time_bucket(row[0]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def line_similarity(left: str, right: str) -> float:
    left_key = line_match_key(left)
    right_key = line_match_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    token_dice = (2 * overlap) / (len(left_tokens) + len(right_tokens))
    char_ratio = SequenceMatcher(None, left_key, right_key, autojunk=False).ratio()
    # A very close substring is usually the same chat line with a mention,
    # emote, or small tail added/removed.
    shorter, longer = sorted((left_key, right_key), key=len)
    substring_bonus = len(shorter) >= 24 and shorter in longer
    if substring_bonus:
        char_ratio = max(char_ratio, 0.97)
    score = (char_ratio * 0.75) + (token_dice * 0.25)
    if substring_bonus:
        score = max(score, 0.97)
    return score


def connect() -> sqlite3.Connection:
    global _conn
    conn = getattr(_thread_state, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(config.ARCHIVE_DB), exist_ok=True)
        # WAL + generous busy timeout so the bot's live writes and in-chat
        # queries keep working while an ingest run holds long write
        # transactions in another process.
        conn = sqlite3.connect(config.ARCHIVE_DB, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        _thread_state.conn = conn
        if _conn is None:
            _conn = conn
    return conn


# ----------------------- Chatterino log parsing -----------------------
# Format spec: docs/CHAT_ARCHIVE.md. One IRC message per line; time-only
# timestamps (date lives in the filename); headers repeat mid-file.

CHAT_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] ([a-z0-9_]+): (.*)$")
SYS_TS_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] ")
FILE_DATE_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})\.log$")
# Moderator actions logged as '<mod>: <mod> timed out <user> for 7d. ' look
# exactly like chat lines; filter the narrow case where the body restates the
# author followed by a moderation verb. Same trick for third-party emote
# changes ('x: x removed 7TV emote Foo.'), which Chatterino logs identically.
MOD_MASQ_RE = re.compile(r"^(?P<a>[a-z0-9_]+) (timed out|untimedout|banned|unbanned) ")
EMOTE_MASQ_RE = re.compile(
    r"^(?P<a>[a-z0-9_]+) (added|removed|updated|renamed|enabled|disabled) "
    r"(?:the )?(?:7TV|BTTV|FFZ) emote ",
    re.IGNORECASE,
)


def parse_line(line: str, author_filter=None):
    """Classify one log line.

    Returns ('chat', time, author, content) for a real chat message, or
    ('header'|'system'|'empty'|'modaction', None, None, None) otherwise.
    """
    if not line or line == "\n":
        return ("empty", None, None, None)
    line = line.rstrip("\n")
    if line.startswith("# "):
        return ("header", None, None, None)
    m = CHAT_RE.match(line)
    if m:
        t, author, content = m.group(1), m.group(2), m.group(3)
        content = content.strip()
        if not content:
            return ("empty", None, None, None)
        mm = MOD_MASQ_RE.match(content)
        if mm and mm.group("a") == author:
            return ("modaction", None, None, None)
        em = EMOTE_MASQ_RE.match(content)
        if em and em.group("a").lower() == author:
            return ("modaction", None, None, None)
        return ("chat", t, author, content)
    if SYS_TS_RE.match(line):
        return ("system", None, None, None)
    return ("system", None, None, None)


def ingest_file(path: str, channel: str, date: str) -> dict:
    """Parse one Chatterino log file into the archive. Idempotent via ledger."""
    conn = connect()
    # normcase so Windows path-casing differences can't bypass the ledger
    # (which would silently duplicate the whole archive on re-run).
    key = os.path.normcase(os.path.abspath(path))
    mtime = os.path.getmtime(path)
    row = conn.execute("SELECT mtime FROM ingested_files WHERE path = ?", (key,)).fetchone()
    if row and row[0] == mtime:
        return {"skipped": True}

    counts = {"chat": 0, "header": 0, "system": 0, "empty": 0, "modaction": 0}
    batch = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            kind, t, author, content = parse_line(line)
            counts[kind] += 1
            if kind == "chat":
                batch.append((channel, normalize_author(author), f"{date} {t}", content,
                              "chatterino", key))

    with conn:
        if row:
            # File changed since last ingest (today's still-growing log):
            # replace exactly this file's rows. Scoping by src_path — not by
            # channel+date — keeps alias-merged sibling directories intact.
            conn.execute("DELETE FROM messages WHERE src_path = ?", (key,))
        conn.executemany(
            "INSERT INTO messages (channel, author, sent_at, content, source, src_path) "
            "VALUES (?,?,?,?,?,?)",
            batch,
        )
        conn.execute(
            "INSERT OR REPLACE INTO ingested_files (path, mtime, rows) VALUES (?,?,?)",
            (key, mtime, counts["chat"]),
        )
    counts["skipped"] = False
    return counts


# ----------------------------- live capture -----------------------------

_seen_ids = set()


def record_author_id(author: str, twitch_id) -> None:
    """Names change, ids don't — keep an author->twitch_id table current from
    live chat (message.author.id), once per author per process. Future
    identity merging is id-dominant for anyone the bot has ever seen."""
    if not author or not twitch_id:
        return
    key = (author.lower(), str(twitch_id))
    if key in _seen_ids:
        return
    try:
        conn = connect()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS author_ids ("
            " author TEXT PRIMARY KEY, twitch_id TEXT, checked_at TEXT)")
        conn.execute(
            "INSERT OR REPLACE INTO author_ids VALUES (?,?,datetime('now'))",
            (author.lower(), str(twitch_id)))
        conn.commit()
        _seen_ids.add(key)
    except Exception as e:
        logging.debug(f"record_author_id failed: {e}")


_live_backlog = []


def record_live(channel: str, author: str, content: str, sent_at: str) -> None:
    """Append one live chat message. Never raises into the bot's message path.
    If a bulk import holds the writer lock past the busy timeout, the row is
    buffered and flushed on a later message instead of silently lost."""
    _live_backlog.append((channel, author, content, sent_at))
    try:
        conn = connect()
        with conn:
            while _live_backlog:
                _record_one(conn, *_live_backlog[0])
                _live_backlog.pop(0)
        return
    except sqlite3.OperationalError:
        if len(_live_backlog) > 5000:
            del _live_backlog[:1000]
        logging.warning(f"archive busy; {len(_live_backlog)} live messages buffered")
        return
    except Exception as e:
        logging.warning(f"record_live failed: {e}")
        _live_backlog.pop()
        return


def _record_one(conn, channel: str, author: str, content: str, sent_at: str) -> None:
    conn.execute(
        "INSERT INTO messages (channel, author, sent_at, content, source) VALUES (?,?,?,?, 'live')",
        (normalize_channel(channel), normalize_author(author), sent_at, content),
    )


# ------------------------------- queries --------------------------------

def _fts_phrase(phrase: str) -> str:
    """Quote user text as a single FTS5 phrase (handles internal quotes)."""
    return '"' + phrase.replace('"', '""') + '"'


_QUERY_STOPWORDS = {
    "able", "after", "all", "also", "and", "any", "are", "around", "ask",
    "asked", "asking", "been", "both", "but", "can", "did", "does", "done",
    "for", "get", "gets", "got", "had", "has", "him", "his", "its", "off",
    "one", "our", "out", "she", "the", "too", "was",
    "about", "after", "again", "against", "also", "because", "been", "before",
    "being", "between", "could", "does", "doing", "dont", "from", "have",
    "here", "into", "just", "like", "more", "most", "much", "need", "only",
    "line", "lines", "message", "messages", "over", "really", "said",
    "same", "should", "some", "stuff", "test", "than", "that", "their",
    "them", "then", "there", "these", "they", "thing", "this", "very",
    "want", "were", "what", "when", "where", "which", "while", "with",
    "would", "write", "wrote", "your", "youre",
    # Bot/persona scaffolding words are common in commands but poor retrieval
    # anchors for a chatter's actual voice.
    "bot", "chat", "hyper", "mimic", "persona", "twitch",
    # Question/smalltalk scaffolding. Lesson from the LoRA/RAG smoke tests:
    # for "hows world of warcraft been treating you?" retrieval matched the
    # QUESTION's words (hows/treating/you) and returned the author asking
    # other people how things are going — query-term echo. Only topic words
    # should anchor retrieval.
    "hows", "whats", "whos", "whens", "wheres", "whys", "why", "who",
    "going", "treating", "think", "thinks", "thinking", "thought", "thoughts",
    "answer", "answers", "felt", "feel", "feels", "see", "seen", "seeing",
    "such", "tell", "tells", "telling", "says", "saying", "thats", "well",
    "know", "knows", "knew", "mean", "means", "meant",
    "you", "yall", "guys", "dude", "anyone", "everyone", "someone",
    "something", "anything", "nothing", "ever", "never", "still", "even",
    "yeah", "yep", "nah", "lol", "lmao", "omegalul", "kekw",
    "gonna", "wanna", "kinda", "sorta", "honestly", "literally", "actually",
    "right", "okay", "today", "tonight", "yesterday", "tomorrow",
}


def _emote_like_token(raw: str) -> bool:
    """True for tokens that look like emote names, not words.

    CamelCase with an internal capital (FeelsOkayMan, PauseChamp) or a long
    all-caps mash (WAYTOODANK). These dominate FTS ranking when left in the
    query — the smoke tests retrieved 'PauseChamp ?' as "relevant" purely
    because the live context contained the emote.
    """
    if re.search(r"[a-z][A-Z]", raw):
        return True
    return len(raw) >= 6 and raw.isupper() and raw.isalpha()

_QUERY_ALIASES = {
    "world of warcraft": ["wow"],
    "world warcraft": ["wow"],
    "league of legends": ["league"],
    "counter strike": ["counterstrike"],
    "counter-strike": ["counterstrike"],
}


def query_terms(text: str, max_terms: int = 12, exclude_terms=None) -> list[str]:
    """Topic terms worth anchoring retrieval on, from natural chat text.

    Drops @mentions, emote-like tokens, stopwords/question scaffolding, and
    any caller-supplied excludes (usernames in the conversation). What's left
    is the actual subject matter.
    """
    exclude_terms = {t.lower() for t in (exclude_terms or set())}
    counts = Counter()
    first_seen = {}
    cleaned = re.sub(r"@\w+", " ", text or "")  # pings are addressing, not topic
    for raw in re.findall(r"\w+", cleaned, flags=re.UNICODE):
        if _emote_like_token(raw.strip("_")):
            continue
        term = raw.strip("_").lower()
        if len(term) < 3 or term in _QUERY_STOPWORDS or term in exclude_terms:
            continue
        if not any(ch.isalpha() for ch in term):
            continue
        counts[term] += 1
        first_seen.setdefault(term, len(first_seen))
    lowered = (text or "").lower()
    for phrase, aliases in _QUERY_ALIASES.items():
        if phrase in lowered:
            for alias in aliases:
                if alias not in exclude_terms:
                    counts[alias] += 1
                    first_seen.setdefault(alias, len(first_seen))
    return sorted(counts, key=lambda t: (-counts[t], first_seen[t]))[:max_terms]


def _fts_query(text: str, max_terms: int = 12, exclude_terms=None) -> str | None:
    """Build a safe, broad FTS query from natural chat text.

    FTS5 treats spaces as AND, which is too strict for noisy live chat context,
    so we OR a small set of useful terms. Each term is quoted to keep user text
    out of the FTS query syntax.
    """
    terms = query_terms(text, max_terms=max_terms, exclude_terms=exclude_terms)
    if not terms:
        return None
    return " OR ".join(_fts_phrase(term) for term in terms)


def search_author_hits(author: str, text: str, limit: int = 40,
                       max_chars: int = 240, exclude_terms=None):
    """Relevant messages by one author for natural-language text, with row ids.

    Returns [(id, sent_at, channel, content), ...] ranked by FTS5 bm25. The id
    lets callers expand a hit into its surrounding chat moment. The FTS table
    stays first in the query plan; see said() for why CROSS JOIN matters.
    """
    keys = author_keys(author)
    excludes = set(keys) | {t.lower() for t in (exclude_terms or set())}
    q = _fts_query(text, exclude_terms=excludes)
    if not q:
        return []
    author_placeholders, author_params = _in_clause(keys)

    conn = connect()
    try:
        return conn.execute(
            "SELECT m.id, m.sent_at, m.channel, m.content FROM messages_fts f "
            "CROSS JOIN messages m ON m.id = f.rowid "
            f"WHERE f.messages_fts MATCH ? AND m.author IN ({author_placeholders}) "
            "AND length(m.content) <= ? "
            "AND ltrim(m.content) NOT LIKE ? "
            "ORDER BY bm25(messages_fts), m.sent_at DESC LIMIT ?",
            [q, *author_params, max_chars, config.PREFIX + "%", limit],
        ).fetchall()
    except sqlite3.OperationalError as e:
        logging.warning(f"chat_archive search_author_hits failed for {author!r}: {e}")
        return []


def search_author(author: str, text: str, limit: int = 40,
                  max_chars: int = 240):
    """Back-compat wrapper: [(sent_at, channel, content), ...]."""
    return [
        (sent_at, channel, content)
        for _, sent_at, channel, content in search_author_hits(
            author, text, limit=limit, max_chars=max_chars
        )
    ]


def _source_needs_context_coverage(source: str, src_path: str | None) -> bool:
    """True when adjacent rows may be from a single-speaker mirror export."""
    source = (source or "").lower()
    path = os.path.normcase(src_path or "").replace("\\", "/").lower()
    return source == "zonian" or "/external_logs/zonian/raw/" in path


def _has_multi_author_coverage(conn, channel: str, sent_at: str, author: str,
                               within_minutes: int) -> bool:
    """Whether a source-limited hit has nearby rows from another chatter."""
    try:
        rows = conn.execute(
            "SELECT author FROM messages WHERE channel = ? "
            "AND sent_at >= datetime(?, ?) AND sent_at <= datetime(?, ?) "
            "LIMIT 250",
            (
                normalize_channel(channel), sent_at, f"-{int(within_minutes)} minutes",
                sent_at, f"+{int(within_minutes)} minutes",
            ),
        ).fetchall()
    except Exception:
        return False
    hit_author = normalize_author(author)
    return any(normalize_author(a) != hit_author for a, in rows)


def _dedupe_context_rows(rows, hit_id: int | None = None):
    """Drop alias-collapsed near-duplicate lines while preserving the hit."""
    hit_key = None
    if hit_id is not None:
        for row_id, row_author, row_content in rows:
            if row_id == hit_id:
                key = line_match_key(row_content)
                if key:
                    hit_key = (normalize_author(row_author), key)
                break

    out, seen = [], set()
    for row_id, row_author, row_content in rows:
        key_text = line_match_key(row_content)
        if not key_text:
            out.append((row_id, row_author, row_content))
            continue
        key = (normalize_author(row_author), key_text)
        if hit_key and key == hit_key and row_id != hit_id:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append((row_id, row_author, row_content))
    return out


def context_window(message_id: int, channel: str, before: int = 2, after: int = 2,
                   within_minutes: int = 20, require_multi_author: bool = True,
                   dedupe: bool = True):
    """The chat moment around one message: [(id, author, content), ...] in
    order, including the message itself. Lets retrieval show what a line was
    responding to instead of the line in isolation.

    For single-speaker mirror logs, surrounding rows are only treated as
    conversation when the same channel/time slice contains another chatter too.
    Otherwise returning neighbors would invent context from an author-only log.
    """
    conn = connect()
    hit = conn.execute(
        "SELECT id, channel, author, content, sent_at, source, src_path "
        "FROM messages WHERE id = ?",
        (message_id,),
    ).fetchall()
    if not hit:
        return []
    hit_id, hit_channel, hit_author, hit_content, ts, source, src_path = hit[0]
    chan = normalize_channel(hit_channel or channel)
    hit_row = (hit_id, hit_author, hit_content)
    if (
        require_multi_author
        and _source_needs_context_coverage(source, src_path)
        and not _has_multi_author_coverage(conn, chan, ts, hit_author, within_minutes)
    ):
        return [hit_row]

    # Order by TIME, not row id: bulk imports interleave sources, so id
    # neighbors can be from a different month. (sent_at, id) breaks ties for
    # same-second messages. Bound by time so sparse imports don't jump sessions.
    prev = conn.execute(
        "SELECT id, author, content FROM messages WHERE channel = ? "
        "AND sent_at >= datetime(?, ?) "
        "AND (sent_at < ? OR (sent_at = ? AND id < ?)) "
        "ORDER BY sent_at DESC, id DESC LIMIT ?",
        (chan, ts, f"-{int(within_minutes)} minutes", ts, ts, message_id, before),
    ).fetchall()
    nxt = conn.execute(
        "SELECT id, author, content FROM messages WHERE channel = ? "
        "AND sent_at <= datetime(?, ?) "
        "AND (sent_at > ? OR (sent_at = ? AND id > ?)) "
        "ORDER BY sent_at, id LIMIT ?",
        (chan, ts, f"+{int(within_minutes)} minutes", ts, ts, message_id, after),
    ).fetchall()
    rows = list(reversed(prev)) + [hit_row] + nxt
    return _dedupe_context_rows(rows, hit_id=hit_id) if dedupe else rows


def _said_legacy(author: str, phrase: str, limit: int = 3):
    """All matches of phrase by author: (total_count, [(sent_at, channel, content)...]).

    CROSS JOIN forces SQLite to run the FTS match once and probe messages by
    rowid; the plain-JOIN plan flips the loop order and re-evaluates the FTS
    match per author row (measured: 79s vs 0.003s on a 741k-row archive).
    """
    conn = connect()
    keys = author_keys(author)
    author_placeholders, author_params = _in_clause(keys)
    searchable = re.sub(r"[^0-9A-Za-z]+", " ", phrase).strip()
    if not searchable:
        # Emoji/symbol-only phrase: the FTS tokenizer would drop everything
        # and report a confident 0 — substring search answers it correctly.
        like = "%" + phrase.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_") + "%"
        total = conn.execute(
            f"SELECT COUNT(*) FROM messages WHERE author IN ({author_placeholders}) "
            r"AND content LIKE ? ESCAPE '\'",
            [*author_params, like],
        ).fetchone()[0]
        rows = conn.execute(
            f"SELECT sent_at, channel, content FROM messages "
            f"WHERE author IN ({author_placeholders}) "
            r"AND content LIKE ? ESCAPE '\' ORDER BY sent_at LIMIT ?",
            [*author_params, like, limit],
        ).fetchall()
        return total, rows
    q = _fts_phrase(phrase)
    total = conn.execute(
        "SELECT COUNT(*) FROM messages_fts f CROSS JOIN messages m ON m.id = f.rowid "
        f"WHERE f.messages_fts MATCH ? AND m.author IN ({author_placeholders})",
        [q, *author_params],
    ).fetchone()[0]
    rows = conn.execute(
        "SELECT m.sent_at, m.channel, m.content FROM messages_fts f "
        "CROSS JOIN messages m ON m.id = f.rowid "
        f"WHERE f.messages_fts MATCH ? AND m.author IN ({author_placeholders}) "
        "ORDER BY m.sent_at LIMIT ?",
        [q, *author_params, limit],
    ).fetchall()
    return total, rows


def _nearest_author_lines_legacy(author: str, phrase: str, limit: int = 3,
                                 min_score: float = 0.82):
    """Closest lines by author after punctuation/case/spacing normalization.

    Intended as a fallback for "did they basically say this?" It does not
    replace exact `said()` answers; callers should show exact matches first.
    Returns [(score, sent_at, channel, content), ...].
    """
    needle = line_match_key(phrase)
    if not needle or len(needle) < 8:
        return []
    needle_tokens = set(needle.split())
    conn = connect()
    keys = author_keys(author)
    author_placeholders, author_params = _in_clause(keys)
    rows = conn.execute(
        f"SELECT sent_at, channel, content FROM messages "
        f"WHERE author IN ({author_placeholders})",
        author_params,
    ).fetchall()

    scored = []
    for sent_at, channel, content in rows:
        hay = line_match_key(content)
        if not hay:
            continue
        max_len = max(len(needle), len(hay))
        if max_len and abs(len(needle) - len(hay)) / max_len > 0.65:
            continue
        hay_tokens = set(hay.split())
        if not hay_tokens:
            continue
        overlap = len(needle_tokens & hay_tokens)
        if overlap == 0:
            continue
        overlap_share = overlap / min(len(needle_tokens), len(hay_tokens))
        if overlap_share < 0.45 and needle not in hay and hay not in needle:
            continue
        score = line_similarity(phrase, content)
        if score >= min_score:
            scored.append((score, sent_at, channel, content))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored[:limit]


def _command_filter(include_commands: bool, alias: str = "m") -> tuple[str, list[str]]:
    if include_commands:
        return "", []
    return f" AND ltrim({alias}.content) NOT LIKE ? ", [config.PREFIX + "%"]


def _channel_filter(channel: str = None, alias: str = "m") -> tuple[str, list[str]]:
    if not channel:
        return "", []
    return f" AND {alias}.channel = ? ", [normalize_channel(channel)]


def said(author: str, phrase: str, limit: int = 3, offset: int = 0,
         channel: str = None, include_commands: bool = False):
    """All matches of phrase by author: (total_count, [(sent_at, channel, content)...])."""
    conn = connect()
    keys = author_keys(author)
    author_placeholders, author_params = _in_clause(keys)
    chan_sql, chan_params = _channel_filter(channel)
    cmd_sql, cmd_params = _command_filter(include_commands)
    searchable = re.sub(r"[^0-9A-Za-z]+", " ", phrase).strip()
    if not searchable:
        like = "%" + phrase.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_") + "%"
        raw_total = conn.execute(
            f"SELECT COUNT(*) FROM messages m WHERE m.author IN ({author_placeholders}) "
            f"{chan_sql}{cmd_sql}"
            r"AND m.content LIKE ? ESCAPE '\'",
            [*author_params, *chan_params, *cmd_params, like],
        ).fetchone()[0]
        if raw_total <= _EXACT_DEDUPE_SCAN_LIMIT:
            all_rows = conn.execute(
                f"SELECT m.sent_at, m.channel, m.content FROM messages m "
                f"WHERE m.author IN ({author_placeholders}) "
                f"{chan_sql}{cmd_sql}"
                r"AND m.content LIKE ? ESCAPE '\' ORDER BY m.sent_at",
                [*author_params, *chan_params, *cmd_params, like],
            ).fetchall()
            unique = _dedupe_search_rows(
                all_rows, author_index=None, channel_index=1, content_index=2
            )
            return len(unique), unique[offset:offset + limit]
        rows = conn.execute(
            f"SELECT m.sent_at, m.channel, m.content FROM messages m "
            f"WHERE m.author IN ({author_placeholders}) "
            f"{chan_sql}{cmd_sql}"
            r"AND m.content LIKE ? ESCAPE '\' ORDER BY m.sent_at LIMIT ? OFFSET ?",
            [*author_params, *chan_params, *cmd_params, like, limit * 5, offset],
        ).fetchall()
        rows = _dedupe_search_rows(rows, author_index=None, channel_index=1, content_index=2)
        return raw_total, rows[:limit]
    q = _fts_phrase(phrase)
    raw_total = conn.execute(
        "SELECT COUNT(*) FROM messages_fts f CROSS JOIN messages m ON m.id = f.rowid "
        f"WHERE f.messages_fts MATCH ? AND m.author IN ({author_placeholders}) "
        f"{chan_sql}{cmd_sql}",
        [q, *author_params, *chan_params, *cmd_params],
    ).fetchone()[0]
    if raw_total <= _EXACT_DEDUPE_SCAN_LIMIT:
        all_rows = conn.execute(
            "SELECT m.sent_at, m.channel, m.content FROM messages_fts f "
            "CROSS JOIN messages m ON m.id = f.rowid "
            f"WHERE f.messages_fts MATCH ? AND m.author IN ({author_placeholders}) "
            f"{chan_sql}{cmd_sql}"
            "ORDER BY m.sent_at",
            [q, *author_params, *chan_params, *cmd_params],
        ).fetchall()
        unique = _dedupe_search_rows(
            all_rows, author_index=None, channel_index=1, content_index=2
        )
        return len(unique), unique[offset:offset + limit]
    rows = conn.execute(
        "SELECT m.sent_at, m.channel, m.content FROM messages_fts f "
        "CROSS JOIN messages m ON m.id = f.rowid "
        f"WHERE f.messages_fts MATCH ? AND m.author IN ({author_placeholders}) "
        f"{chan_sql}{cmd_sql}"
        "ORDER BY m.sent_at LIMIT ? OFFSET ?",
        [q, *author_params, *chan_params, *cmd_params, limit * 5, offset],
    ).fetchall()
    rows = _dedupe_search_rows(rows, author_index=None, channel_index=1, content_index=2)
    return raw_total, rows[:limit]


def nearest_author_lines(author: str, phrase: str, limit: int = 3,
                         min_score: float = 0.82, channel: str = None,
                         include_commands: bool = False):
    """Closest lines by author after punctuation/case/spacing normalization."""
    needle = line_match_key(phrase)
    if not needle or len(needle) < 8:
        return []
    needle_tokens = set(needle.split())
    conn = connect()
    keys = author_keys(author)
    author_placeholders, author_params = _in_clause(keys)
    chan_sql, chan_params = _channel_filter(channel, alias="m")
    cmd_sql, cmd_params = _command_filter(include_commands)
    rows = conn.execute(
        f"SELECT m.sent_at, m.channel, m.content FROM messages m "
        f"WHERE m.author IN ({author_placeholders}){chan_sql}{cmd_sql}",
        [*author_params, *chan_params, *cmd_params],
    ).fetchall()

    scored = []
    for sent_at, channel, content in rows:
        hay = line_match_key(content)
        if not hay:
            continue
        max_len = max(len(needle), len(hay))
        if max_len and abs(len(needle) - len(hay)) / max_len > 0.65:
            continue
        hay_tokens = set(hay.split())
        if not hay_tokens:
            continue
        overlap = len(needle_tokens & hay_tokens)
        if overlap == 0:
            continue
        overlap_share = overlap / min(len(needle_tokens), len(hay_tokens))
        if overlap_share < 0.45 and needle not in hay and hay not in needle:
            continue
        score = line_similarity(phrase, content)
        if score >= min_score:
            scored.append((score, sent_at, channel, content))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored[:limit]


def random_quote(author: str, min_words: int = 3):
    """One random, reasonably substantial message by author (or None)."""
    conn = connect()
    keys = author_keys(author)
    author_placeholders, author_params = _in_clause(keys)
    row = conn.execute(
        f"SELECT sent_at, channel, content FROM messages WHERE author IN ({author_placeholders}) "
        "AND length(content) - length(replace(content, ' ', '')) >= ? "
        "ORDER BY RANDOM() LIMIT 1",
        [*author_params, min_words - 1],
    ).fetchone()
    if row is None:  # fall back to any message at all
        row = conn.execute(
            f"SELECT sent_at, channel, content FROM messages WHERE author IN ({author_placeholders}) "
            "ORDER BY RANDOM() LIMIT 1",
            author_params,
        ).fetchone()
    return row


def first_seen(author: str):
    """Earliest archived message by author: (sent_at, channel, content) or None."""
    conn = connect()
    keys = author_keys(author)
    author_placeholders, author_params = _in_clause(keys)
    return conn.execute(
        f"SELECT sent_at, channel, content FROM messages WHERE author IN ({author_placeholders}) "
        "ORDER BY sent_at LIMIT 1",
        author_params,
    ).fetchone()


def stats(author: str, channel: str = None):
    """Summary numbers for author (optionally one channel), or None if unseen."""
    conn = connect()
    keys = author_keys(author)
    author_placeholders, author_params = _in_clause(keys)
    where = f"author IN ({author_placeholders})"
    params = list(author_params)
    if channel:
        where += " AND channel = ?"
        params.append(normalize_channel(channel))
    row = conn.execute(
        "SELECT COUNT(*), MIN(sent_at), MAX(sent_at), AVG(length(content)) "
        f"FROM messages WHERE {where}",
        params,
    ).fetchone()
    if not row or row[0] == 0:
        return None
    busiest = conn.execute(
        "SELECT substr(sent_at, 12, 2) AS hh, COUNT(*) AS n FROM messages "
        f"WHERE {where} GROUP BY hh ORDER BY n DESC LIMIT 1",
        params,
    ).fetchone()
    return {
        "messages": row[0],
        "first": row[1],
        "last": row[2],
        "avg_chars": round(row[3] or 0),
        "busiest_hour": int(busiest[0]) if busiest else None,
    }


# Generic platform bots only — add your own bot account and any
# community-specific bots via [persona] exclude_users in config.toml.
_DEFAULT_NOISE_USERS = {
    "automod",
    "mtgbot",
    "nightbot",
    "potatbotat",
    "streamelements",
    "streamlabs",
    "supibot",
}


def _is_noise_author(author: str) -> bool:
    author = normalize_author(author)
    excluded = {normalize_author(u) for u in getattr(config, "EXCLUDE_USERS", set())}
    return (
        author in excluded
        or author in _DEFAULT_NOISE_USERS
        or author.endswith("bot")
    )


def channel_regulars(channel: str, min_messages: int = 5000, limit: int = 20,
                     include_bots: bool = False):
    """Top non-bot authors in a channel, alias-collapsed.

    Returns [(author, count), ...]. Counts are canonicalized through
    archive.user_aliases, so alt accounts merge for this rollup.
    """
    conn = connect()
    channel = normalize_channel(channel)  # spelling variants via [archive.aliases]
    rows = conn.execute(
        "SELECT author, COUNT(*) FROM messages WHERE channel = ? GROUP BY author",
        (channel,),
    ).fetchall()
    counts = Counter()
    for author, count in rows:
        canon = normalize_author(author)
        if not include_bots and _is_noise_author(canon):
            continue
        counts[canon] += count
    return [(a, n) for a, n in counts.most_common(limit) if n >= min_messages]


def context_before(channel: str, sent_at: str, n: int = 4, within_minutes: int = 15):
    """The up-to-n messages just before sent_at in channel (oldest first).

    Reconstructs what a message was responding to — a line like "exactly" or
    "I am black" is meaningless alone but clear with the two lines above it.
    Bounded by a time window so it doesn't reach back into an unrelated session.
    """
    conn = connect()
    # 'YYYY-MM-DD HH:MM:SS' sorts and subtracts lexically only within a day;
    # for a short window, datetime() math in SQL is exact and simple.
    rows = conn.execute(
        "SELECT sent_at, author, content FROM messages "
        "WHERE channel = ? AND sent_at < ? "
        "AND sent_at >= datetime(?, ?) "
        "ORDER BY sent_at DESC LIMIT ?",
        (normalize_channel(channel), sent_at, sent_at, f"-{within_minutes} minutes", n),
    ).fetchall()
    return list(reversed(rows))


def latest(channel: str, n: int = 25):
    """The last n messages in a channel (oldest first) — live conversation
    context for a persona to react to. Includes live-captured messages."""
    conn = connect()
    rows = conn.execute(
        "SELECT sent_at, author, content FROM messages WHERE channel = ? "
        "ORDER BY id DESC LIMIT ?",
        (normalize_channel(channel), n),
    ).fetchall()
    return list(reversed(rows))


_members_cache = {}


def channel_members(channel: str, min_messages: int = 10):
    """Everyone with real history in a channel — its membership, regardless of
    whether they've spoken today. (~whosaid scopes to THE CHATROOM, not to
    whoever happened to talk in the last few minutes.) Cached per process."""
    key = (normalize_channel(channel), min_messages)
    if key not in _members_cache:
        conn = connect()
        rows = conn.execute(
            "SELECT author FROM messages WHERE channel = ? "
            "GROUP BY author HAVING COUNT(*) >= ?", key).fetchall()
        _members_cache[key] = {normalize_author(a) for a, in rows}
    return _members_cache[key]


def recent_authors(channel: str, scan: int = 400, limit: int = 60):
    """Distinct authors among the last `scan` messages of a channel — the pool
    of people currently around to mimic for a random reaction."""
    conn = connect()
    rows = conn.execute(
        "SELECT DISTINCT author FROM "
        "(SELECT author FROM messages WHERE channel = ? ORDER BY id DESC LIMIT ?)",
        (normalize_channel(channel), scan),
    ).fetchall()
    return [r[0] for r in rows][:limit]


def messages_for(author: str, channel: str = None, year: int = None):
    """All archived message texts by author (for offline persona building).
    channel limits to one chat, year to one calendar year (scoped ~markers)."""
    conn = connect()
    keys = author_keys(author)
    author_placeholders, author_params = _in_clause(keys)
    sql = f"SELECT content FROM messages WHERE author IN ({author_placeholders})"
    params = list(author_params)
    if channel:
        sql += " AND channel = ?"
        params.append(normalize_channel(channel))
    if year:
        sql += " AND sent_at >= ? AND sent_at < ?"
        params += [f"{int(year)}-01-01", f"{int(year) + 1}-01-01"]
    return [r[0] for r in conn.execute(sql + " ORDER BY sent_at", params).fetchall()]


def merge_utterances(rows, gap_seconds: int = 45):
    """Merge consecutive same-author messages within `gap_seconds` into single
    utterances. Chat fragments ("I wish" sent 3s before the punchline) are
    unreadable alone — semantic analysis should see the merged turn, not the
    fragment. rows = [(sent_at, author, content), ...] time-ordered;
    returns the same shape with contents joined.
    """
    import datetime
    def _t(s):
        try:
            return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    out = []
    for sent_at, author, content in rows:
        if out:
            p_sent, p_author, p_content = out[-1]
            t0, t1 = _t(p_sent), _t(sent_at)
            if (p_author == author and t0 and t1
                    and (t1 - t0).total_seconds() <= gap_seconds):
                out[-1] = (p_sent, p_author, p_content + " " + (content or ""))
                continue
        out.append((sent_at, author, content or ""))
    return out


def utterances_for(author: str, channel: str = None, year: int = None,
                   gap_seconds: int = 45):
    """The author's messages merged into utterances (see merge_utterances)."""
    conn = connect()
    keys = author_keys(author)
    ph, params = _in_clause(keys)
    sql = f"SELECT sent_at, author, content FROM messages WHERE author IN ({ph})"
    params = list(params)
    if channel:
        sql += " AND channel = ?"
        params.append(normalize_channel(channel))
    if year:
        sql += " AND sent_at >= ? AND sent_at < ?"
        params += [f"{int(year)}-01-01", f"{int(year) + 1}-01-01"]
    rows = conn.execute(sql + " ORDER BY sent_at, id", params).fetchall()
    return [c for _s, _a, c in merge_utterances(rows, gap_seconds)]


def _search_all_legacy(phrase: str, limit: int = 10):
    """Full-text search across all authors: [(sent_at, channel, author, content)...]."""
    conn = connect()
    return conn.execute(
        "SELECT m.sent_at, m.channel, m.author, m.content FROM messages_fts f "
        "CROSS JOIN messages m ON m.id = f.rowid WHERE f.messages_fts MATCH ? "
        "ORDER BY m.sent_at LIMIT ?",
        (_fts_phrase(phrase), limit),
    ).fetchall()


def search_all_count(phrase: str, channel: str = None,
                     include_commands: bool = False) -> int:
    """Count full-text matches across all authors."""
    conn = connect()
    chan_sql, chan_params = _channel_filter(channel)
    cmd_sql, cmd_params = _command_filter(include_commands)
    raw_total = conn.execute(
        "SELECT COUNT(*) FROM messages_fts f CROSS JOIN messages m ON m.id = f.rowid "
        f"WHERE f.messages_fts MATCH ? {chan_sql}{cmd_sql}",
        [_fts_phrase(phrase), *chan_params, *cmd_params],
    ).fetchone()[0]
    if raw_total > _EXACT_DEDUPE_SCAN_LIMIT:
        return raw_total
    rows = conn.execute(
        "SELECT m.sent_at, m.channel, m.author, m.content FROM messages_fts f "
        "CROSS JOIN messages m ON m.id = f.rowid "
        f"WHERE f.messages_fts MATCH ? {chan_sql}{cmd_sql}"
        "ORDER BY m.sent_at",
        [_fts_phrase(phrase), *chan_params, *cmd_params],
    ).fetchall()
    return len(_dedupe_search_rows(rows, author_index=2, channel_index=1, content_index=3))


def search_all(phrase: str, limit: int = 10, offset: int = 0,
               channel: str = None, include_commands: bool = False):
    """Full-text search across all authors: [(sent_at, channel, author, content)...]."""
    conn = connect()
    chan_sql, chan_params = _channel_filter(channel)
    cmd_sql, cmd_params = _command_filter(include_commands)
    raw_total = conn.execute(
        "SELECT COUNT(*) FROM messages_fts f CROSS JOIN messages m ON m.id = f.rowid "
        f"WHERE f.messages_fts MATCH ? {chan_sql}{cmd_sql}",
        [_fts_phrase(phrase), *chan_params, *cmd_params],
    ).fetchone()[0]
    if raw_total <= _EXACT_DEDUPE_SCAN_LIMIT:
        rows = conn.execute(
            "SELECT m.sent_at, m.channel, m.author, m.content FROM messages_fts f "
            "CROSS JOIN messages m ON m.id = f.rowid "
            f"WHERE f.messages_fts MATCH ? {chan_sql}{cmd_sql}"
            "ORDER BY m.sent_at",
            [_fts_phrase(phrase), *chan_params, *cmd_params],
        ).fetchall()
        unique = _dedupe_search_rows(rows, author_index=2, channel_index=1, content_index=3)
        return unique[offset:offset + limit]
    rows = conn.execute(
        "SELECT m.sent_at, m.channel, m.author, m.content FROM messages_fts f "
        "CROSS JOIN messages m ON m.id = f.rowid "
        f"WHERE f.messages_fts MATCH ? {chan_sql}{cmd_sql}"
        "ORDER BY m.sent_at LIMIT ? OFFSET ?",
        [_fts_phrase(phrase), *chan_params, *cmd_params, limit * 5, offset],
    ).fetchall()
    unique = _dedupe_search_rows(rows, author_index=2, channel_index=1, content_index=3)
    return unique[:limit]


def random_match(phrase: str, author: str = None, channel: str = None,
                 include_commands: bool = False, min_words: int = 3):
    """A RANDOM archived message matching `phrase` (full-text), optionally by
    `author` and/or in `channel`. Returns (sent_at, channel, author, content) or
    None. Skips bot commands and prefers messages with >= min_words real words.

    ORDER BY RANDOM() sorts the whole FTS-matched set, so cost scales with the
    phrase's frequency: a normal word is ~instant, a very common one ("the") is a
    few hundred ms. Per-user command cooldowns guard against spam; if that proves
    insufficient, bound the candidate pool with a capped subquery first."""
    q = _fts_phrase(phrase)
    if not q:
        return None
    conn = connect()
    chan_sql, chan_params = _channel_filter(channel)
    cmd_sql, cmd_params = _command_filter(include_commands)
    auth_sql, auth_params = "", []
    if author and author.lower() not in ("anyone", "*", "everyone"):
        placeholders, auth_params = _in_clause(author_keys(author))
        auth_sql = f" AND m.author IN ({placeholders}) "
    rows = conn.execute(
        "SELECT m.sent_at, m.channel, m.author, m.content FROM messages_fts f "
        "CROSS JOIN messages m ON m.id = f.rowid "
        f"WHERE f.messages_fts MATCH ? {auth_sql}{chan_sql}{cmd_sql}"
        "ORDER BY RANDOM() LIMIT 40",
        [q, *auth_params, *chan_params, *cmd_params],
    ).fetchall()
    if not rows:
        return None
    # skip OTHER bots' command invocations ($gpt, !x, <groq, #cmd); the bot's own
    # ~ prefix is already filtered in SQL.
    def _ok(content):
        return content.lstrip()[:1] not in ("$", "!", "<", "#")
    for sent_at, ch, auth, content in rows:
        if _ok(content) and len(content.split()) >= min_words:
            return (sent_at, ch, normalize_author(auth), content)
    for sent_at, ch, auth, content in rows:
        if _ok(content):
            return (sent_at, ch, normalize_author(auth), content)
    sent_at, ch, auth, content = rows[0]
    return (sent_at, ch, normalize_author(auth), content)


def author_name_search(pattern: str, channel: str = None, limit: int = 12,
                       include_bots: bool = False):
    """Regex-search archived usernames. Returns [(author, count, first, last), ...]."""
    import re as _re
    try:
        rx = _re.compile(pattern, _re.IGNORECASE | _re.UNICODE)
    except _re.error:
        return None
    conn = connect()
    if channel:
        rows = conn.execute(
            "SELECT author, COUNT(*), MIN(sent_at), MAX(sent_at) FROM messages "
            "WHERE channel = ? GROUP BY author",
            (normalize_channel(channel),),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT author, COUNT(*), MIN(sent_at), MAX(sent_at) FROM messages "
            "GROUP BY author",
        ).fetchall()
    out = []
    for author, count, first, last in rows:
        if not include_bots and _is_noise_author(author):
            continue
        if rx.search(author or ""):
            out.append((author, count, first, last))
    out.sort(key=lambda row: (-row[1], row[0]))
    return out[:limit]


def regex_search(pattern, author=None, limit=5, scan_cap=300000):
    """Regex-search archived messages (case-insensitive).

    author = a name → only their lines (scans all of them, indexed by author).
    author None / '*' / 'anyone' → everyone, but bounded to the most recent
    `scan_cap` messages so a live ~regex can't scan all 3M rows.
    Returns [(sent_at, channel, author, content), ...], or None on a bad pattern.
    """
    import re as _re
    try:
        rx = _re.compile(pattern, _re.IGNORECASE | _re.UNICODE)
    except _re.error:
        return None
    conn = connect()
    if author and str(author).lower() not in ("*", "anyone", "everyone"):
        ph, params = _in_clause(author_keys(author))
        cur = conn.execute(
            f"SELECT sent_at, channel, author, content FROM messages "
            f"WHERE author IN ({ph}) ORDER BY sent_at DESC", params)
    else:
        cur = conn.execute(
            "SELECT sent_at, channel, author, content FROM messages "
            "ORDER BY id DESC LIMIT ?", (scan_cap,))
    out = []
    for row in cur:
        if rx.search(row[3] or ""):
            out.append(tuple(row))
            if len(out) >= limit:
                break
    return out
