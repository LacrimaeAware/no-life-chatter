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

import config

# Name-spelling variants that should count as the same channel/user.
ALIASES = {alias.lower(): real.lower() for alias, real in config.ARCHIVE_ALIASES.items()}

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


def normalize(name: str) -> str:
    """Lowercase, strip @ / whitespace, and resolve known spelling aliases."""
    name = (name or "").strip().lstrip("@").rstrip(",").lower()
    return ALIASES.get(name, name)


def connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(config.ARCHIVE_DB), exist_ok=True)
        # WAL + generous busy timeout so the bot's live writes and in-chat
        # queries keep working while an ingest run holds long write
        # transactions in another process.
        _conn = sqlite3.connect(config.ARCHIVE_DB, timeout=30)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.executescript(_SCHEMA)
    return _conn


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
                batch.append((channel, normalize(author), f"{date} {t}", content,
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

def record_live(channel: str, author: str, content: str, sent_at: str) -> None:
    """Append one live chat message. Never raises into the bot's message path."""
    try:
        conn = connect()
        with conn:
            conn.execute(
                "INSERT INTO messages (channel, author, sent_at, content, source) VALUES (?,?,?,?, 'live')",
                (normalize(channel), normalize(author), sent_at, content),
            )
    except Exception as e:
        logging.warning(f"chat_archive live write failed: {e}")


# ------------------------------- queries --------------------------------

def _fts_phrase(phrase: str) -> str:
    """Quote user text as a single FTS5 phrase (handles internal quotes)."""
    return '"' + phrase.replace('"', '""') + '"'


def said(author: str, phrase: str, limit: int = 3):
    """All matches of phrase by author: (total_count, [(sent_at, channel, content)...]).

    CROSS JOIN forces SQLite to run the FTS match once and probe messages by
    rowid; the plain-JOIN plan flips the loop order and re-evaluates the FTS
    match per author row (measured: 79s vs 0.003s on a 741k-row archive).
    """
    conn = connect()
    author = normalize(author)
    searchable = re.sub(r"[^0-9A-Za-z]+", " ", phrase).strip()
    if not searchable:
        # Emoji/symbol-only phrase: the FTS tokenizer would drop everything
        # and report a confident 0 — substring search answers it correctly.
        like = "%" + phrase.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_") + "%"
        total = conn.execute(
            r"SELECT COUNT(*) FROM messages WHERE author = ? AND content LIKE ? ESCAPE '\'",
            (author, like),
        ).fetchone()[0]
        rows = conn.execute(
            r"SELECT sent_at, channel, content FROM messages "
            r"WHERE author = ? AND content LIKE ? ESCAPE '\' ORDER BY sent_at LIMIT ?",
            (author, like, limit),
        ).fetchall()
        return total, rows
    q = _fts_phrase(phrase)
    total = conn.execute(
        "SELECT COUNT(*) FROM messages_fts f CROSS JOIN messages m ON m.id = f.rowid "
        "WHERE f.messages_fts MATCH ? AND m.author = ?",
        (q, author),
    ).fetchone()[0]
    rows = conn.execute(
        "SELECT m.sent_at, m.channel, m.content FROM messages_fts f "
        "CROSS JOIN messages m ON m.id = f.rowid "
        "WHERE f.messages_fts MATCH ? AND m.author = ? ORDER BY m.sent_at LIMIT ?",
        (q, author, limit),
    ).fetchall()
    return total, rows


def random_quote(author: str, min_words: int = 3):
    """One random, reasonably substantial message by author (or None)."""
    conn = connect()
    author = normalize(author)
    row = conn.execute(
        "SELECT sent_at, channel, content FROM messages WHERE author = ? "
        "AND length(content) - length(replace(content, ' ', '')) >= ? "
        "ORDER BY RANDOM() LIMIT 1",
        (author, min_words - 1),
    ).fetchone()
    if row is None:  # fall back to any message at all
        row = conn.execute(
            "SELECT sent_at, channel, content FROM messages WHERE author = ? "
            "ORDER BY RANDOM() LIMIT 1",
            (author,),
        ).fetchone()
    return row


def first_seen(author: str):
    """Earliest archived message by author: (sent_at, channel, content) or None."""
    conn = connect()
    return conn.execute(
        "SELECT sent_at, channel, content FROM messages WHERE author = ? "
        "ORDER BY sent_at LIMIT 1",
        (normalize(author),),
    ).fetchone()


def stats(author: str):
    """Summary numbers for author, or None if unseen."""
    conn = connect()
    author = normalize(author)
    row = conn.execute(
        "SELECT COUNT(*), MIN(sent_at), MAX(sent_at), AVG(length(content)) "
        "FROM messages WHERE author = ?",
        (author,),
    ).fetchone()
    if not row or row[0] == 0:
        return None
    busiest = conn.execute(
        "SELECT substr(sent_at, 12, 2) AS hh, COUNT(*) AS n FROM messages "
        "WHERE author = ? GROUP BY hh ORDER BY n DESC LIMIT 1",
        (author,),
    ).fetchone()
    return {
        "messages": row[0],
        "first": row[1],
        "last": row[2],
        "avg_chars": round(row[3] or 0),
        "busiest_hour": int(busiest[0]) if busiest else None,
    }


def search_all(phrase: str, limit: int = 10):
    """Full-text search across all authors: [(sent_at, channel, author, content)...]."""
    conn = connect()
    return conn.execute(
        "SELECT m.sent_at, m.channel, m.author, m.content FROM messages_fts f "
        "JOIN messages m ON m.id = f.rowid WHERE f.messages_fts MATCH ? "
        "ORDER BY m.sent_at LIMIT ?",
        (_fts_phrase(phrase), limit),
    ).fetchall()
