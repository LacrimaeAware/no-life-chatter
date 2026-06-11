# Chat Archive — design

A single searchable SQLite database of every chat message you have: historical
Chatterino logs ingested once, plus everything the bot sees live going forward.
This is the foundation layer — personas, "did user X ever say Y?", stats, and
trivia all read from it.

**Status: design — not built yet.** See [PERSONA_BOT_ROADMAP.md](PERSONA_BOT_ROADMAP.md)
for where this fits.

## Why SQLite + FTS5 (and not an LLM)

"Did user X ever say Y?" is an *exact retrieval* question. A full-text index
answers it instantly, perfectly, offline, and for free. SQLite ships with
[FTS5](https://www.sqlite.org/fts5.html) built in — no server, no dependencies,
and Python's stdlib `sqlite3` can use it directly. An LLM only enters the
picture for *fuzzy* questions ("what does X usually talk about?"), and even
then it works best reading rows this index retrieved first.

A ~13 GB text corpus is comfortably within SQLite's range (the format is used
for databases orders of magnitude larger). Expect the DB + FTS index to be
roughly the size of the ingested text; ingest only the channels you care about
if disk matters.

## Sources

### 1. Chatterino2 logs (historical, one-time ingest)

Chatterino logs live under `<Chatterino2>/Logs/Twitch/Channels/`, one
subdirectory per channel, one file per channel per day, named
`<channel>-YYYY-MM-DD.log`. The format (verified against a real multi-GB
archive):

- **Encoding:** UTF-8 without BOM, bare `\n` line endings (even on Windows).
  Real emoji and arbitrary bytes appear — open with `encoding="utf-8",
  errors="replace"`.
- **Header lines:** `# Start logging at 2025-03-06 14:37:36 Eastern Standard Time`
  and a matching `# Stop logging at ...`. These repeat **mid-file** every time
  Chatterino restarts or the tab reopens, and Stop is often missing after
  crashes. Treat any line starting with `# ` as a header.
- **Chat lines:** `[HH:MM:SS] username: message text` — 24-hour **local** time,
  time-only. The *date comes from the filename*; the timezone only from the
  header text. Twitch logins are `[a-z0-9_]+`, so splitting on the **first**
  `": "` after the timestamp is safe (message bodies freely contain colons —
  never split on the last one).
- **System lines:** timestamped but with no `user:` part — `connected`,
  `disconnected`, `joined channel`, emote-reload notices, etc. A chat-line
  regex like `^\[(\d{2}:\d{2}:\d{2})\] ([a-z0-9_]+): (.*)$` must be allowed to
  fail and fall through to a system branch.
- **Moderation lines:** `user has been timed out for 1m. ` (note trailing
  space + period), bans, unbans. **Trap:** moderator actions can masquerade as
  chat lines — `[12:10:48] somemod: somemod timed out someuser for 7d. ` parses
  as a "message" from the moderator. Filter lines whose body matches the
  moderation-notice shapes if purity matters.
- **Empty lines:** `[21:56:54] ` (timestamp + single space) occur; guard
  against index errors.
- Messages are always single-line (one IRC message = one log line); no `/me`
  marker survives into the log.
- Channel directories accumulate typo/rename duplicates of the same streamer
  over time — keep an optional alias map to merge them at ingest.
- Sibling log streams exist next to `Channels/`: `Mentions/`, `Whispers/`,
  `AutoMod/`, `Live/` (stream up/down events — handy as an activity index).
  Ingest is per-stream optional.

### 2. Live capture (ongoing)

The bot already receives every message in its joined channels. A small hook in
`handlers.process_message` inserts each message into the same table. This makes
the archive self-maintaining from the day it's built, independent of
Chatterino running.

## Schema

```sql
CREATE TABLE messages (
    id       INTEGER PRIMARY KEY,
    channel  TEXT NOT NULL,        -- lowercase channel login
    author   TEXT NOT NULL,        -- lowercase user login
    sent_at  TEXT NOT NULL,        -- 'YYYY-MM-DD HH:MM:SS' (local time)
    content  TEXT NOT NULL,
    source   TEXT NOT NULL DEFAULT 'chatterino'  -- 'chatterino' | 'live'
);
CREATE INDEX idx_msg_author  ON messages(author, sent_at);
CREATE INDEX idx_msg_channel ON messages(channel, sent_at);

-- Full-text index over content, contentless-delete style keyed to messages.
CREATE VIRTUAL TABLE messages_fts USING fts5(
    content, content='messages', content_rowid='id'
);

-- Ingest ledger so re-running the importer is incremental and idempotent.
CREATE TABLE ingested_files (
    path  TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    rows  INTEGER NOT NULL
);
```

Dedup between Chatterino history and live capture: the live hook only writes
messages the bot receives in real time, and the one-time ingest covers the
past; overlap is limited to the build day and is harmless for every use case
here (stats, search, personas). If exactness matters later, add a
`UNIQUE(channel, author, sent_at, content)` index with `INSERT OR IGNORE`.

The DB lives at `data/unsynced/chat_archive.db` — inside an already-gitignored
directory, so chat content can never reach the public repo.

## Ingest CLI

```
python scripts/ingest_chatterino.py <logs-root> [--channels a,b,c] [--since 2025-01-01]
```

- Walks `<logs-root>/<channel>/<channel>-YYYY-MM-DD.log`, skips files already
  in `ingested_files` with unchanged mtime (so re-runs only pick up new days).
- Parses per the format spec above; system/moderation/header/empty lines are
  counted but not stored (or stored with `source='system'` if you want them).
- Batched inserts inside transactions (`executemany`, ~10k rows/commit) — a
  multi-GB ingest is minutes, not hours.

## Query layer

Offline (full archive):

```
python scripts/ask_archive.py said <user> "<phrase>"     # did X ever say Y -> matching rows
python scripts/ask_archive.py quotes <user> [n]          # random quotes by X
python scripts/ask_archive.py stats <user>               # counts, first/last seen, top words/emotes
python scripts/ask_archive.py search "<fts5 query>"      # raw full-text search
```

In chat (bot commands, follow the existing `commands/` auto-discovery pattern):

| Command | Does |
| --- | --- |
| `~said <user> <phrase>` | "user said that 3 times, first on 2025-04-02: '...'" |
| `~quote <user>` | random real quote from the archive |
| `~firstseen <user>` | first recorded message + date |
| `~chatstats <user>` | message count, top emotes, busiest hour |

FTS5 `MATCH` handles word/phrase/prefix/boolean queries natively
(`"exact phrase"`, `term1 AND term2`, `wor*`).

## Optional later: semantic layer

For "did X ever talk about <concept>" (not exact words): embed messages with a
local sentence-transformers model (CPU is fine at this scale, done once) into a
vector table, and answer fuzzy questions by nearest-neighbor + showing the
actual rows. Strictly optional — exact search covers most of the fun, and this
adds a dependency. Don't build until exact search feels limiting.
