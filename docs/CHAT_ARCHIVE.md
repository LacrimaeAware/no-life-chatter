# Chat Archive — design

A single searchable SQLite database of every chat message you have: historical
Chatterino logs ingested once, plus everything the bot sees live going forward.
This is the foundation layer — personas, "did user X ever say Y?", stats, and
trivia all read from it.

**Status: built.** `utils/chat_archive.py` (schema, parser, queries),
`scripts/ingest_chatterino.py` (historical ingest), `scripts/ask_archive.py`
(offline CLI), live capture in `handlers.py`, and the `~said` / `~quote` /
`~firstseen` / `~chatstats` commands. See [ROADMAP.md](ROADMAP.md) for where
this fits in the current build order.

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

### 3. External user logs via Zonian/IVR mirrors

`scripts/download_zonian_user_logs.py` can enrich the private archive with
public historical logs for specific users in a specific channel. It calls the
Zonian mirror API (`https://logs.zonian.dev/api/<channel>/<user>`) to discover
which logging instance has the best coverage, downloads monthly raw logs from
that instance, and optionally imports non-duplicate rows into
`data/unsynced/chat_archive.db`.

It selects usernames already present in your local archive for a channel,
defaults to users with at least 25 local messages, skips configured bot/noise
accounts (`config` `EXCLUDE_USERS` plus common public bots) and obvious `*bot`
accounts unless asked otherwise, and optionally imports after download. Raw
logs are stored under
`data/unsynced/external_logs/zonian/raw/<channel>/<user>/`; this path is
gitignored and must stay private. (Convenience double-click launchers that hard-
code a specific channel and roster are kept private under `_private/`.)

CLI form:

```
python scripts/download_zonian_user_logs.py --channel yourchannel --users user1,user2 --import-archive
```

Automated from the existing local archive roster:

```
python scripts/download_zonian_user_logs.py --channel yourchannel --from-archive --min-archive-messages 25 --import-archive
```

Cross-channel pull, but only for users from one home channel's roster:

```
python scripts/download_zonian_user_logs.py --channel otherchannel --from-archive --users-from-channel yourchannel --min-archive-messages 25 --import-archive
```

Useful test run:

```
python scripts/download_zonian_user_logs.py --channel yourchannel --from-archive --min-archive-messages 10000 --limit-months 1
```

The mirror's raw monthly logs use UTC timestamps; the importer converts them to
the local archive timezone (`America/New_York` by default) before inserting.
Duplicate protection first checks exact `(channel, author, sent_at, content)`,
so rerunning the downloader is safe. For overlapping logs whose timestamps may
be shifted by timezone/source differences, import also skips substantial same
author/channel lines with the same normalized text within a configurable window
(`--dedupe-window-hours`, default 12). Short repeated chat/emote lines still
need exact timestamps, so genuinely repeated short emote-style chatter is not
collapsed.

This same two-layer rule is the baseline overlap strategy for future older local
logs too: import every source into the same normalized schema, then skip exact
matches and substantial near-time text matches.

Important context rule: Zonian/IVR imports are often one-speaker monthly files
(`raw/<channel>/<user>/<month>.log`), not complete channel-day transcripts.
Archive search may still use those rows as quote/message evidence, but
conversation reconstruction must not treat adjacent rows as surrounding chat
unless the same channel/time slice contains another imported speaker. The shared
`chat_archive.context_window()` helper enforces this by returning only the hit
for author-only mirror neighborhoods, while normal Chatterino/live channel logs
and multi-author mirror neighborhoods can still provide timestamp-local context.
It also removes alias-collapsed duplicate copies of the same line so merged alts
do not echo the same evidence repeatedly in prompts.

## Schema

```sql
CREATE TABLE messages (
    id       INTEGER PRIMARY KEY,
    channel  TEXT NOT NULL,        -- lowercase channel login
    author   TEXT NOT NULL,        -- canonical lowercase user login
    sent_at  TEXT NOT NULL,        -- 'YYYY-MM-DD HH:MM:SS' (local time)
    content  TEXT NOT NULL,
    source   TEXT NOT NULL DEFAULT 'chatterino',  -- 'chatterino' | 'live' | 'zonian'
    src_path TEXT, -- originating log file; re-ingest of a grown file replaces
                   -- exactly its own rows (alias-merged channels stay intact)
    raw_author TEXT -- login before alias normalization
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

-- ID/display cache. Offline lookup time is deliberately separate from real
-- live activity so maintenance order cannot make an old alias the display name.
CREATE TABLE author_ids (
    author         TEXT PRIMARY KEY,
    twitch_id      TEXT,
    checked_at     TEXT,
    display        TEXT,
    last_seen_live TEXT
);
```

Schema changes are additive migrations in `chat_archive.connect()`. Alias maps
are also fingerprinted into generated semantic/persona artifacts; changing an
identity map makes artifact status warn until those files are rebuilt.

Semantic utterance v3 merges a person's short message bursts independently in
each channel and collapses duplicate text components inside the merged text.
The record still retains every source message id/part for chronology and
provenance. This matters when logs from several live channels overlap or the
same imported event appears through an alias mirror: unrelated rooms must never
be concatenated, and repeated storage must not become repeated semantic
evidence. Generated message indexes, person vectors, and IQ caches record the
utterance version so older files produce an artifact warning.

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
python scripts/ask_archive.py said <user> <phrase...>    # did X ever say Y -> matching rows
python scripts/ask_archive.py near <user> <phrase...>    # closest normalized lines by X
python scripts/ask_archive.py quote <user>               # random quote by X
python scripts/ask_archive.py stats <user>               # count, first/last seen, busiest hour
python scripts/ask_archive.py search <phrase...>         # phrase search, all authors
```

Phrases are matched as whole-word exact phrases (FTS5 under the hood, with the
input always quoted — chat text can't inject query syntax). A phrase with no
alphanumeric characters at all (emoji-only) falls back to substring search,
since the tokenizer would otherwise drop it.

In chat (bot commands, follow the existing `commands/` auto-discovery pattern):

| Command | Does |
| --- | --- |
| `~said <user> <phrase>` | exact quote search first; if none, closest normalized match |
| `~quote <user>` | random real quote from the archive |
| `~firstseen <user>` | first recorded message + date |
| `~chatstats <user>` | message count, top emotes, busiest hour |

One hard-won planner note: query the FTS table with `CROSS JOIN messages` (or
an `IN (SELECT rowid ...)` subquery) so SQLite runs the FTS match once and
probes by rowid — the innocent-looking plain `JOIN` flips the loop order and
re-evaluates the match per candidate row, turning a 3 ms query into minutes on
a large archive.

When exact `said` search finds nothing, the command and CLI fall back to a
normalized close-match check. That treats straight vs curly apostrophes,
punctuation, case, and spacing as irrelevant, so copied lines are still caught
when chat/LLM output changes typography or adds a small mention/emote tail.
Use `scripts/ask_archive.py near <user> <line>` when you explicitly want the
closest lines instead of an exact-first answer.

## Optional later: semantic layer

For "did X ever talk about <concept>" (not exact words): embed messages with a
local sentence-transformers model (CPU is fine at this scale, done once) into a
vector table, and answer fuzzy questions by nearest-neighbor + showing the
actual rows. Strictly optional — exact search covers most of the fun, and this
adds a dependency. Don't build until exact search feels limiting.
