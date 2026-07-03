# NoLifeChatter Command Bible

This is the public, source-of-truth command map for the live bot. Commands are
auto-discovered from `commands/*.py`: if a file defines `handle_name`, then
the matching prefixed chat command exists.

For compact chat help, use `~help` or `~help <command>`.

## Quick Index

### Translation And Language

| Command | Access | What it does |
| --- | --- | --- |
| `~autotl` | admin | Toggle auto-translation for your own messages. |
| `~setlang <LANG>` | admin | Set your translation target language, such as `EN` or `ES`. |
| `~tloutput local|whisper|channel <name>` | admin | Choose where translations are sent. |
| `~chan_autotl <channel> on|off` | super admin | Enable or disable auto-translation for one channel. |
| `~global_autotl on|off` | super admin | Global auto-translation switch. |
| `~notranslate <user> [undo]` | super admin | Stop auto-translating a specific user (they write slang the detector misreads); `undo` re-enables; no arg lists opted-out users. |
| `~practice on <learn_langs_csv> [native]` | anyone | Practice mode: translate your native language into study languages and your study-language attempts back to native. |
| `~practice off` / `~practice show` | anyone | Stop or inspect practice mode. |
| `~romanize on|off|show` | anyone | Toggle romanized readings for supported non-Latin scripts. |
| `~speak <lang>` / `~speak <lang> off` / `~speak show` | anyone | Tell the bot which language(s) you speak so short messages are handled better. |

### Archive Search And Stats

| Command | Access | What it does |
| --- | --- | --- |
| `~said <user> [chat=<channel>] [sort=] <phrase>` | anyone | Search whether a user said a phrase. Exact match first, then normalized close match. Results shuffled by default; `sort=chrono\|newest\|name` to order them. |
| `~said anyone [chat=<channel>] [sort=] <phrase>` | anyone | Search the whole archive for a phrase. Same `sort=` flag; default is a seeded shuffle. |
| `~saidnext` | anyone | Continue your last `~said` search for about a minute, keeping the same order (the shuffle seed is remembered). |
| `~saidmost <phrase> [chat=<channel>] [n]` | anyone | Who has said a phrase the most — ranks chatters by how many of their messages contain it (aliases merged, bots skipped). |
| `~regex <user|anyone> <pattern>` | anyone | Case-insensitive regex search over archived messages. |
| `~userregex <pattern> [chat=<channel>]` | anyone | Regex-search archived usernames. |
| `~quote <user>` | anyone | Post a random real quote from a user. |
| `~random <word> [user=] [chat=]` | anyone | A random real quote containing a word/phrase. Anyone by default; narrow with user= and/or chat=. |
| `~firstseen <user>` | anyone | Show a user's first archived message. |
| `~chatstats <user> [chat=<channel>]` | anyone | Show message count, first/last seen, average length, and busiest hour. |
| `~regulars [channel] [min_messages] [limit]` | anyone | Top regulars in a channel, ignoring obvious bots by default. |
| `~askchat [raw] [user=<name>\|<name>] [chat=<channel>] <question>` | anyone | Evidence-backed archive/lore QA. Normal mode retrieves receipts and asks the local LLM for a cautious cited answer; `raw` returns receipts only. |

### Persona And Generation

| Command | Access | What it does |
| --- | --- | --- |
| `~markov <user>` | anyone | Generate a local Markov-chain line from that chatter's messages. |
| `~mimic <user>` | anyone | Alias-style Markov persona command; same family as `~markov`. |
| `~persona <user> [message] [model=llama|lora]` | anyone | Ask the local LLM to reply as a chatter using archive evidence and live context. |
| `~hyper <user> [message] [model=llama|lora]` | anyone | Same as `~persona`, but exaggerated for comedy. |
| `~generate <tags...>` | anyone | Generate an example message from a recipe: users, traits, topic words, `chat=`, `year=`, `engine=markov|llm`, `model=...`. |
| `~generate save <name> <tags...>` | anyone | Save a per-user generation recipe. |
| `~generate <name> [more tags]` | anyone | Use a saved recipe and optionally stack more tags. |
| `~generate list` / `~generate del <name>` | anyone | List or delete your saved recipes. |
| `~botpersona status [chat=<channel>]` | super admin | Inspect the resident persona for a channel. |
| `~botpersona off [chat=<channel>]` | super admin | Clear the resident persona for a channel. |
| `~botpersona <user> [chat=<channel>] [minutes=360] [mode=regular\|response\|random\|silent] [chance=] [topic=] [curve=] [directed=] [directed_cooldown=] [idle=]` | super admin | Set a channel-scoped resident persona that can autonomously reply and idle-chat. |
| `~botmode regular\|response\|random\|silent [minutes] [chat=<channel>]` | super admin | Change resident persona response mode and optional expiry. |
| `~botcontext [chat=<channel>] <text\|clear>` | super admin | Set extra standing instruction for the resident persona. |
| `~botchance <base> [directed] [chat=<channel>] [topic=] [idle=] [greeting=] [cooldown=]` | super admin | Tune resident persona reaction, topic-boost, idle, greeting, and cooldown probabilities. |

Important distinction: `~persona`, `~hyper`, and `~generate` are one-shot
commands. The resident persona commands above are channel-scoped and
super-admin-only.

### Stylometry, Similarity, Traits

| Command | Access | What it does |
| --- | --- | --- |
| `~whosaid <sentence>` | anyone | Guess who in the current chatroom is most likely to have said a line. |
| `~whosaid anyone <sentence>` | anyone | Guess against the whole archive roster. |
| `~markers <user> [chat=all|<channel>] [year=YYYY]` | anyone | Show distinctive words and word-pairs for a chatter. |
| `~like <user> [chat=all|<channel>] [year=YYYY]` | anyone | Lexical/emote voice neighbors with shared-marker evidence. Useful for alt-hunting. |
| `~vibes <user>` | anyone | Semantic embedding-space neighbors: same topics/energy even without shared catchphrases. |
| `~twin <user>` | anyone | Overall nearest match, blending lexical and semantic similarity. |
| `~traits <user>` | anyone | Project a chatter onto built-in trait axes relative to the room average. ⚡ marks an axis where their messages span both poles, not one side. |
| `~style <user>` | anyone | How a chatter TYPES — standout behaviors vs the room (emote rate, verbosity, caps, @mentions, profanity, vocabulary). Data-driven "structural" personality; unlike ~traits it doesn't touch the topic embedder. |
| `~top <trait> [n] [burst]` | anyone | Trait leaderboard. Built-in traits answer instantly; new words build dynamic axes. Add `@user` (or `user=`) to show that user's rank, σ, and the people just above/below instead of the leaderboard. |
| `~bottom <trait> [n] [burst]` | anyone | Reverse leaderboard — who leans LEAST toward a trait (most toward its opposite). |
| `~distinct [top\|bottom] [n]` | anyone | Chatters farthest from or closest to the room average across built-in trait axes. |
| `~why <user> <trait> [words]` | anyone | Show real indexed messages that drive a user's score on a trait axis. |
| `~axis <trait> [n]` | anyone | Inspect a trait/custom axis by showing nearest neighboring axes. |
| `~emote <emote>` | anyone | Explain what the bot thinks an emote means from registry facts and usage-context neighbors. |
| `~irony <message> [context=...]` | anyone | Experimental irony/sincerity read using sarcasm and content-extremity axes. |
| `~iq <user>` | anyone | Roster-relative text-IQ style score: peak expressed cognition in chat, not actual IQ. |
| `~iq top [n]` / `~iq bottom [n]` | anyone | Highest or lowest text-IQ style scores in the current cache. |
| `~funny <user>` | anyone | Comedy-influence score: how much more OTHER people laugh in the ~30s after you talk than the ~30s before (before/after delta cancels stream-wide reactions). Roster-relative index (100 = average chatter); `breadth` = distinct people you've made laugh. Self-laughs excluded; bot/command lines excluded. Default chats are the configured conversational ones; `chat=` overrides. |
| `~funny top [n]` / `~funny bottom [n]` | anyone | Funniest / least-funny chatters by comedy influence across the configured chats. |

The analysis commands are exploratory. They are useful for debugging and fun
comparisons, not clinical or total-person labels.

### Moderation And Utility

| Command | Access | What it does |
| --- | --- | --- |
| `~help [page|command]` | anyone | List commands or show details for one command. `~mimic`/`~markov` work but are unlisted; admin commands are listed only via `~admin`. |
| `~admin` | anyone | List the admin / super-admin commands (kept out of the public `~help`; the commands still enforce their own permissions). |
| `~ping` | anyone | Latency and host stats. |
| `~artifacts` | anyone | Compact status for generated persona artifacts and stale semantic caches. |
| `~banuser <name>` | super admin | Ban a chatter from using bot commands. Translation/archive capture are unaffected. |
| `~banuser list` | super admin | Show command-banned users. |
| `~unbanuser <name>` | super admin | Remove a command ban. |
| `~warnings [n]` | super admin | Review recent anti-spam cooldown offenses. |

## Planned But Not Live

The current resident layer supports one real chatter persona per channel, topic
affinity boosting from archive hits, rare idle messages, and Twitch reply-thread
responses when a triggering message ID is available. Full `~generate` recipes
as resident personas and queue-depth feedback are still future work.

## Artifact Dependencies

Some commands read live SQLite tables only; others need offline artifacts:

| Artifact | Built by | Used by |
| --- | --- | --- |
| `data/unsynced/chat_archive.db` | live capture and `scripts/ingest_chatterino.py` | archive, stats, Markov, persona evidence, IQ |
| `data/unsynced/persona_classifier.pkl` | `scripts/train_classifier.py` | `~whosaid`, `~markers`, `~like`, persona eval pieces |
| `data/unsynced/persona_embeddings.pkl` | `scripts/build_persona_embeddings.py` | `~vibes`, `~twin`, trait averages |
| `data/unsynced/msg_index/` | `scripts/build_message_index.py` | `~why`, burst `~top`, semantic persona retrieval |
| `data/unsynced/iq_scores.pkl` | `scripts/build_iq_v2.py` | `~iq` |
| `data/unsynced/fact_bank.jsonl` | `scripts/build_fact_bank.py` | `~askchat`, offline archive/lore QA |

Use `~artifacts` or `python scripts/artifact_status.py` to check whether these
artifacts are missing, missing build metadata, or built with the current
semantic unit/model.

After identity merges, embedding-model changes, or major filter changes, run
`10-rebuild-persona-artifacts.bat` or `scripts/rebuild_persona_artifacts.py`.
For an unattended run, use `10-rebuild-persona-artifacts-background.bat`; it
writes stdout/stderr logs under `data/unsynced`.
The default semantic unit for embeddings/indexes is now a merged utterance; use
`--semantic-unit message` only for a deliberate A/B against the old behavior.
