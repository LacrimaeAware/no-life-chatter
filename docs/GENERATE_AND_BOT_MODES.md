# ~generate, Saved Combos, And Bot Modes

Originally specified 2026-06-11; refreshed 2026-07-15. Generation recipes,
resident personas, queue-depth feedback, and ban/cooldown controls are live.

## 1. ~generate — tag-driven example generation (IMPLEMENTED)

One command that takes an unordered bag of tags/filters and produces an
example message from that recipe:

    ~generate <tags...>

Tag kinds (order never matters; comma or space separated):
- **chatter names** — one = that person's voice; several = a FUSION of them
- **trait poles** — optimist/doomer, professor/brainrot, menace/wholesome,
  ironic/sincere, unhinged/chill: pushes the message maximally that way
- **chat=<channel>** — only use the chatters' messages from that channel
  (repeatable; each adds a channel to the allowed set)
- **year=<YYYY>** — only their messages from that year
- **topic=... / any leftover words** — free-text topic the message is about
- **engine=llm|markov** — markov works for chatter tags only (no traits/topic)
- **model=llama|lora** — LLM override, same shortcuts as ~persona
- **a saved combo name** — expands to its stored tags (see below)

Examples:
    ~generate somechatter doomer
    ~generate chatterA chatterB chat=somechannel year=2022
    ~generate optimist topic=world of warcraft
    ~generate optimist            <- generic maximal optimist, varied each call

Engine details: traits/topic-only recipes are pure prompting (pole example
sentences as register hints + a variety nudge). Chatter recipes feed each
person's (scoped) exemplars into the prompt; fusion asks for one blended
voice. Candidate checks (no URLs, no bot-command lines, output filter) reuse
the persona pipeline's rules.

## 2. Saved combos (IMPLEMENTED) — per-user, not global

    ~generate save florpface chatterA chatterB optimist chat=somechannel
    ~generate florpface                  <- uses the recipe
    ~generate florpface professor        <- recipe + extra tags stacked on top
    ~generate list / ~generate del florpface

Stored per Twitch user in the settings DB (gen_combos table); combos expand
one level (a combo may not reference another combo).

## 3. Bot Modes (IMPLEMENTED)

Super-admin only. The bot can now get a channel-scoped "resident persona"
using one real chatter voice. Full `~generate` recipes and saved combos as
resident personas are still future work.

- **regular** — may respond to normal incoming chat, with higher odds for
  direct messages, greetings, and topics the persona has many archived hits
  for. It can also send rare idle/empty-chat lines after chat has been quiet,
  capped by a bot-streak guard.
- **response** — only replies when @mentioned, in the resident persona.
- **random** — random chance per message (chance settable by command).
- **silent** — commands only.

Live commands:

    ~botpersona status [chat=<channel>]
    ~botpersona off [chat=<channel>]
    ~botpersona <user> [chat=<channel>] [minutes=360] [mode=regular|response|random|silent] [chance=] [topic=] [curve=] [directed=] [directed_cooldown=] [idle=]
    ~botmode regular|response|random|silent [minutes] [chat=<channel>]
    ~botcontext [chat=<channel>] <free text|clear>
    ~botchance <base> [directed] [chat=<channel>] [topic=] [idle=] [greeting=] [cooldown=]

Time-limited modes expire automatically. Resident output can also carry a
configured prefix, which is live runtime state in `data/unsynced` rather than
public repository config. Triggered resident replies use Twitch reply tags when
the incoming message has an ID and the installed TwitchIO websocket exposes the
low-level reply method; otherwise they fall back to normal channel send.

## 4. Admin And Abuse Controls (IMPLEMENTED)

- `~banuser <name>` / `~unbanuser <name>` is implemented. Super-admin only;
  banned users' commands are ignored entirely by `command_processor`.
- Escalating anti-spam cooldowns are implemented for command stacking while a
  previous command is still pending. Recent offenses are reviewable with
  `~warnings`.
- GPU-heavy commands, resident replies, and ambient generation share
  `services/model_queue.py`. It announces start/queue position, allows one
  active request per user, suppresses duplicate signatures, distinguishes a
  busy model from an offline server, and exposes super-admin status/clear.

## Remaining Work

- Current resident mode state is `data/unsynced/resident_personas.json`.
- The live queue is process-wide, not cross-process. Offline maintenance scripts
  should run with the bot paused until a shared worker/lease exists.
- A future richer volition gate could use a cheap "reply or STAY SILENT"
  classifier before generation. The current live path uses probability,
  direct/greeting heuristics, curved FTS topic-affinity boost, cooldown,
  idle-roll limits, and a STOP instruction in the persona prompt to keep model
  calls bounded.
- Ban list: data/synced settings DB, cached set, checked first thing in
  process_command. Implemented.
