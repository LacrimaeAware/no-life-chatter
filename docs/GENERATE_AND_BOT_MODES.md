# ~generate, saved combos, and bot modes — design

User-requested feature family, 2026-06-11. Part 1 (~generate + saved combos)
is implemented; parts 2-4 are specced here for implementation (bot modes,
admin controls, queueing).

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

## 3. Bot modes (SPEC — not yet implemented)

Super-admin only. The bot gets a "resident persona" (any ~generate recipe,
including saved combos) and a response mode:

- **regular** — responds of its own volition: per-message, an LLM gate asks
  "would <persona> jump in here?" (cheap context check) capped by a rate
  limit; persona decides what to react to. Optional extra prompt context
  ("be zany", "make everything about world of warcraft").
- **response** — only replies when @mentioned, in the resident persona.
- **random** — random chance per message (chance settable by command).
- **silent** — commands only.

Proposed commands:
    ~botmode regular|response|random|silent [minutes]   (auto-revert after)
    ~botpersona <recipe tags or combo name>
    ~botcontext <free text>            (extra standing instruction; 'clear')
    ~botchance 0.002                   (random-mode odds; also reaction odds)
@-mention responses use the resident persona when set, else the old behavior.
Time-limited modes revert to silent when the timer lapses.

## 4. Admin & abuse controls (SPEC — not yet implemented)

- ~banuser <name> / ~unbanuser <name> — super admin; banned users' commands
  are ignored entirely (checked in command_processor before dispatch).
- **Queue + cooldown**: LLM commands run through a small queue; if 3+ jobs
  are pending, the bot replies once "queue full (3 ahead), hold on" and
  processes when space frees. Per-user cooldown (e.g. one LLM command per
  20s) with a polite nudge instead of silence.

## Implementation notes for whoever picks up 3/4

- The queue belongs around services/llm.chat's _chat_lock (it already
  serializes; add a counter + reject/notify at depth >= 3).
- Mode state: small table or config-backed dict {channel: (mode, persona,
  context, until_ts)}; check in message_service.handle/maybe_react.
- The volition gate for regular mode can reuse the persona prompt with a
  cheap "reply or STAY SILENT" instruction and max_tokens=8 — but it costs a
  model call per message, so guard with rate limits + random pre-gate.
- Ban list: data/synced settings DB, cached set, checked first thing in
  process_command.
