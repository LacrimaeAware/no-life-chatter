# NoLifeChatter

A modular Twitch chat bot with two halves that share one local SQLite archive:

- **Live translation** — detects non-target-language messages and translates
  them on the fly, with a language-learning "practice mode."
- **Chat personas** — it archives chat, then can answer *as* a given chatter —
  a quick local Markov version, or a context-aware LLM version running on a
  local model — plus archive search (`~said`, `~quote`, stats) and rare,
  in-character "the bot does a bit" reactions.

> A personal project, cleaned up and published as a showcase. It's a record of
> something I built and a reasonable example of a modular Python bot —
> auto-discovered commands, a swappable translation backend, a local FTS5 chat
> archive, and a retrieval/fine-tuning persona pipeline. Read it, run it, or
> borrow ideas. It is not a polished product for daily use, and the persona
> features operate on real chat, so see the privacy notes before pointing it at
> anyone's logs.

## What it does

### Translation
- **Auto-translation** — flags messages that aren't already in the target
  language and translates them via DeepL (or Google). Per-user, per-channel, or
  global. Detection is local/free, so target-language messages never hit an API.
- **Practice mode** — when you write in your native language the bot whispers
  you the translation in the language(s) you're studying; when you write in a
  language you're learning it posts the native translation to chat.
- **Romanization** — optional readings for non-Latin scripts (JA Hepburn, KO
  Revised, TH via `pythainlp`).
- **Speaker profiles** — learns which languages each user actually writes in, so
  even their short messages get translated once they're established.

### Chat archive & personas
- **Searchable archive** — every message the bot sees, plus historical
  Chatterino logs, in one SQLite + FTS5 database. Powers `~said` ("did they ever
  say X?"), `~regex`, `~quote`, `~firstseen`, `~chatstats`, `~regulars` —
  instant, local, no API.
- **Stylometry** — a TF-IDF + logistic-regression authorship classifier
  (`~whosaid`: who would say this line?) and log-odds voice profiles
  (`~markers`: someone's favorite words and word-pairs vs the average chatter;
  `~like`: who shares their distinctive voice — which doubles as an
  alt-account detector). Profiles weight overuse against rarity (a term most
  of the panel uses can't be a marker) and can be scoped per chat or year.
- **Markov personas** (`~markov`/`~mimic`) — a recombination of a chatter's own
  words. Instant, fully local, no model.
- **LLM personas** (`~persona`, `~hyper`) — many-shot voice cloning: the prompt
  is built from a distinctiveness-ranked sample of the chatter's real messages
  plus topic-relevant retrieved lines and the live conversation, run against a
  **local** OpenAI-compatible model (LM Studio by default — free, private, and
  uncensored models won't refuse edgy chat). `~hyper` exaggerates their traits.
- **Reactions** — at a low per-message chance the bot spontaneously chimes in as
  a random recent chatter, reacting to the actual conversation.
- **Fine-tuning pipeline** — an offline path to export distinctiveness-weighted
  training data and train a per-persona LoRA on a rented GPU. See
  [docs/FINE_TUNING.md](docs/FINE_TUNING.md). A live A/B mode
  (`[llm] ab_models`) rolls a model per generation and tags posted lines
  (`#llama` / `#lora`) so chat itself judges the candidates.
- **Output filter** — anything the bot is about to post is checked against a
  denylist first, so recombined real chat can't get the account banned.

Per-user / per-channel / global settings persist in SQLite.

## How it works

Two products, one archive. The **runtime bot** is the root Python modules +
`commands/ services/ utils/`; the **offline pipeline** is under `scripts/`.

```
chatbot.py              # entry point: bot + background token refresh
config.py               # loads config.toml (+ .env secrets); no hard-coded data
handlers.py             # routes each message; also archives it (live capture)
command_processor.py    # parses "~command args" and dispatches
command_registry.py     # auto-discovers command modules in commands/
auth.py                 # admin / super-admin checks (from config)
commands/               # one file per command (translation + archive + persona)
services/
  message_service.py    # translation, practice mode, whispers, persona reactions
  translators.py        # DeepL / Google behind one interface
  llm.py                # async client for any OpenAI-compatible chat endpoint
  emotes.py             # per-channel 7TV/BTTV/FFZ emote fetch + strip
utils/
  language_detect.py    # local lingua detection (the free translation gate)
  speaker_profile.py    # per-user language tracking
  romanize.py + roman_scripts/   # JA / KO / TH romanizers
  token_manager.py      # keeps the Twitch OAuth token fresh
  user_settings.py      # SQLite per-user settings
  chat_archive.py       # SQLite + FTS5 archive: ingest, search, retrieval
  persona_markov.py     # Markov persona generator
  persona_llm.py        # many-shot LLM persona engine (RAG + candidate select)
  output_filter.py      # denylist gate before the bot posts anything
scripts/                # offline pipeline: init/auth, log ingest, archive query,
                        #   persona preview, fine-tune export/train (RunPod)
data/                   # gitignored: SQLite DBs, bot.log, OAuth tokens
docs/                   # design docs (start at docs/HANDOFF.md)
```

Commands are auto-discovered: drop a `commands/foo.py` with a
`handle_foo(bot, message, params)` function and `~foo` just works.

## Commands

| Command | Who | Description |
| --- | --- | --- |
| `~help [command]` | anyone | List commands, or details for one. |
| `~ping` | anyone | Latency + host stats. |
| `~practice on <langs> [native]` | anyone | Start practice mode. e.g. `~practice on es,ja en` |
| `~practice off` / `~practice show` | anyone | Stop / inspect practice mode. |
| `~romanize on\|off\|show` | anyone | Toggle romanized readings in practice mode. |
| `~speak <lang>` / `~speak show` | anyone | Flag your language so even your short messages translate. |
| `~said <user|anyone> <phrase>` | anyone | Search the chat archive: exact matches first, then closest normalized match. `anyone` searches every author. |
| `~regex <user|anyone> <pattern>` | anyone | Case-insensitive regex search over the archive. |
| `~quote <user>` | anyone | Random real quote from the chat archive. |
| `~firstseen <user>` | anyone | A user's first archived message. |
| `~chatstats <user> [chat=ch]` | anyone | Archive stats: count, first/last seen, busiest hour. |
| `~regulars [channel] [min] [n]` | anyone | Top chatters of a channel above a message floor, bots excluded. |
| `~whosaid <sentence>` | anyone | Stylometry: which chatter most likely said a line (novel sentences work). Ranks people active in this chat; `anyone` ranks the whole archive. |
| `~markers <user> [chat=] [year=]` | anyone | A chatter's voice profile — favorite words + word-pairs vs the average chatter. Scoped to the current chat by default. |
| `~like <user> [chat=] [year=]` | anyone | Who shares their distinctive voice, with the shared markers as evidence — also a decent alt-account detector. |
| `~markov <user>` / `~mimic <user>` | anyone | Quick Markov-chain line in a chatter's style (no model needed). |
| `~persona <user> [msg] [model=x]` | anyone | Talk to an AI persona of a chatter (local LLM, context-aware). `model=` picks a configured model shortcut. |
| `~vibes <user>` | anyone | Semantic twins — embedding-space similarity (same topics/energy, no shared words needed). |
| `~hyper <user> [msg]` | anyone | Same, but their traits exaggerated for comedy. |
| `~autotl` | admin | Toggle auto-translate for yourself. |
| `~setlang <LANG>` | admin | Set your translation target language (e.g. `EN`). |
| `~tloutput local\|whisper\|channel <name>` | admin | Where your translations are sent. |
| `~chan_autotl <channel> on\|off` | super admin | Per-channel auto-translate. |
| `~global_autotl on\|off` | super admin | Global auto-translate switch. |

Admins and super-admins are configured in `config.toml` — no hard-coded users.

## Setup

Requires **Python 3.12+**.

1. **Install dependencies:** `poetry install` (or `pip install -r requirements.txt`).
2. **Configure** — copy the templates and fill them in:
   ```bash
   cp config.example.toml config.toml   # channels, admins, paths, persona/llm knobs
   cp .env.example .env                 # Twitch client id/secret, DeepL key
   ```
   - Create a Twitch app at <https://dev.twitch.tv/console/apps> for a client ID
     and secret. Add `http://localhost:3000` to its *OAuth Redirect URLs*.
   - For translation, get a free **DeepL API key** (<https://www.deepl.com/pro-api>)
     and set `DEEPL_API_KEY` in `.env`. (Google Cloud Translation works as an
     optional fallback via a service-account JSON.)
3. **Initialise the database:** `python scripts/init_db.py`
4. **Authorize the bot account** (one-time, fully local):
   `python scripts/get_initial_token.py` — opens Twitch in your browser,
   captures the redirect locally, saves tokens; refreshes automatically after.
5. **Run it:** `python chatbot.py`

The bot needs no web server — it talks to Twitch directly over IRC/Helix.

### Personas (optional)

The Markov persona (`~markov`) and archive commands work with no extra setup
once you've ingested logs (`python scripts/ingest_chatterino.py`). The LLM
personas (`~persona`/`~hyper`) need an OpenAI-compatible endpoint — point
`[llm].endpoint` in `config.toml` at a local **LM Studio** server (load a model,
Developer tab → Start Server) or any compatible API.

### Windows quickstart & background running

Double-click instead of using the terminal:

| Script | Does |
| --- | --- |
| `1-setup.bat` | Create the venv, install deps, build the DB (run once). |
| `2-login.bat` | Authorize the bot account (one-time). |
| `3-run.bat` | Run the bot in a visible window. |
| `run-background.vbs` | Run the bot **hidden**, restarting it if it crashes, logging to `data/bot.log`. |
| `show-log.bat` | Live view of the background bot's log (closing it doesn't stop the bot). |
| `stop-bot.bat` | Stop the background bot. |

To auto-start at login, put a shortcut to `run-background.vbs` in your Startup
folder (`shell:startup`). Additional numbered helpers for the offline
persona/fine-tuning pipeline exist alongside these; the channel-specific ones
are kept private under `_private/` (see [docs/FINE_TUNING.md](docs/FINE_TUNING.md)).

## A note on cost

- **Detection is free** — local `lingua` (`utils/language_detect.py`), no API.
  It's a gate: a message goes to the translator only if it's *confidently not
  the target language* — judged by the **distribution** (the best foreign guess
  must win the head-to-head share against the target, `min_foreign_share`), not
  an absolute score, since lingua's absolute scores aren't comparable across
  languages. Short messages need a *single* language at high confidence
  (`min_short_confidence`), because both the detector and translator invent
  output for tiny fragments and emote names.
- **Translation** uses **DeepL** by default (free ~500k chars/month), with
  Google Cloud Translation as an optional fallback. Both sit behind
  `services/translators.py`, so swapping or rotating backends is contained;
  self-host options: **LibreTranslate** or **Argos Translate** (offline).
- **Personas are free to run locally.** The LLM personas use a local model (LM
  Studio), so there's no per-message charge. Fine-tuning is the only paid step,
  and it's a one-time rented-GPU run (a few dollars), not a subscription.

## Known issues & notes

- **Twitch whispers are unreliable by design** — Twitch rate-limits/blocks bot
  whispers to fight spam; the account also needs a verified phone and the
  `user:manage:whispers` scope. Practice-mode whispers can just not arrive.
- **Romanization is approximate** — good enough to read along, not authoritative.
- **Language codes differ between services** — normalized to one scheme with
  small alias maps; uncommon languages can still slip through.
- **Emotes can skew detection** — the bot strips @mentions, URLs, known chatters'
  names and the channel's 7TV/BTTV/FFZ emotes, and catches stray emote names
  structurally (internal-capital or long all-caps tokens), but it isn't perfect.
- **Personas reflect their source chat.** They recombine/continue real messages,
  so quality and content track the logs; the output filter blocks a denylist
  before posting, but you own what your bot posts on Twitch.
- The dials live in `config.toml`: `translation.min_foreign_share` (auto-translate
  sensitivity), `translation.min_short_confidence`, and the `[persona]`/`[llm]`
  sections (reaction chance, exemplar counts, endpoint).

## Roadmap & docs

Design docs live in [`docs/`](docs/) — start with
[HANDOFF.md](docs/HANDOFF.md) for the current state of the persona/archive work.
Also: the [chat archive design](docs/CHAT_ARCHIVE.md) (SQLite + FTS5 + the
Chatterino log format), the [persona roadmap](docs/PERSONA_BOT_ROADMAP.md),
[fine-tuning](docs/FINE_TUNING.md), a
[chat-personality research note](docs/CHAT_PERSONALITY_RESEARCH.md), an
[idea bank](docs/IDEA_BANK.md), and the planned
[repo reorganization](docs/REORG_PLAN.md).

## License

[MIT](LICENSE)
