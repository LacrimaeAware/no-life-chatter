# NoLifeChatter

A Twitch chat bot that translates messages on the fly — with a built-in
**language-learning "practice mode"** that turns a stream's chat into a little
study tool.

> This is a small personal project I built a while back, cleaned up and
> published as a showcase. It isn't a polished product meant for daily use; it's
> here as a record of something I made and as a reasonable example of a modular
> Python bot. Feel free to read it, run it, or borrow ideas from it.

## What it does

- **Auto-translation** — flags messages that aren't already in the target
  language and translates them (e.g. into English) via DeepL (or Google). Runs
  per-user, per-channel, or globally. Detection is local/free, so messages
  already in the target language never hit a translation API.
- **Practice mode** — for language learners. When you write in your native
  language, the bot privately *whispers* you the translation in the language(s)
  you're studying. When you write in a language you're learning, it posts the
  native translation to chat so others can follow along.
- **Romanization** — optional romanized readings for non-Latin scripts:
  Japanese (Hepburn, via `pykakasi`), Korean (Revised Romanization), and Thai
  (via `pythainlp`, with word segmentation).
- **Flexible output** — send translations to the same channel, to another
  channel, or as a whisper.
- **Speaker profiles** — the bot learns which languages each user actually
  writes in (a simple count, no ratios). Once you've written a language enough
  times, even your short messages in it get translated; everyone else needs a
  longer, clearly-foreign message. `~speak <lang>` flags it instantly.
- **Per-user / per-channel / global settings** persisted in SQLite.

## How it works

```
chatbot.py                 # entry point: starts the bot + token-refresh thread
├── config.py              # loads config.toml (+ .env secrets) — no hard-coded data
├── handlers.py            # routes each message: command vs. regular chat
├── command_processor.py   # parses "~command args" and dispatches
├── command_registry.py    # auto-discovers command modules in commands/
├── commands/              # one file per command (~help, ~practice, ~setlang, ...)
├── services/
│   └── message_service.py # translation + practice-mode logic, whispers
└── utils/
    ├── token_manager.py   # keeps the Twitch OAuth token fresh in the background
    ├── user_settings.py   # SQLite read/write for per-user settings
    ├── romanize.py        # romanization dispatcher
    └── roman_scripts/     # per-language romanizers (ja / ko / th)
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
| `~said <user> <phrase>` | anyone | Search the chat archive: did they ever say it, and when first. |
| `~quote <user>` | anyone | Random real quote from the chat archive. |
| `~firstseen <user>` | anyone | A user's first archived message. |
| `~chatstats <user>` | anyone | Archive stats: count, first/last seen, busiest hour. |
| `~mimic <user>` | anyone | Quick bot-made line in a chatter's style (local Markov, instant, no model needed). |
| `~persona <user> [msg]` | anyone | Talk to an AI persona of a chatter (LLM, context-aware, natural). |
| `~hyper <user> [msg]` | anyone | Same, but their traits exaggerated for comedy. |
| `~autotl` | admin | Toggle auto-translate for yourself. |
| `~setlang <LANG>` | admin | Set your translation target language (e.g. `EN`). |
| `~tloutput local\|whisper\|channel <name>` | admin | Where your translations are sent. |
| `~chan_autotl <channel> on\|off` | super admin | Per-channel auto-translate. |
| `~global_autotl on\|off` | super admin | Global auto-translate switch. |

Admins and super-admins are configured in `config.toml` — there are no
hard-coded users.

## Setup

Requires **Python 3.12+**.

1. **Install dependencies** (with [Poetry](https://python-poetry.org/)):

   ```bash
   poetry install
   ```

2. **Configure** — copy the templates and fill them in:

   ```bash
   cp config.example.toml config.toml   # channels, admins, paths
   cp .env.example .env                 # Twitch client id/secret
   ```

   - Create a Twitch application at <https://dev.twitch.tv/console/apps> to get
     a **client ID** and **client secret**. Add `http://localhost:3000` to its
     *OAuth Redirect URLs*.
   - For translation, get a free **DeepL API key**
     (<https://www.deepl.com/pro-api>) and set `DEEPL_API_KEY` in `.env`.
     That's the easy path. (Optionally, you can also/instead use Google Cloud
     Translation by pointing `GOOGLE_APPLICATION_CREDENTIALS` at a
     service-account JSON; it's used as a fallback.)

3. **Initialise the database:**

   ```bash
   python scripts/init_db.py
   ```

4. **Authorize the bot account** (one-time, fully local):

   ```bash
   python scripts/get_initial_token.py
   ```

   This opens Twitch in your browser, captures the redirect locally, and saves
   the tokens. From then on the bot refreshes them automatically.

5. **Run it:**

   ```bash
   python chatbot.py
   ```

The bot runs anywhere Python does and doesn't need a web server — it talks to
Twitch directly over IRC/Helix.

### Windows quickstart & background running

If you're on Windows, double-click these instead of using the terminal:

| Script | Does |
| --- | --- |
| `1-setup.bat` | Create the venv, install deps, build the DB (run once). |
| `2-login.bat` | Authorize the bot account (one-time). |
| `3-run.bat` | Run the bot in a visible window. |
| `run-background.vbs` | Run the bot **hidden** (no window), restarting it if it crashes, logging to `data/bot.log`. |
| `show-log.bat` | Live view of the background bot's log (closing it doesn't stop the bot). |
| `stop-bot.bat` | Stop the background bot. |

To start the bot automatically at login, put a shortcut to `run-background.vbs`
in your Startup folder (`shell:startup`).

## A note on cost

- **Detection is free** — it runs locally via `lingua`
  (`utils/language_detect.py`), with no API and no per-message charge. It's used
  only as a gate: a message is sent to the translator only if it's *confidently
  not the target language*. The test is about the **distribution**, not an
  absolute score — the best foreign guess has to win the head-to-head *share*
  against the target (`best / (best + target) ≥ min_foreign_share`). That matters
  because lingua's absolute scores aren't comparable across languages (German
  routinely scores higher than Spanish for equally-clear text), so a flat
  absolute floor under-fires for some languages while letting English-ish junk
  through for others. Detection doesn't need to pick the exact foreign language —
  only rule out the target — because the translator re-detects the source
  itself. **Short messages are held to a stricter bar**: a one-to-three-word
  message is only translated when a *single* language is detected with high
  absolute confidence (`min_short_confidence`). Both the detector and the
  translator tend to invent a "translation" for tiny fragments and for emote
  names, so unless something like "danke schön" or "buongiorno" is clearly
  there, short text is left alone.
- **Translation** uses **DeepL** by default (`DEEPL_API_KEY`), which has a free
  tier (~500k characters/month). **Google Cloud Translation** (also free up to a
  monthly limit) can be used as a fallback via a service-account key.

Both providers sit behind `services/translators.py`, so adding another — or
rotating between them to stretch multiple free tiers — is a contained change.
Other options if you'd rather self-host: **LibreTranslate** (open source) or
**Argos Translate** (fully offline, no API).

## Known issues & notes

- **Twitch whispers are unreliable by design.** Twitch silently rate-limits or
  blocks whispers from bots to fight spam, sometimes with no clear error. The
  sending account also needs a verified phone number and the
  `user:manage:whispers` scope. Practice-mode whispers can just… not arrive.
  This is on Twitch's side, not the bot's.
- **Romanization is approximate.** Japanese and Thai readings depend on the
  underlying libraries and won't always match a textbook; Korean uses a
  straight Hangul-to-Latin mapping. It's "good enough to read along," not
  authoritative. This part could still use work.
- **Language codes differ between services.** `lingua`, DeepL, Google, and what
  users type don't all use the same codes (e.g. DeepL wants `EN-US` and `ZH`).
  The bot normalizes to one uppercase scheme with small alias maps, but uncommon
  languages can still slip through.
- **Emotes can skew detection.** Twitch and third-party emotes are just words
  inside a message, so a line with an emote in it can confuse the detector. The
  bot strips @mentions, URLs, known chatters' names, and the channel's 7TV /
  BTTV / FFZ emotes before detecting. Emotes that aren't in those lists (e.g.
  native Twitch emotes, which the bot doesn't enumerate) are caught by a
  structural heuristic — tokens with internal capitals (`CuldBeWorthIt`,
  `hesRight`) or long all-caps mashes (`TELLMEHEDIDNTJUSTSAYTHAT`) are dropped,
  since real words don't look like that. It isn't perfect, but combined with the
  short-message rule above it stops the common case of an emote name being
  "translated" into a random phrase.
- **The `translation.min_foreign_share` knob** (in `config.toml`) is the dial
  for channel auto-translate: how decisively the best foreign guess must beat the
  target in the head-to-head share before the bot acts (default `0.63`). Lower it
  to catch more, raise it to skip more. (`min_confidence` still governs the
  separate practice-mode gate.)

## Roadmap & ideas

Design docs live in [`docs/`](docs/): a searchable
[chat archive](docs/CHAT_ARCHIVE.md) (SQLite + FTS5 over Chatterino logs and
live chat — built; it powers the `~said`/`~quote`/`~firstseen`/`~chatstats`
commands), a [persona bot roadmap](docs/PERSONA_BOT_ROADMAP.md) (per-user chat
personas, rare in-character reactions, playful psychometrics, trivia), and an
[idea bank](docs/IDEA_BANK.md) of smaller things.

## License

[MIT](LICENSE)
