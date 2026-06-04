# NoLifeChatter

A Twitch chat bot that translates messages on the fly — with a built-in
**language-learning "practice mode"** that turns a stream's chat into a little
study tool.

> This is a small personal project I built a while back, cleaned up and
> published as a showcase. It isn't a polished product meant for daily use; it's
> here as a record of something I made and as a reasonable example of a modular
> Python bot. Feel free to read it, run it, or borrow ideas from it.

## What it does

- **Auto-translation** — detects the language of chat messages and translates
  them (e.g. into English) using Google Cloud Translation. Can run per-user,
  per-channel, or globally.
- **Practice mode** — for language learners. When you write in your native
  language, the bot privately *whispers* you the translation in the language(s)
  you're studying. When you write in a language you're learning, it posts the
  native translation to chat so others can follow along.
- **Romanization** — optional romanized readings for non-Latin scripts:
  Japanese (Hepburn, via `pykakasi`), Korean (Revised Romanization), and Thai
  (via `pythainlp`, with word segmentation).
- **Flexible output** — send translations to the same channel, to another
  channel, or as a whisper.
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
   - For translation, create a Google Cloud service account with the Cloud
     Translation API enabled, download its JSON key, and point
     `paths.google_credentials` (or `GOOGLE_APPLICATION_CREDENTIALS`) at it.

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

## A note on cost

- **Language detection is free** — it runs locally via `langdetect`
  (`utils/language_detect.py`), no API and no per-message charge.
- **Translation** still uses the **Google Cloud Translation API**, which is
  paid (it has a monthly free tier, but heavy chat traffic can exceed it).

If you want to drop the paid part too, the translation call is isolated in
`services/message_service.py`, so swapping providers is a small change:

- **DeepL API Free** — free tier; also auto-detects the source language.
- **LibreTranslate** — open source; self-host for free.
- **Argos Translate** — fully offline, on-device translation (no API at all).

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
- **Language codes differ between services.** `langdetect`, Google, and what
  users type don't all use the same codes. The bot normalizes to one uppercase
  scheme with a small alias map, but uncommon languages can still slip through.
- **Emotes can skew detection.** Twitch and third-party emotes are just words
  inside a message, so a mostly-German line with one English emote can confuse
  the detector. The bot already strips @mentions, URLs and known chatters'
  names and uses heuristics that work reasonably well. A fuller fix — fetching
  the channel's emote list and stripping emotes before detecting — was left out
  on purpose to avoid the extra computation, since this is aimed at translating
  slower / offline-style chat rather than fast live spam.
- **The `translation.min_confidence` knob** (in `config.toml`) is the dial for
  how sure the detector must be before the bot acts. Lower it to catch more,
  raise it to skip more.

## License

[MIT](LICENSE)
