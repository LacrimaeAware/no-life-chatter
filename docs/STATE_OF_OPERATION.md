# State of operation — 2026-06-12

Single source of truth for the current system: what exists, how it fits, what's
pending, and what's intentionally deferred. Written for a session that will
organize/refactor the repo. Pairs with [HANDOFF.md](HANDOFF.md) (persona/archive
narrative) and the private `_private/WORK_BUCKETS.md` (task spectrum).

## What this project is

A Python `twitchio` Twitch bot, two halves over one local SQLite archive:
1. **Live translation** (the original bot) — DeepL/Google with local detection,
   practice mode, romanization, per-user speaker profiles.
2. **Chat-archive analytics + personas** (everything built recently) — a ~6M
   message archive powering stylometry, voice/semantic profiling, persona
   generation, and a fine-tune pipeline.

Runs on the user's Windows PC. Uses a **local** LM Studio server for both chat
(persona LLM) and embeddings (nomic) — nothing leaves the machine.

## How it starts / stays alive (operational)

- **Autostart**: a Startup-folder shortcut `NoLifeChatter.lnk` → `wscript.exe`
  → `run-background.vbs` → `_bot-loop.bat` (the keep-alive loop). Launches
  hidden at login.
- `_bot-loop.bat` runs `chatbot.py`, restarts it 10s after any exit, logs to
  `data/bot.log`. `chatbot.py` holds a **single-instance lock** (binds
  127.0.0.1:48917) so only one bot is ever connected.
- **To restart with new code**: kill the python `chatbot.py` process — the loop
  respawns it. NEVER `Start-Process chatbot.py` manually (lock collision; a
  second instance just exits — this is why a manual `3-run.bat` "launches but
  doesn't post" while the loop's bot is the real one).
- **To stop for real**: `stop-bot.bat` (until next login) or
  `stop-bot-FOREVER.bat` (also removes autostart).
- NEVER edit `_bot-loop.bat` while it runs (cmd re-reads batch files by byte
  offset → crash loop).
- Random reactions are currently OFF (`[persona] reaction_chance = 0`).

## Architecture map (where things live)

- `chatbot.py` — entry, single-instance lock, twitchio Bot, token thread.
- `command_processor.py` / `command_registry.py` — auto-discovers
  `commands/<name>.py` exposing `handle_<name>(bot, message, params)` + a
  `description`. Bare `~`/`~~` guarded.
- `services/message_service.py` — translation pipeline + random-reaction path.
- `services/llm.py` — async OpenAI-compatible client; `_chat_lock` serializes;
  `model=` param; resolves legacy `"local"` id.
- **Archive**: `utils/chat_archive.py` — SQLite+FTS5, `messages_for(author,
  channel, year)`, `said`, `search_all`, `regex_search`, `channel_members`,
  `channel_regulars`, `context_window` (time-ordered), `record_live` (buffered).
- **Stylometry**: `utils/persona_classifier.py` — TF-IDF+LR authorship
  (`classify`/`~whosaid`), and voice profiles (`build_style_profiles`,
  `profile_for`, `most_like`): per-author favorite **words + word-pairs +
  emotes**, Fightin'-Words log-odds × panel-rarity. Pickle at
  `data/unsynced/persona_classifier.pkl` holds `pipe`, `authors`, `profiles`,
  `prevalence`.
- **Semantic**: `utils/persona_embeddings.py` (person mean-vectors, centered,
  calibrated alt-flag) + `utils/persona_msg_index.py` (per-message vectors:
  burst scores + semantic retrieval). Built by
  `scripts/build_persona_embeddings.py` and `scripts/build_message_index.py`.
- **Traits**: `utils/persona_traits.py` (5 static axes) + `utils/persona_axes.py`
  (dynamic LLM-built axes, validated poles, organic merge, emote-name blend,
  burst). Custom axes saved to `data/unsynced/custom_axes.pkl`.
- **Generation**: `utils/persona_llm.py` (RAG persona engine, channel-scoped,
  semantic retrieval), `utils/persona_markov.py` (markov + fusion),
  `utils/persona_generate.py` (`~generate` recipes + saved combos).
- **Rebuild**: `scripts/rebuild_persona_artifacts.py` /
  `10-rebuild-persona-artifacts.bat` chains classifier → profiles → embeddings
  → message index → calibration.

## Commands (26)

Translation: `~autotl ~setlang ~tloutput ~chan_autotl ~global_autotl ~practice
~romanize ~speak`. Archive: `~said ~regex ~quote ~firstseen ~chatstats
~regulars ~whosaid`. Analysis: `~markers ~like ~twin ~traits ~top ~vibes`.
Personas: `~markov ~mimic ~persona ~hyper ~generate`. Util: `~help ~ping`.
All have `description`s; README table is current.

## Live data artifacts (all gitignored under data/)

| artifact | built by | feeds |
|---|---|---|
| chat_archive.db (~6M msgs) | ingest + live capture | everything |
| persona_classifier.pkl | train_classifier + build_style_profiles | ~whosaid ~markers ~like ~twin |
| persona_embeddings.pkl | build_persona_embeddings | ~vibes ~twin ~traits ~top |
| msg_index/*.npz (53, ~108MB) | build_message_index | ~top burst, semantic persona retrieval |
| custom_axes.pkl, emote_embeddings.pkl | persona_axes (lazy) | ~top dynamic axes |
| gen_combos / gen settings (settings DB) | ~generate save | saved recipes |

## PENDING — important

**Alias merges applied to config, artifacts NOT yet rebuilt.** A batch of
confirmed alt-account merges was written to `[archive.user_aliases]` in the
gitignored `config.toml` (the specifics, with the confirm/deny rationale, are
in `_private/ALT_CANDIDATES.md`). Archive *reads* honor these now, but the
classifier/profiles/embeddings/msg-index were built when those were separate
accounts. **Run `10-rebuild-persona-artifacts.bat`** (or the `.py` with
`--skip-embeddings` if LM Studio is off, then embeddings later) so the merged
accounts collapse into one identity everywhere. Spot-check a couple of the
newly-merged names with `~like` / `~persona` / `~vibes` after.

## Design intentions / open threads (for the organizer)

- **Work spectrum** in `_private/WORK_BUCKETS.md`: A = menial (delegate), B =
  mid, C = hard. Standing rules (secret-grep before commit, never launch the
  bot manually, never edit the loop live) are at its top.
- **Bot modes** family specced in `GENERATE_AND_BOT_MODES.md`: resident
  persona, regular/response/random/silent modes, `~banuser`, queue+cooldown.
  `~generate` + saved combos DONE; the rest split into buckets.
- **Embeddings research** in `CHAT_PERSONALITY_RESEARCH.md` + `IDEA_BANK.md`:
  trait axes (built), alt-detection scoring harness (needs the now-available
  labels), 2D community map for Pages (`_private/PAGES_IDEAS.md`), multilingual
  embedder swap (German confound), supervised classifiers for sensitive axes.
- **Known honest limitations**: zero-shot trait axes measure *register*, not
  ground truth (a charged-trait axis once mislabeled a blunt-but-innocent
  chatter — fixed via abliterated + validated poles, but still register-based);
  nomic is English-leaning;
  `~whosaid` on single words is noisy by design; the v2 LoRA reads unreactive
  (v3 needs conversation-context training pairs).
- **Refactor candidates** (REORG_PLAN.md): runtime (`commands/ services/
  utils/`) vs offline (`scripts/`) split; the persona_* utils have grown into a
  natural `persona/` subpackage (classifier, embeddings, msg_index, traits,
  axes, llm, markov, generate); the old translation modules predate the
  newer code's docstring/structure quality.

## Privacy

Public repo is sanitized: NO real handles in tracked code (CI of the habit:
grep staged ADDED lines before every commit). Real usernames/secrets live only
in gitignored `config.toml`, `data/`, and `_private/`. GitHub Pages enabled
(README mirror) at lacrimaeaware.github.io/NoLifeChatter.
