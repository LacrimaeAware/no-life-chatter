# State Of Operation

Last refreshed: 2026-06-14.

Read this first after a break. Then read [ROADMAP.md](ROADMAP.md) for the
ranked next work and [COMMANDS.md](COMMANDS.md) for the live command bible.
Older handoffs and dated audits are preserved under [archive/](archive/).

## Short Version

NoLifeChatter is a Windows-hosted Python Twitch bot with two connected halves:

1. Live translation and language-practice tools.
2. A local chat archive that powers search, stats, stylometry, persona
   generation, embeddings, trait axes, IQ experiments, lore QA, and fine-tune
   experiments.

The public repo is a sanitized showcase. Real config, archive data, generated
reports, user rosters, aliases, logs, tokens, and model outputs stay ignored in
`config.toml`, `data/`, and `_private/`.

## Current Runtime

- The bot runs through `run-background.vbs` -> `_bot-loop.bat` -> `chatbot.py`.
- `_bot-loop.bat` restarts the bot after exits and logs to `data/bot.log`.
- `chatbot.py` has a single-instance lock on `127.0.0.1:48917`, so manual
  launches normally exit if the background bot is already alive.
- To reload code/config, stop the `python.exe ... chatbot.py` worker and let
  the loop restart it. Use `stop-bot.bat` for an intentional stop.
- Random ambient persona reactions are currently config-driven, not chat-command
  driven. Resident persona commands are still planned.

## Active Surfaces

- `commands/`: auto-discovered chat commands. `scripts/audit_commands.py` checks
  imports, async handlers, descriptions, and `docs/COMMANDS.md` coverage.
- `services/`: Twitch message handling, translation, local LLM calls, emotes,
  and ambient reaction plumbing.
- `utils/chat_archive.py`: SQLite + FTS5 archive, live capture, imports,
  search, regex, quotes, stats, source-aware context windows, and retrieval.
- `utils/persona_llm.py`: many-shot local-LLM persona engine with archive
  retrieval, recent chat context, copy checks, candidate selection, and private
  JSONL logs.
- `utils/persona_markov.py`: local Markov personas.
- `utils/persona_classifier.py`: authorship classifier and lexical voice
  profiles for `~whosaid`, `~markers`, and `~like`.
- `utils/persona_embeddings.py`: person-level semantic vectors.
- `utils/persona_msg_index.py`: per-message or utterance semantic vectors for
  semantic retrieval, `~why`, and burst leaderboards.
- `utils/persona_axes.py` / `utils/persona_traits.py`: built-in and dynamic
  trait axes.
- `utils/persona_generate.py`: `~generate` recipe system and saved combos.
- `utils/archive_qa.py`: evidence-backed `~askchat` reports over fact-bank
  rows, broad archive hits, near matches, and emotes.
- `scripts/`: offline ingestion, rebuilds, evals, fine-tune helpers, audits,
  fact-bank tooling, and smoke checks.
- `index.html` + `assets/`: public GitHub Pages showcase with anonymized
  visuals.

## Commands

The source of truth is [COMMANDS.md](COMMANDS.md). High-level groups:

- Translation/language: `~autotl`, `~setlang`, `~tloutput`, `~chan_autotl`,
  `~global_autotl`, `~practice`, `~romanize`, `~speak`.
- Archive/search/lore: `~said`, `~saidnext`, `~regex`, `~userregex`, `~quote`,
  `~firstseen`, `~chatstats`, `~regulars`, `~askchat`.
- Persona/generation: `~markov`, `~mimic`, `~persona`, `~hyper`, `~generate`.
- Analysis: `~whosaid`, `~markers`, `~like`, `~twin`, `~traits`, `~top`,
  `~vibes`, `~distinct`, `~why`, `~emote`, `~irony`, `~iq`.
- Moderation/utility: `~help`, `~ping`, `~artifacts`, `~banuser`,
  `~unbanuser`, `~warnings`.

`~botmode`, `~botpersona`, `~botcontext`, and `~botchance` are documented as a
planned resident-persona layer in [GENERATE_AND_BOT_MODES.md](GENERATE_AND_BOT_MODES.md).
They are not live commands yet.

## Current Build State

- `~askchat` is live as the first archive-QA/lore command.
- `~artifacts` is live and mirrors `scripts/artifact_status.py`.
- `scripts/audit_commands.py` is live and should be run before commits that
  touch commands or docs.
- `scripts/freshness_check.py` is the repo-level freshness wrapper for command
  docs, generated artifact status, docs layout, rebuild logs, and git dirtiness.
- The fact bank exists as candidate evidence rows. It is not a truth database
  yet.
- The held-out reply eval harness exists. A serious baseline run is still a
  next-step item.
- The full artifact rebuild should be checked before trusting rankings after
  alias/filter/semantic-unit changes.
- **Embedding-geometry pass (2026-06-14, see
  [RESEARCH_TO_APPLIED.md](RESEARCH_TO_APPLIED.md)):**
  - `scripts/eval_geometry.py` is the geometry dial — anisotropy, axis
    collinearity, axis-score entanglement, ABTT safety guard. Run it before/after
    any embedding-space change; it reads on-disk artifacts (embedder needed only
    for the axis section).
  - Trait axes are now decorrelated with **Löwdin symmetric orthogonalization**
    in `persona_traits.ortho_axis_vectors()` (single source of truth).
    `traits_for`, `leaderboard`, `axis_scores`, and `burst_scores` all route
    through it, so `~traits`/`~top`/`~top burst` finally agree. Axis-score
    entanglement dropped 0.483 -> 0.249.
  - `persona_embeddings._centered()` applies an **ABTT-k isotropy correction**
    (`ABTT_K = 2`, chosen empirically — the effect is non-monotonic). Every
    cosine consumer inherits it.
  - `archive_qa.build_report` (author path) now fuses bm25 + dense semantic
    retrieval by **Reciprocal Rank Fusion** (`_rrf_author_hits`); bm25 stays an
    independent lane and dense adds paraphrase recall. Safe no-op when the
    message index or embedder is down.
  - `scripts/irony_confound.py` measured the charged-axis irony confound: the
    zero-shot "ironic" axis cannot detect deadpan/charged irony, so `~traits`/
    `~top` on charged axes read words not intent, and naive irony-discounting
    does not work (corr +0.17). Recorded as a **known limitation** in
    `docs/GROUND_TRUTH.md`; real fix is a supervised irony detector from the
    oracle queue (ROADMAP item 8).
  - **Self-contradiction reliability flag (shipped).**
    `persona_msg_index.contradiction_scores(axis)` measures both-pole occupancy
    from the per-message clouds (the no-oracle "performative person" signal), and
    `~traits` now marks a charged-axis lean ⚡ when the chatter also lives at the
    opposite pole. Diagnostic: `scripts/contradiction.py`. First slice of the
    distributional person model.
  - **Data-driven axis discovery (research finding, not yet a command).**
    `scripts/discover_axes.py` (unsupervised PCA/ICA over person vectors) showed
    the embedding space is dominated by TOPIC and LANGUAGE, not personality — the
    ceiling on embedding-based personas. `scripts/behavior_axes.py` discovers
    personality axes from topic-free BEHAVIORAL features (caps/emote/length/
    mentions/profanity/doubling rates); the behavioral axes are ~0.22 correlated
    with the topic axes (mostly independent) and match human reads. This is the
    foundation for the "better axes" direction. `scripts/person_cards.py` renders
    per-person cards (readout + real messages) for labeling; labels in
    `_private/PERSON_LABELS.md`. See `docs/RESEARCH_TO_APPLIED.md` §7.

## Artifact Rule

After changing aliases, message filters, embedding model, roster thresholds, or
semantic units, rebuild generated artifacts before trusting analysis commands:

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_persona_artifacts.py --semantic-unit utterance --continue-on-error
```

For a background run:

```powershell
.\scripts\start_rebuild_background.ps1 -SemanticUnit utterance
```

Then restart the bot worker so cached artifacts are reloaded.

## Next Work

The active ranking is in [ROADMAP.md](ROADMAP.md). Current top items:

1. Finish/verify the utterance-unit artifact rebuild.
2. Run a held-out reply baseline.
3. Improve archive-QA/lore ranking and contradiction handling.
4. Build a persona output reranker.
5. Build fact-bank v2.
6. Add IQ receipts.
7. Implement resident persona controls.

## Privacy / Public Boundary

Never stage or publish:

- `.env`
- `config.toml`
- `_private/`
- `data/`
- `*.db`
- token files
- raw logs
- private model outputs
- private blocklists

Before committing, check staged files and staged added lines:

```powershell
git diff --cached --name-only
git diff --cached --unified=0
```

Public docs should stay architecture/product oriented and anonymized. Do not
publish raw chat evidence, exact private aliases, private queues, tokens, or
user-identifying generated reports.

## Return Checklist

```powershell
.\.venv\Scripts\python.exe scripts\freshness_check.py
.\.venv\Scripts\python.exe -m unittest tests.test_pure_functions
```

If the freshness check reports stale artifacts, inspect `~artifacts`, latest
`data/unsynced/rebuild_persona_artifacts_*.log`, or run
`scripts/artifact_status.py` directly.
