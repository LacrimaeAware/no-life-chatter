# Reorganization plan (not yet executed)

The repo grew from a translation bot into a multi-tool (translation + chat
archive + personas + a fine-tuning pipeline + log scrapers), and the root has
accumulated loose scripts and a dozen numbered `.bat` files. This is the agreed
target structure and a mechanical checklist to get there. **Nothing here is
done yet** — execute when convenient (Phase 1 is safe anytime; Phase 2 wants
the bot stopped).

## The mental model: two products, one database

- **Runtime Twitch bot** — `chatbot.py` + `commands/ services/ utils/`. Loaded
  by the live process; must stay stable. **It only imports from
  `utils/`, `services/`, `commands/`** — never from `scripts/` or any `.bat`.
- **Offline data/ML pipeline** — ingest, archive queries, persona export,
  fine-tune, eval. Currently scattered across `scripts/` and root bats. This is
  what gets grouped.

Root should hold only: configs, the core bot modules, and the *operate-the-bot*
launchers.

## Hard constraints (do not break)

1. **Auto-start**: `shell:startup\NoLifeChatter.lnk` → `run-background.vbs`,
   which derives the project dir from its own location. Keep `run-background.vbs`
   (and the chain it launches, `_bot-loop.bat` → `chatbot.py`) at the repo root.
   Decision: **all runtime launchers stay at root.**
2. **Command auto-discovery**: `command_registry.py` does
   `import_module("commands.<name>")` and `help.py` re-imports the same. The
   `commands/` package must stay importable as `commands.*`.
3. **Live bot**: moving anything under `scripts/` or any `.bat` does NOT affect
   the running bot (it doesn't import them). Moving anything under
   `utils/ services/ commands/` or the root `.py` files DOES — that's Phase 2.
4. **Privacy**: personal launchers (hard-coded channels/rosters) live under
   gitignored `_private/`. Keep them there.
5. **Path-relative bats**: every `.bat` uses `cd /d "%~dp0"`. Any bat that moves
   deeper must become `cd /d "%~dp0..\.."` (back to repo root) so its
   `.venv\Scripts\python.exe` and `scripts\...` references still resolve.

## Target structure

```
NoLifeChatter/
  README.md  LICENSE  pyproject.toml  requirements.txt  .gitignore
  config.example.toml  .env.example
  # core bot (UNCHANGED — runtime imports depend on these paths)
  chatbot.py  handlers.py  command_processor.py  command_registry.py
  config.py  auth.py
  commands/  services/  utils/  data/  docs/
  # operate-the-bot launchers (STAY at root; run-background.vbs is Startup-pinned)
  1-setup.bat  2-login.bat  3-run.bat
  run-background.vbs  _bot-loop.bat  show-log.bat  stop-bot.bat
  scripts/
    setup/     init_db.py  get_initial_token.py  check_user_settings.py
    archive/   ingest_chatterino.py  ask_archive.py  download_zonian_user_logs.py
    persona/   persona_preview.py  persona_rag_preview.py
    finetune/  export_persona_sft.py  train_persona_lora_unsloth.py
               runpod_train_persona_lora.sh  RUN_ME_ON_RUNPOD.sh
               smoke_test_persona_lora_with_rag.py
               runpod_*.sh  runpod_*_command.txt
  tools/       6-preview-persona-rag.bat        # dev/diagnostic launchers
  _private/finetune/   4,5,7,8,9,10,11,12,13 bats + private eval scripts (already here)
```

## Phase 1 — safe tidy (do anytime; live bot unaffected)

Group `scripts/` by concern and move experiment bats off the root. No core
Python moves, no bot restart.

**Moves** (`git mv`):
- `scripts/init_db.py`, `get_initial_token.py`, `check_user_settings.py` → `scripts/setup/`
- `scripts/ingest_chatterino.py`, `ask_archive.py`, `download_zonian_user_logs.py` → `scripts/archive/`
- `scripts/persona_preview.py`, `persona_rag_preview.py` → `scripts/persona/`
- `scripts/export_persona_sft.py`, `train_persona_lora_unsloth.py`,
  `runpod_*.sh`, `RUN_ME_ON_RUNPOD.sh`, `smoke_test_persona_lora_with_rag.py`,
  `runpod_*_command.txt` → `scripts/finetune/`
- root `6-preview-persona-rag.bat` → `tools/` (or `_private/finetune/` if you'd
  rather it not ship publicly; it's a generic dev tool, no personal data)
- root `5/7/8/11-*.bat` → `_private/finetune/` (finetune/runpod helpers, join
  the 4/9/10/12/13 already there)

**References to fix after each move** (this is the whole risk surface):
- Each script's `sys.path.insert(0, dirname(dirname(abspath(__file__))))` adds
  ONE `dirname` now that they're one level deeper → make it
  `dirname(dirname(dirname(abspath(__file__))))` so the repo root is still on
  the path for `import config` / `from utils import ...`.
- Any `.bat` that moved deeper: `cd /d "%~dp0..\.."` and confirm its
  `scripts\<sub>\<file>.py` path.
- Bats that stay at root but now call relocated scripts: update the path, e.g.
  `scripts\persona\persona_rag_preview.py`.
- Cross-imports: `smoke_test_persona_lora_with_rag.py` and the private compare/
  export scripts import `PROMPTS` from `smoke_test_persona_lora` and call
  `persona_llm` — verify the sibling import + the 3×`dirname` bootstrap.
- Docs: update path references in `HANDOFF.md`, `CHAT_ARCHIVE.md`,
  `FINE_TUNING.md`, `RUNPOD_FINE_TUNE_README.txt`, and the pilot bat's copy list.
- The pilot bat (`_private/finetune/4-export-finetune-pilot.bat`) copies helper
  scripts into the bundle by path — update to `scripts\finetune\...`.

**Verify**: `python -m py_compile` every moved script; run
`scripts/setup/init_db.py --help` style smoke; double-click one relocated bat.

## Phase 2 — core package (deferred; needs bot stopped)

Move the core bot modules into a `nolifechatter/` package for a clean import
root. **High churn, mostly-cosmetic benefit — only do it when the persona
experiment has settled and the bot can be stopped.** It touches:
- every `import config` / `from utils import` / `from services import` /
  `from commands import` across the codebase;
- `command_registry.import_module("commands.<name>")` →
  `import_module("nolifechatter.commands.<name>")`;
- `config.py`'s `BASE_DIR` (it resolves data/ and config.toml relative to the
  module file — must still point at the repo root, not the package dir);
- the supervisor command `_bot-loop.bat`: `python chatbot.py` →
  `python -m nolifechatter` (needs a `__main__.py`);
- `1/2/3-*.bat` python invocations;
- `pyproject.toml` packaging.

Because the live bot imports these, do it with the bot stopped, then restart via
`run-background.vbs` and confirm `Joined channels` + a `~ping` in chat.

## Recommended sequencing

1. Phase 1 whenever (it's reversible and doesn't touch the running bot).
2. Leave Phase 2 until personas are "done enough" that you're not restarting the
   bot constantly — doing it mid-experiment just adds breakage risk for polish.
