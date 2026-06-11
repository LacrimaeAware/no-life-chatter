# Handoff — persona/archive feature set

Single doc to bring a fresh session (or a different model) fully up to speed on
the chat-archive + persona work, with no prior conversation context. Written
2026-06-11.

## TL;DR of current state

NoLifeChatter is a Python `twitchio` Twitch bot (translation, plus a new
persona/archive feature set). The bot runs on the user's Windows PC in the
background. As of 2026-06-11 the following is **built, live, committed, and
pushed** to `github.com/LacrimaeAware/no-life-chatter` (`main`):

- A searchable **chat archive** (SQLite + FTS5) of 741k+ real messages.
- **Markov** personas (`~mimic`) — instant, no model.
- **LLM** personas (`~persona`, `~hyper`) — context-aware, run on a **local
  LM Studio** model (free, private).
- **Retrieval/RAG exemplars** - each LLM reply blends random signature lines
  with author-only messages relevant to the current chat/topic.
- **Random reactions** — the bot rarely speaks up in a chatter's persona.

What's NOT done yet (the work to pick up): an **organic reply-frequency**
system, a **Turing-test game**, and optional **fine-tuning**. See "Next work"
below.

## How to run / verify

- **Bot**: runs via `run-background.vbs` → `_bot-loop.bat` (hidden, crash-restart,
  logs to `data/bot.log`). `show-log.bat` tails it; `stop-bot.bat` stops it.
  Restart after code changes: kill the `python.exe ...chatbot.py` process (the
  supervisor relaunches it), or stop+relaunch the vbs for env changes.
- **Python**: use the project venv: `.venv\Scripts\python.exe`. On Windows set
  `PYTHONUTF8=1` for any script that prints emoji (the launcher already does).
- **Ingest more logs**: `python scripts/ingest_chatterino.py [logs-root]
  [--channels a,b,c] [--since YYYY-MM-DD]` — incremental & idempotent.
- **Query archive offline**: `python scripts/ask_archive.py said|quote|stats|search ...`
- **Preview a Markov persona (no chat post)**: `python scripts/persona_preview.py <user>`
- **Preview LLM/RAG exemplar selection (no model/chat post)**:
  `python scripts/persona_rag_preview.py <user> "topic or message" [--channel channel]`
- **LLM personas need LM Studio**: load a model, Developer tab → Start Server
  (OpenAI-compatible at `http://127.0.0.1:1234`). The bot reads
  `config.LLM_ENDPOINT`.

## In-chat commands (all auto-discovered from `commands/`)

| Command | What |
| --- | --- |
| `~said <user> <phrase>` | Did they ever say it? count + first occurrence |
| `~quote <user>` | Random real quote from their history |
| `~firstseen <user>` | Their first archived message |
| `~chatstats <user>` | Count, first/last seen, busiest hour |
| `~mimic <user>` | Markov line in their style (instant, no model) |
| `~persona <user> [msg]` | LLM persona, natural mode, context-aware |
| `~hyper <user> [msg]` | LLM persona, traits exaggerated for comedy |

`~help` auto-lists everything; `~help <cmd>` shows its `description`.

## Data

- Archive DB: `data/unsynced/chat_archive.db` (gitignored). **741,094 messages**
  from **thickpoo, duardo1, earnestsinceresugmamale, fernardo** (f3rnard0 merged
  via alias). 164 authors. duardo1 (467k) and thickpoo (213k) are the big ones.
- Source logs: Chatterino2 at
  `C:\Users\EcceNihilum\AppData\Roaming\Chatterino2\Logs\Twitch\Channels`
  (13.4 GB, 135 channels, since 2025-03-06; nothing older exists on this
  machine — user has older logs on another computer to ingest later).
- Live capture: the bot appends every message it sees (`handlers.py` →
  `chat_archive.record_live`), so the archive self-maintains.

## Architecture (files that matter)

- `utils/chat_archive.py` — schema (WAL), Chatterino parser, queries: `said`,
  `random_quote`, `first_seen`, `stats`, `search_all`, `search_author`,
  `context_before`, `latest`, `recent_authors`, `messages_for`. **FTS queries MUST use
  `CROSS JOIN` (FTS-first) — plain JOIN took 79s vs 0.007s.**
- `utils/persona_markov.py` — order-2 word chains per user, cached.
- `utils/persona_llm.py` — **many-shot LLM personas with lightweight RAG.**
  `exemplars(author)` is still a random signature sample across the author's
  whole history. Per reply, `select_exemplars()` blends that with
  `relevant_exemplars()` from `chat_archive.search_author()`, scoped strictly
  to the target author and keyed on the current chat/topic. Default prompt mix:
  150 total examples, up to 90 retrieved (`llm.relevant_exemplars`) and the
  rest random signature lines. `generate(author, channel, user_message, mode)`
  builds: system prompt ("you ARE <author>" + both exemplar sections) + user
  turn (recent `latest()` channel context + optional directed message) →
  `services/llm.chat`. Modes: `normal`, `hyper`. Output is de-quoted /
  name-stripped / single-line.
- `services/llm.py` — async client for any OpenAI-compatible `/v1/chat/completions`
  (LM Studio default). Returns None on failure (graceful).
- `services/message_service.py` — `maybe_react()` (random persona reaction,
  LLM or Markov per `reaction_use_llm`) called at the top of
  `handle_regular_message`; plus the translation pipeline.
- `utils/output_filter.py` — denylist gate (`is_clean`) applied before posting
  any persona text. Denylist in gitignored `data/unsynced/blocklist.txt`.
- `commands/{said,quote,firstseen,chatstats,mimic,persona,hyper}.py` — thin
  command wrappers.
- Config: `config.py` reads `config.toml` (gitignored; `config.example.toml` is
  the public template). Relevant sections: `[archive]`, `[persona]`, `[llm]`.
  `[archive.user_aliases]` can merge alt accounts for persona/archive queries
  without renaming channels.

## Key design facts (don't re-litigate these)

- **Personas are per-author, never merged.** `exemplars(author)`,
  `relevant_exemplars(author, ...)`, and `messages_for(author)` all filter
  strictly to the target author/alias group. `[archive.user_aliases]` lets known
  alt accounts count as one person while leaving channel names alone. The only
  other-people text in a prompt is the labeled *recent conversation* the persona
  is reacting to. If output feels "merged," it's the non-abliterated base
  model's voice bleeding through or context echo — not data mixing.
- **Signature exemplars are RANDOM across full history, not recent** (user
  preference; still implemented). Retrieval also searches the target author's
  full history, using recent chat only as the query/context, not as somebody
  else's exemplar text.
- **Two TOS walls:** (1) hosted Claude/OpenAI refuse slur generation & OpenAI
  fine-tuning rejects edgy data → use a **local** model for edgy content (the
  user runs LM Studio: Llama-3.1-8B-Instruct-Q4_K_M, Vulkan, 8k ctx). (2)
  Posting slurs to Twitch bans the bot AND the operator → every persona post is
  `output_filter.is_clean`-gated. Design rule: **generate → filter → send.**
- **Commit rules (strict):** NEVER add a `Co-Authored-By` trailer. Before every
  commit run the secret check:
  `git ls-files --cached | grep -E '^(_private/|\.env$|config\.toml$|.*\.db$|.*token_data\.json$|blocklist\.txt$)'`
  (must be empty). Never stage `config.toml`, `*.db`, `.env`, `blocklist.txt`,
  `data/unsynced/*`. `git push origin main` works (cached HTTPS creds).
- **Model swap is free:** point `[llm].endpoint` anywhere OpenAI-compatible, or
  load a different GGUF in LM Studio (e.g. an *abliterated* model for edgy
  content) — no code change.

## Known issues / perf notes

- **Prompt-processing latency**: first persona call for a user builds/caches the
  random signature sample and each call does a small FTS retrieval for relevant
  examples. The model still has to process the ~150-message prompt (a few
  seconds on the RX 5700 XT, 8 GB, Vulkan). LM Studio's prompt cache makes
  repeats faster. This is expected on this hardware, not a bug.
- The user currently runs the **non-abliterated** Llama 3.1 — it will refuse
  hard slurs (fine; the filter would block them anyway). Swap GGUF for edgy.
- `~mimic`/Markov can be too terse and has no context (by design); the LLM
  personas fixed the terseness.

## Next work (priority order, with the user's latest asks)

1. **Organic reply-frequency for conversation/reactions** — a persona shouldn't
   answer every line. Make "how often it responds" a parameter, and prefer
   replying when a message is **directed at it** (@mention / name) or continues
   a loop it's already in, rather than uniformly random. Generalize
   `reaction_chance` into: directed-at-persona detection + a response
   probability + cooldown.
2. **"Real or AI?" Turing-test game** (user's idea, and chat literally played it
   — earnest: "ok turing test, was that a message i wrote or ai generated").
   A command (e.g. `~realorai [user]`): pick a chatter, 50/50 either pull a real
   archived line or generate a persona line, post it, players guess, then
   reveal. Heuristics for the *real* pull (and ideally generated too): skip
   single emotes / 1-word / pure-link / too-short and too-long lines; prefer
   lines that "make a statement" (declarative, has a verb) so it's a fun guess,
   not a boring quote. Add a scoreboard table.
3. **Fine-tuning** (later) — one model, all personas via a `<persona=user>`
   prefix per training row; LoRA on a rented cloud GPU (hours, ~$5–20), then run
   the GGUF locally. "Train once, top up later" = re-run with new archive rows.
   Optionally distill/synthetic-data from an uncensored teacher for edgy voice.
4. **Retrieval polish** — the lightweight FTS RAG is built. Future upgrades:
   include lead-in context around retrieved lines (`context_before`) and/or add
   embeddings when plain keyword search misses semantic matches.
5. **Bigger context / model options** — LM Studio at 16k ctx for more exemplars;
   try an abliterated 8B for edgy; or hosted cheap model for the benign Q&A.

See `docs/PERSONA_BOT_ROADMAP.md` (full roadmap), `docs/CHAT_ARCHIVE.md` (data
layer + Chatterino format spec), `docs/FINE_TUNING.md` (LoRA pilot runbook),
`docs/IDEA_BANK.md` (more ideas). Private, machine-specific notes (paths,
hardware, decisions) are in gitignored `_private/PERSONA-NOTES.md`.
