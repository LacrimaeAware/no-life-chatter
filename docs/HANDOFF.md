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
- **Markov** personas (`~markov`, alias `~mimic`) — instant,
  explicit-command only, no model.
- **LLM** personas (`~persona`, `~hyper`) — context-aware, run on a **local
  LM Studio** model (free, private).
- **Retrieval/RAG exemplars** - each LLM reply blends random signature lines
  with author-only messages relevant to the current chat/topic.
- **Organic reaction knobs** — random reactions can use ambient chance, higher
  directed-at-persona chance, cooldowns, and optional one-line follow-ups.
- **Random reactions** — the bot rarely speaks up in a chatter's persona.

What's NOT done yet (the work to pick up): a **Turing-test game**, a **frozen
scored eval** over the smoke cases, and archive/general-knowledge Q&A. See
"Next work" below.

**2026-06-11 (latest) — stylometry suite, live A/B, ops hardening:**

- **Archive grew to ~3M messages** (24 channels) via a mirror-log import.
- **Authorship classifier** (TF-IDF char 2-5 + word 1-2 grams -> logistic
  regression; ~22 authors x 5k random msgs; ~51% top-1 vs 4.5% random) powers
  `~whosaid`, which ranks only chatters active in the current channel
  (renormalized) unless `anyone` is given.
- **Voice profiles** power `~markers` (favorite words + word-pairs) and
  `~like` (who shares your distinctive voice; alt-detector — it has surfaced
  real alt accounts with shared-catchphrase evidence). Mechanics: Fightin'
  Words log-odds vs a shared background over ALL of a person's messages,
  weighted by panel rarity (terms >=80% of the panel uses can never be
  markers), with URL/username/@-mention/invisible-char filtering (chat
  clients hide U+E0000 inside mentions — strip or you mint decapitated
  usernames as "words"). Scopable per chat (`chat=`, defaults to the current
  channel) and per year (`year=`).
- **LoRA v2 merged to GGUF** and A/B'd vs plain Llama 3.1 8B on the same RAG
  prompt: plain Llama won on reactivity (~26% vs ~16% reads-as-them; the
  adapter trained without conversation context and partially ignores it).
  v3 must include preceding-chat context in training pairs and exclude
  bot-command lines ($gpt etc. — now also rejected at generation time and
  filtered from the SFT export). A **live A/B** (`[llm] ab_models`) rolls a
  model per generation and tags posted lines #llama/#lora so chat judges.
- **Ops**: chatbot.py holds a single-instance lock (port 48917) — the
  keep-alive loop (`_bot-loop.bat`, login autostart) plus manual launches had
  the bot double-posting; now extra instances exit immediately. NEVER launch
  chatbot.py by hand: kill the python process and the loop respawns it.
  `stop-bot-FOREVER.bat` = kill + stop loop + remove autostart. Newer LM
  Studio rejects model id "local" — config pins a real id, with auto-detect
  fallback. Random reactions currently OFF (`reaction_chance = 0`).
- **Next direction:** embedding-based voice/topic space (local nomic
  embeddings via LM Studio) -> semantic ~like, personality clusters/maps. See
  IDEA_BANK "Embedding-based voice/topic space".

**2026-06-11 (late): retrieval/diagnosis fixes landed** — addressing the
smoke-test findings: (1) query hygiene (question-scaffolding stopwords,
@mention/emote-token stripping, context-author exclusion) so retrieval anchors
on TOPIC words instead of echoing the question ("hows/treating/you" no longer
retrieve the author asking how things are going); (2) post-ranking of hits
(topic-term overlap + conversation-sized shape, junk dropped); (3) **evidence
snippets** — top hits expanded into ±2-line chat moments with the author's
line marked `>>`, teaching how they RESPOND, not just vocabulary; (4) stable
seeded signature core (80%) + fresh tail (20%) — kills the random-sample
lottery that made the same persona great one day and mush the next; junk
exemplars (other-bot `<` commands, links, ping+emote-only lines) filtered;
(5) engage rule in the prompt (react to direct questions in-character, never
ignore) + anti-bleed rule and guard (candidates echoing another chatter's
context line ≥0.9 similarity are rejected); (6) **candidate selection** —
`[llm] candidates = 2` samples per reply, best valid one posted. Verified on
the exact failing smoke case (a user asked a game question): retrieval now
returns that game's opinions and the live answer was a real, in-voice reply.

## How to run / verify

- **Bot**: runs via `run-background.vbs` → `_bot-loop.bat` (hidden, crash-restart,
  logs to `data/bot.log`). `show-log.bat` tails it; `stop-bot.bat` stops it.
  Restart after code changes: kill the `python.exe ...chatbot.py` process (the
  supervisor relaunches it), or stop+relaunch the vbs for env changes.
- **Python**: use the project venv: `.venv\Scripts\python.exe`. On Windows set
  `PYTHONUTF8=1` for any script that prints emoji (the launcher already does).
- **Ingest more logs**: `python scripts/ingest_chatterino.py [logs-root]
  [--channels a,b,c] [--since YYYY-MM-DD]` — incremental & idempotent.
- **Download public mirror logs for a channel's users**:
  `scripts/download_zonian_user_logs.py --channel <ch> --from-archive` selects
  users already present in the local archive for that channel, then optionally
  imports. Raw downloads go under gitignored
  `data/unsynced/external_logs/zonian/`. (Convenience launchers that hard-code
  a specific channel + roster live privately under `_private/`.)
- **Cross-channel pull for one channel's roster**: same script with
  `--users-from-channel <homechannel>` selects the user list from your home
  channel's local archive while pulling from other channels.
  Import dedupe is two-layer: exact `(channel, author, sent_at, content)` first,
  then substantial same-author/channel normalized text within
  `--dedupe-window-hours` (default 12) so UTC/local timestamp shifts do not
  duplicate older overlapping logs.
- **Query archive offline**: `python scripts/ask_archive.py said|quote|stats|search ...`
- **Preview a Markov persona (no chat post)**: `python scripts/persona_preview.py <user>`
- **Preview LLM/RAG exemplar selection (no model/chat post)**:
  double-click `6-preview-persona-rag.bat`, or run
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
| `~markov <user>` | Markov line in their style (instant, no model) |
| `~mimic <user>` | Alias for `~markov` |
| `~persona <user> [msg]` | LLM persona, natural mode, context-aware |
| `~hyper <user> [msg]` | LLM persona, traits exaggerated for comedy |

`~help` auto-lists everything; `~help <cmd>` shows its `description`.

## Data

- Archive DB: `data/unsynced/chat_archive.db` (gitignored). ~740k messages
  across the configured channels (a couple of large channels dominate the
  count), ~160 authors, with alias-merged alt accounts.
- Source logs: Chatterino2 at
  `%AppData%\Chatterino2\Logs\Twitch\Channels` (tens of GB, ~135 channels;
  ingest only the channels you care about). Older logs on another machine can
  be ingested later — the importer is incremental.
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
  name-stripped / single-line. If output is an exact/near copy of an archived
  line, explicit `~persona` gets one cheap repair prompt using a small style
  sample instead of regenerating the full 150-example prompt; ambient random
  reactions just drop copied lines.
- `services/llm.py` — async client for any OpenAI-compatible `/v1/chat/completions`
  (LM Studio default). Returns None on failure (graceful).
- `services/message_service.py` — `maybe_react()` (random LLM persona reaction;
  Markov is explicit-command only via `~mimic`/`~markov`) called at the top of
  `handle_regular_message`; plus the translation pipeline.
- `utils/output_filter.py` — denylist gate (`is_clean`) applied before posting
  any persona text. Denylist in gitignored `data/unsynced/blocklist.txt`.
- `commands/{said,quote,firstseen,chatstats,mimic,persona,hyper}.py` — thin
  command wrappers.
- Config: `config.py` reads `config.toml` (gitignored; `config.example.toml` is
  the public template). Relevant sections: `[archive]`, `[persona]`, `[llm]`.
  `[archive.user_aliases]` can merge alt accounts for persona/archive queries
  without renaming channels.
- `services/llm.py` — serializes local LLM calls through one async lock so
  simultaneous `~persona` commands don't queue into LM Studio timeouts. Tracks
  `last_error()` for clearer in-chat failures.

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
- **RAG examples are evidence, not quotes.** `utils/chat_archive.line_match_key`
  normalizes punctuation/case/spacing so straight/curly apostrophes and tiny
  mention/emote tails do not bypass copy checks. `~said` also uses this as a
  close-match fallback when exact search finds nothing.
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
- `~markov`/`~mimic` can be too terse and has no context (by design); the LLM
  personas fixed the terseness.

## Next work (priority order, with the user's latest asks)

1. **"Real or AI?" Turing-test game** (user's idea, and chat literally played it
   — earnest: "ok turing test, was that a message i wrote or ai generated").
   A command (e.g. `~realorai [user]`): pick a chatter, 50/50 either pull a real
   archived line or generate a persona line, post it, players guess, then
   reveal. Heuristics for the *real* pull (and ideally generated too): skip
   single emotes / 1-word / pure-link / too-short and too-long lines; prefer
   lines that "make a statement" (declarative, has a verb) so it's a fun guess,
   not a boring quote. Add a scoreboard table.
2. **Fine-tuning pilot** — current pilot export is single-channel and curated:
   selected high-value chatters, bot accounts excluded, known alt accounts
   merged, max 5,000 examples per author. Train a LoRA using
   `docs/FINE_TUNING.md` and `scripts/train_persona_lora_unsloth.py`.
   Current pilot shape: 41,278 train examples, 2,186 validation examples,
   2,580 optimizer steps on
   `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, LoRA rank 16, bf16 on RTX 4090,
   prompt+completion SFT. Qwen was chosen as an ungated, low-friction pipeline
   pilot, not as the guaranteed final personality model. If outputs feel bland,
   rerun the same pipeline on Llama 3.1/3.2 8B Instruct or another preferred
   7B-9B local chat model. The first RunPod training run completed, the user
   downloaded `persona_lora_result.zip`, and `7-install-runpod-lora-result.bat`
   installed it under the gitignored private fine-tune folder. The zip is a
   LoRA adapter, not a standalone LM Studio GGUF. The first LoRA-only smoke
   test looked mixed/bland; do **not** treat it as ready for the live bot. A
   local RAG-only comparison helper now exists:
   `9-compare-lora-vs-local-rag.bat` /
   `scripts/compare_lora_smoke_with_local_rag.py`, writing the private report
   `data/unsynced/fine_tune/persona_lora_vs_local_rag.md`. The actual
   **LoRA+RAG** test path is now staged too: run
   `10-export-lora-rag-smoke-cases.bat` locally to create
   `data/unsynced/fine_tune/persona_lora_rag_smoke_cases.json`, upload it to
   `/workspace/nlc_persona/persona_lora_rag_smoke_cases.json`, then run
   `11-copy-runpod-lora-rag-smoke-command.bat` and paste the copied command in
   RunPod. That writes `/workspace/nlc_persona/persona_lora_rag_smoke_test.txt`
   without touching the live bot. After seeing those outputs, decide whether to
   keep Qwen, rerun on Llama, or adjust the export/eval prompts.
3. **Archive/general-knowledge Q&A** — a separate `~askchat`-style route for
   questions like "do we have an emote of the bottle dog?", using archive/emote
   retrieval plus a stronger answer model. Do not solve this via fine-tuning.
4. **Fine-tuning durable run** — one model, all personas via a `<persona=user>`
   prefix per training row; LoRA on a rented cloud GPU (hours, ~$5–20), then run
   the GGUF locally. "Train once, top up later" = re-run with new archive rows.
   Optionally distill/synthetic-data from an uncensored teacher for edgy voice.
5. **Organic reply polish** — basic ambient/directed chances and follow-ups are
   built. Future improvements: sleep hours, per-person activity weights, and
   thread memory so a persona continues a loop it recently joined.
6. **Retrieval polish** — the lightweight FTS RAG is built. Future upgrades:
   include lead-in context around retrieved lines (`context_before`) and/or add
   embeddings when plain keyword search misses semantic matches.
7. **Chat personality research** — saved in
   `docs/CHAT_PERSONALITY_RESEARCH.md`. Keep style/personality features separate
   from interaction/social-graph features so clusters do not simply mean "these
   people talk to each other."
8. **Emote/lore glossary RAG** — archive-Q&A/personas should eventually retrieve
   local emote meanings, shortened game names, and recurring bits from evidence
   in chat instead of guessing.
9. **Bigger context / model options** — LM Studio at 16k ctx for more exemplars;
   try an abliterated 8B for edgy; or hosted cheap model for the benign Q&A.

See `docs/PERSONA_BOT_ROADMAP.md` (full roadmap), `docs/CHAT_ARCHIVE.md` (data
layer + Chatterino format spec), `docs/FINE_TUNING.md` (LoRA pilot runbook),
`docs/CHAT_PERSONALITY_RESEARCH.md` (psychometrics/personality-map plan), and
`docs/IDEA_BANK.md` (more ideas). Private, machine-specific notes (paths,
hardware, live RunPod state, and next-session handoff details) are in gitignored
`_private/PERSONA-NOTES.md` and `_private/AI-HANDOFF-PERSONA-RUNPOD.md`.
