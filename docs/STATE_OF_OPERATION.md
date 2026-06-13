# State Of Operation - 2026-06-12

This is the return-to-project map. Read this first after a break, then use
[HANDOFF.md](HANDOFF.md) for the longer persona/archive history and the private
`_private/WORK_BUCKETS.md` for task sizing.

## Short Version

NoLifeChatter is a Windows-hosted Python Twitch bot with two connected halves:

1. Live translation and language-practice tools.
2. A local chat archive that powers search, stats, stylometry, persona
   generation, embeddings, trait axes, and fine-tune experiments.

The public repo is a sanitized showcase. The real config, archive, generated
reports, user rosters, model outputs, aliases, logs, and tokens stay ignored in
`config.toml`, `data/`, and `_private/`.

## Current Runtime

- The bot runs through `run-background.vbs` -> `_bot-loop.bat` -> `chatbot.py`.
- `_bot-loop.bat` restarts the bot after exits and logs to `data/bot.log`.
- `chatbot.py` has a single-instance lock on `127.0.0.1:48917`, so manual
  launches usually exit if the background bot is already alive.
- To reload code/config, stop the `python.exe ... chatbot.py` worker and let
  the loop restart it. Do not start another bot manually.
- To stop it intentionally, use `stop-bot.bat`. To stop autostart too, use
  `stop-bot-FOREVER.bat`.
- Random ambient persona reactions are currently off in private config.

## What Exists

- `commands/`: auto-discovered chat commands.
- `services/`: Twitch message handling, translation, LLM client, emotes.
- `utils/chat_archive.py`: SQLite + FTS5 archive, live capture, imports, search,
  regex, quotes, stats, context windows, retrieval helpers.
- `utils/persona_llm.py`: local-LLM persona engine with archive retrieval,
  recent chat context, copy checks, candidate selection, and private JSONL logs.
- `utils/persona_markov.py`: quick local Markov personas.
- `utils/persona_classifier.py`: authorship classifier and lexical voice
  profiles for `~whosaid`, `~markers`, `~like`, and related tools.
- `utils/persona_embeddings.py`: person-level semantic vectors.
- `utils/persona_msg_index.py`: per-message semantic vectors for burst
  leaderboards and semantic persona retrieval.
- `utils/persona_axes.py` / `utils/persona_traits.py`: built-in and dynamic
  trait axes.
- `utils/persona_generate.py`: `~generate` recipe system and saved combos.
- `scripts/`: offline ingestion, classifier/profile/embedding rebuilds,
  fine-tune export, RunPod helpers, smoke tests, audits, and comparisons.
- `index.html` + `assets/`: public GitHub Pages showcase with anonymized visuals.

## Current Commands

Full command details live in [COMMANDS.md](COMMANDS.md).

Translation:
`~autotl`, `~setlang`, `~tloutput`, `~chan_autotl`, `~global_autotl`,
`~practice`, `~romanize`, `~speak`

Archive:
`~said`, `~saidnext`, `~regex`, `~userregex`, `~quote`, `~firstseen`,
`~chatstats`, `~regulars`, `~askchat`

Analysis:
`~whosaid`, `~markers`, `~like`, `~twin`, `~traits`, `~top`, `~vibes`,
`~iq`, `~distinct`, `~why`, `~irony`

Personas:
`~markov`, `~mimic`, `~persona`, `~hyper`, `~generate`

Moderation/utility:
`~help`, `~ping`, `~banuser`, `~unbanuser`, `~warnings`

## Recent Work (evening additions, same day)

- Embedder swapped to **bge-m3** (multilingual — Chinese/German now carry
  meaning, not noise); ALL artifacts rebuilt on it with alias merges.
- **Identity is Twitch-id-dominant**: `author_ids` table (resolver script +
  the live bot records every chatter's id on sight). 79 current authors have
  hard ids; 30 dead old-names remain oracle territory.
- **Anti-spam cooldowns** (escalating, punishes stacking-while-pending only,
  per-user, offenses reviewable via `~warnings`) + `~banuser`/`~unbanuser`.
- **~iq** (component text-register estimate, measured r=+0.33 vs the
  professor axis, i.e. not a clone), **~distinct**, **~why** (per-message
  receipts, per-sentence z, and `words` occlusion attribution).
- **Reaction tracker**: chat's laughter after bot lines is logged as
  funniness labels (persona log, type=reaction_feedback).
- Built-in axes Gram-Schmidt **orthogonalized**; dynamic-axis generation
  hardened (tolerant JSON, backoff, validated poles via the abliterated
  model); organic merge threshold re-measured for bge (0.72).
- **Oracle queues delivered and consumed** via the ai-prompt-engineering review
  tool. Alias, emote-suspect, and irony queues are answered; the empty rename
  queue was removed.
- The three codex/* bucket branches were verified merged and deleted
  (worktrees removed); single main worktree remains.
- **Emote semantics pipeline** (largely built): a ground-truth registry
  (scripts/build_emote_registry.py -> emote_registry.json: ~3k emotes across
  channels, 507 ALIASED, 1713 TAGGED, image URLs stored) and usage-context
  meaning vectors (scripts/build_emote_semantics.py -> emote_semantics.pkl:
  meaning = mean embedding of an emote's message contexts, which uniquely
  covers dead/old-log emotes and fake ones). Five-source meaning architecture
  documented in CHAT_PERSONALITY_RESEARCH.md.
- **Emote-suspect oracle queue**: answered and normalized. It exposed that
  `username` needs to be a first-class token class, not shoved into free-text
  notes.
- Known systematic blind spot (research-doc case study): emotes are stance
  OPERATORS (DansGame inverts a proposition) and the pipeline strips them
  pre-embedding — the concrete motivation for a domain-adapted embedder
  trained on context windows, which starts once irony labels exist.
- **Irony oracle v1 completed**: the 60 labeled items were projected into a
  private multi-axis dataset at `data/unsynced/oracle/irony_v1_multi_axis.jsonl`.
  The key correction is that hyperbole is not the same thing as irony:
  hyperbole usually preserves intended direction while distorting magnitude.
- **Intent-probe v0 trained**: `scripts/train_intent_probes.py` learns separate
  lightweight heads from the private multi-axis irony labels and writes the
  ignored model/report to `data/unsynced/intent_probes.pkl` and
  `_private/INTENT_PROBES_REPORT.md`. First run used bge-m3 embeddings. The
  useful signals are currently hyperbole/magnitude, play frame, and hostility;
  masking and shock need more positive oracle labels before they are useful.
- **Intent-axis queue v2 built/cleaned**: `scripts/build_intent_axis_queue.py`
  uses the v0 probes only as a sampler and writes one focused axis question per
  review item. It now includes author/channel on each card and auto-filters
  obvious bot/mod/art/link junk instead of asking the human to label it. The
  current private queue is
  `../ai-prompt-engineering/private_docs/review_queues/nolifechatter_intent_axes_v2.jsonl`
  with 127 items after removing 13 obvious invalid pending rows. The completed
  old NoLifeChatter queues were archived out of the active review folder.

## Recent Work

- RAG persona generation is live: persona prompts combine signature examples,
  topic-relevant retrieved lines, recent chat context, and optional semantic
  retrieval when the message index exists.
- Persona retrieval guardrails were tightened: direct `~persona` prompts now
  outweigh incidental recent-context terms, semantic message-index hits require
  confidence floors before entering evidence, repeated-token spam is filtered
  out of prompt evidence, and `scripts/persona_rag_preview.py` now prints the
  same snippet/flat evidence sections used by the live prompt.
- Historical context windows are now source-aware. Normal Chatterino/live
  channel logs can provide timestamp-local surrounding chat, but author-only
  Zonian/IVR mirror neighborhoods return only the hit unless another speaker is
  present nearby. Prompt context also de-dupes alias-collapsed repeated lines.
- Archive search ergonomics: `~said` now parses `chat=<channel>`, ignores bot
  command lines by default, stores a short-lived `~saidnext` continuation, and
  `~userregex` finds archived usernames by regex.
- Copy/echo checks were added so generated lines are less likely to be straight
  archived quotes or another chatter's recent line.
- `~generate` exists for tag recipes: users, traits, topic text, channel/year
  scopes, Markov/LLM engines, and saved combinations.
- Dynamic trait axes and burst leaderboards exist; they are useful but still
  register-based rather than a ground-truth psychology instrument.
- The LoRA v2 WAS A/B'd with the normal RAG prompt against plain llama:
  plain llama won on reactivity (~26% vs ~16% reads-as-them; full lines in
  `_private/model_ab_side_by_side.md`). The LoRA stays opt-in via
  `~persona ... model=lora`; a v3 needs conversation-context training pairs.
- Confirmed alt-account decisions were applied to private `config.toml`. The
  detailed accepted/rejected list is private in `_private/ALT_CANDIDATES.md`.
- The public Pages site is live at
  `https://lacrimaeaware.github.io/no-life-chatter/`.
- The rebuild pipeline now includes the per-message semantic index, so a single
  rebuild actually refreshes the artifacts that alias merges affect.
- The next semantic rebuild will use merged same-author utterances by default
  for person embeddings and the message index (`--semantic-unit utterance`).
  This changes semantic retrieval units without changing line-level classifier
  training.
- Future classifier/style-profile rebuilds still train on individual messages,
  but now use the shared message-quality filter instead of a minimal
  "two words and not a prefix command" gate.
- Generated artifact freshness is now visible through `~artifacts` and
  `scripts/artifact_status.py`. Current status: `~iq` has v2 metadata, while
  the person semantic vectors and semantic message index are recent but missing
  semantic-unit metadata because they predate the utterance-unit artifact
  change.
- Future classifier/style-profile rebuilds now write build metadata into the
  shared classifier pickle, so stale profile state is easier to identify.
- The next-work ranking was refreshed in
  `docs/NEXT_WORK_RANKING_2026-06-13.md`. The top implementation is now a
  held-out reply eval harness, because it measures whether persona/RAG changes
  predict real hidden replies rather than only sounding classifier-like.
- `scripts/eval_heldout_replies.py` now samples private held-out reply cases
  from valid chronological channel logs, can run the normal persona generator
  against historical context, excludes the hidden target line from prompt
  evidence, and writes private JSONL/Markdown results under `data/unsynced/`.
- `scripts/build_fact_bank.py` and `scripts/query_fact_bank.py` now provide the
  first evidence-only memory-bank prototype. It extracts candidate user claims,
  preferences, beliefs, and habits into ignored JSONL with timestamps/channels
  and original evidence; rows are claims to review, not verified facts.
- `~askchat` and `scripts/ask_chat.py` now provide the first archive-QA/lore
  surface over fact-bank claims, broad archive hits, near matches, and emote
  meaning. It returns evidence snippets rather than unsupported conclusions.

## First Thing To Do When Returning

Check `~artifacts` or `python scripts/artifact_status.py` before trusting
persona artifact rankings. The semantic vectors and message index still need
the next long rebuild to become tagged utterance-unit artifacts; use
`10-rebuild-persona-artifacts.bat`,
`10-rebuild-persona-artifacts-background.bat`, or
`python scripts/rebuild_persona_artifacts.py --semantic-unit utterance --continue-on-error`
when that runtime cost is acceptable. The active follow-up is still the
oracle-label pipeline: the first intent probes and v2 review queue exist. The
next human step is labeling `nolifechatter_intent_axes_v2.jsonl`, then
retraining the probes and checking whether masking/shock and literal alignment
improve.

## Next Work

High priority:

1. Run the held-out reply eval harness on a frozen case set. Start with
   `scripts/eval_heldout_replies.py --sample-only`, then run `--generate` with
   LM Studio up and compare future retrieval/model changes against that report.
2. Run the next artifact rebuild when a long rebuild is acceptable. It should
   pick up alias changes and the new utterance-based semantic units.
3. Inspect and harden archive-QA/lore results:
   `python scripts/ask_chat.py --author <name> <terms>`, then improve ranking,
   contradiction handling, and answer synthesis.
4. Build fact-bank v2: review queues, contradiction grouping, confidence
   calibration, and decay/recency weighting.
5. Label the private `nolifechatter_intent_axes_v2` review queue, retrain with
   `12-train-intent-probes.bat`, then compare the new report.
6. Run focused smoke tests for persona RAG after the utterance-based semantic
   rebuild.
7. Decide whether the fine-tune path deserves a v3 dataset/model run, or whether
   RAG + better retrieval is the better short-term win.
8. Implement resident persona controls from `GENERATE_AND_BOT_MODES.md`:
   `~botpersona`, `~botmode`, `~botcontext`, and `~botchance`.
9. Implement the Turing-test game: real archived line versus generated persona
   line, chat guesses, then reveal.

Medium priority:

- Build the anonymized Pages similarity map from real embeddings.
- Add an archive-QA command for local lore/emote questions. This should use
  retrieval, not fine-tuning.
- Add source-coverage diagnostics for archive imports: per-channel/day coverage,
  how much evidence comes from full channel logs versus one-speaker mirror logs,
  and where context reconstruction is safe.
- Improve `~said` / `~regex` ergonomics for close matches and saved examples.
- Cleanly separate runtime bot code from offline research/training scripts.

Research / hard mode:

- v3 LoRA design with conversation-context training pairs.
- (orthogonalization and the multilingual swap are DONE as of 2026-06-12;
  remaining axis research = data-driven axes via PCA + per-axis validation.)
- A labeled alt-detection scoring harness using the confirmed aliases.
- Persona-quality evaluation that tracks funniness and in-character behavior,
  not only classifier similarity.

## Known Caveats

- Local LLM calls can be slow on the user's GPU, especially with large persona
  prompts. LM Studio prompt cache helps repeats.
- Fine-tuning is not memory. RAG retrieves memories; LoRA teaches style priors.
- Dynamic trait axes are exploratory. They measure register and evidence
  patterns, not clinical personality truth.
- Short or emote-heavy chatters are hard for semantic averages; burst scoring
  and emote-name semantics help but are not magic.
- Archive imports can make the SQLite database busy. Live capture buffers
  during temporary lock pressure, but long-running imports should still be
  monitored.
- Public docs and Pages must stay anonymized. Do not publish raw logs, real
  rosters, private smoke tests, tokens, config, or exact private aliases.

## Privacy / Public Boundary

Before committing, check staged added lines for private material. At minimum:

```powershell
git diff --cached --name-only
git diff --cached --unified=0
```

Never stage:

- `.env`
- `config.toml`
- `_private/`
- `data/`
- `*.db`
- token files
- raw logs
- private model outputs
- private blocklists

The public project page should stay product/showcase oriented and anonymized.
It should not narrate private operating rules or expose real chat evidence.

## Useful Entry Points

- Current-state map: `docs/STATE_OF_OPERATION.md`
- Next-work ranking: `docs/NEXT_WORK_RANKING_2026-06-13.md`
- Command bible: `docs/COMMANDS.md`
- Persona/archive narrative: `docs/HANDOFF.md`
- Fine-tuning runbook: `docs/FINE_TUNING.md`
- Persona roadmap: `docs/PERSONA_BOT_ROADMAP.md`
- Chat archive design: `docs/CHAT_ARCHIVE.md`
- Personality/research notes: `docs/CHAT_PERSONALITY_RESEARCH.md`
- Idea bank: `docs/IDEA_BANK.md`
- Repo reorg plan: `docs/REORG_PLAN.md`
- Private task buckets: `_private/WORK_BUCKETS.md`
