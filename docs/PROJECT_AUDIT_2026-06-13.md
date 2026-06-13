# Project Audit - 2026-06-13

This audit separates what is live, what is partially built, and what is still a
spec. It is intentionally public-safe: architecture and commands only, no raw
logs, private aliases, private queues, or generated evidence.

## Executive Summary

NoLifeChatter is in a useful but very "research workbench" state. The strongest
live systems are the local archive, command auto-discovery, archive search,
persona RAG, stylometry, embeddings, and offline rebuild pipeline. The biggest
quality risks are not the individual trait/IQ axes; they are data hygiene,
identity merging, stale generated artifacts, retrieval shape, and missing
operator controls for autonomous persona behavior.

The next best implementation target became the evidence-backed archive-QA/lore
surface after the held-out reply eval harness and first fact-bank prototype
landed. Resident-persona controls are still important, but they should not
amplify low-evidence retrieval before the archive/memory layer is inspectable.

## Live Architecture

- `chatbot.py` owns the Twitch bot and background token refresh.
- `handlers.py` archives live messages, routes commands, and delegates regular
  message handling.
- `command_registry.py` auto-loads every `commands/*.py` module with a matching
  `handle_<command>` function.
- `command_processor.py` parses the prefix command, checks command bans and
  cooldowns, dispatches, and catches exceptions.
- `services/message_service.py` handles translation, practice mode, and ambient
  persona reaction plumbing.
- `services/llm.py` serializes local/OpenAI-compatible chat calls.
- `utils/chat_archive.py` is the shared SQLite/FTS5 archive and retrieval
  foundation.
- Persona/stylometry systems live mostly in `utils/persona_*.py`.
- Offline pipelines live in `scripts/`.
- Public project page is `index.html`; generated/private data stays ignored in
  `data/`, `_private/`, and `config.toml`.

## Live Commands

The live command source of truth is now `docs/COMMANDS.md`.

High-level groups:

- Translation and language: `~autotl`, `~setlang`, `~tloutput`,
  `~chan_autotl`, `~global_autotl`, `~practice`, `~romanize`, `~speak`.
- Archive: `~said`, `~saidnext`, `~regex`, `~userregex`, `~quote`,
  `~firstseen`, `~chatstats`, `~regulars`.
- Persona/generation: `~markov`, `~mimic`, `~persona`, `~hyper`, `~generate`.
- Analysis: `~whosaid`, `~markers`, `~like`, `~vibes`, `~twin`, `~traits`,
  `~top`, `~distinct`, `~why`, `~irony`, `~iq`.
- Moderation/utility: `~help`, `~ping`, `~banuser`, `~unbanuser`, `~warnings`.

## Spec Versus Code

Implemented:

- `~generate` tag recipes and per-user saved combos.
- One-shot persona commands: `~persona`, `~hyper`, `~markov`, `~mimic`.
- Ambient persona reaction plumbing through config.
- Command bans and escalating anti-spam cooldowns.
- Archive search/stat commands.
- First archive-QA/lore command: `~askchat`, backed by fact-bank claims, broad
  archive hits, near matches, and emote meaning.
- Authorship classifier, lexical markers, semantic neighbors, trait axes,
  dynamic axes, per-message index, IQ v2 cache, and audit script.
- Source-aware context windows: retrieved snippets are timestamp ordered,
  bounded by time, reject fake surrounding chat from author-only mirror logs,
  and de-dupe alias-collapsed repeated lines.

Partially implemented:

- Autonomous persona reactions exist, but only through global/private config.
  There is no per-channel resident persona state and no timed chat command to
  enter or leave a mode.
- LLM calls are serialized, but there is no explicit queue-depth command/user
  feedback layer.
- Irony/intent probes exist as experimental tooling, but the live `~irony`
  command is still a seed heuristic, not a mature classifier.
- Fine-tune/LoRA tooling exists, but current evidence still favors RAG plus
  better context over another blind LoRA run.

Not implemented:

- `~botmode regular|response|random|silent [minutes]`
- `~botpersona <recipe tags or combo name>`
- `~botcontext <free text>`
- `~botchance <odds>`
- Real-or-AI/Turing-test game.
- Archive lore/emote QA command.
- Public anonymized similarity map.

## Immediate Findings

1. `~irony` had a real runtime bug. It was running in a worker thread while
   `utils/chat_archive.connect()` reused a SQLite connection from another
   thread. The durable fix is thread-local archive connections.
2. `~help` lagged the real command surface. Newer commands were falling into
   "other" instead of being categorized.
3. The README command table had become too small to be the command source of
   truth. A dedicated command bible is the cleaner artifact.
4. Identity merges must be followed by artifact rebuilds. Otherwise classifiers,
   embeddings, message indexes, and IQ caches can continue to expose old names.
5. IQ v2 is less random than it looks, but its embedding layer still samples a
   bounded number of utterances per author. Confidence and split stability help,
   but an evidence view is needed for trust.

## IQ V2 Audit Notes

What IQ v2 currently does:

- Reads canonical roster authors.
- Merges burst messages into utterances.
- Filters bot commands, repeated spam, pasted model output, links-only junk, and
  other low-signal lines through `utils/message_quality.py`.
- Uses up to `author_cap` filtered utterances for lexical/syntax features.
- Uses a stable, seeded sample of up to `max_utterances` utterances for the
  embedding features.
- Scores the median of the top tail rather than the average line.
- Combines groups: reasoning, abstraction, vocabulary, syntax, breadth, depth.
- Adds confidence from utterance count, split-half stability, and embedding use.

Important correction: classifier training may use 4,000 messages per author,
but IQ v2 does not simply use 4,000 messages. In the current rebuild settings,
IQ uses up to 15,000 filtered utterances for non-embedding features and up to
600 utterances for embedding features.

Best next improvement:

- Add a public-safe `~iqwhy <user>` or admin-only evidence command that shows
  n, confidence, split delta, component z-scores, and a few sanitized driver
  utterances. Without receipts, the score feels arbitrary even when the math is
  doing something reasonable.

## Recommended Next Implementation Order

Updated ranking after reviewing the state docs, private work buckets, smoke
tests, and current RAG/persona-memory practice lives in
`docs/NEXT_WORK_RANKING_2026-06-13.md`.

1. Held-out reply eval harness. This is now implemented as
   `scripts/eval_heldout_replies.py`: sample real moments, hide the target
   user's actual reply, generate from prior context only, and score generated
   output against the real line.
2. Memory/fact bank prototype with evidence rows, timestamps, confidence, and
   contradiction/decay handling.
3. Archive-QA / lore command with retrieval evidence, not fine-tuning. First
   pass is now implemented as `~askchat` plus `scripts/ask_chat.py`.
4. Persona output reranker using held-out eval and reaction feedback as labels.
5. Resident persona controls: `~botpersona`, `~botmode`, `~botcontext`,
   `~botchance`, with per-channel state and timed auto-revert.
6. Semantic utterance chunking for persona embeddings and the message index.
   This is now implemented for future rebuilds: those scripts default to
   `--unit utterance`, while `--unit message` remains available for A/B.
7. Source-aware context reconstruction for retrieved snippets. This is now
   implemented in `chat_archive.context_window()` and live prompt context:
   author-only mirror logs do not fabricate surrounding conversation, and
   alias-duplicate lines are collapsed before prompt use.
8. LLM queue/cap feedback around `services.llm.chat`, so heavy persona commands
   fail gracefully instead of feeling frozen.
9. IQ receipts: a command or report that explains component drivers per user.
10. Real-or-AI game, because it uses existing archive/persona machinery and gives
   chat a direct eval loop.
11. More intent/irony labels, then retrain probes. The current seed model is not
   enough to make irony a primary live decision layer.

## Follow-Up Implementation Note

After this audit, the semantic rebuild path was changed so person embeddings and
the semantic message index use merged same-author utterances by default. This
does not retrain or rewrite current artifacts by itself. It changes the next
planned rebuild from "embed isolated chat fragments" to "embed conversational
turns," which should improve persona retrieval, `~vibes`, `~why`, trait burst
evidence, and IQ embedding features without tuning any one user's score.

The authorship classifier still trains on individual messages. That is
intentional for now because `~whosaid` often receives a single line, so changing
the classifier's unit should be a separate A/B, not a blind migration. Its
line-level filter now uses the shared message-quality rules, so future
classifier/style-profile rebuilds also drop bot commands, translation
boilerplate, and repeated spam.

Historical snippet context is now source-aware too. Full channel-day logs and
live capture can provide nearby chat by `(channel, sent_at, id)`, but one-speaker
Zonian/IVR mirror neighborhoods return only the target line unless another
speaker is present in the same time window. This prevents author-only imports
from pretending to be real conversation context.

## Operational Rule

After changing aliases, filters, embedding model, or roster thresholds, run the
artifact rebuild pipeline before trusting command outputs:

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_persona_artifacts.py --continue-on-error
```

Then restart the bot worker and let `_bot-loop.bat` relaunch it.
