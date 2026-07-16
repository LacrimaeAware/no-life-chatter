# State Of Operation

Last refreshed: 2026-07-16.

Read this first after a break. [ROADMAP.md](ROADMAP.md) is the ranked work
queue, [COMMANDS.md](COMMANDS.md) is the audited command bible, and dated
investigations live under [archive/](archive/).

## Short Version

NoLifeChatter is a Windows-hosted Python Twitch bot with four connected parts:

1. Translation and language-practice commands.
2. A local SQLite/FTS5 chat archive with alias-aware search and context.
3. Persona generation, stylometry, embeddings, axes, and experimental scores.
4. Evidence-backed lore/profile memory and autonomous resident personas.

The public repo is sanitized. Real config, aliases, chat logs, databases,
generated artifacts, model output, and review queues stay ignored in
`config.toml`, `data/`, and `_private/`.

## Runtime

- `run-background.vbs` starts `_bot-loop.bat`, which runs `chatbot.py` and
  restarts it after an exit. Runtime logs go under `data/`.
- `chatbot.py` holds a single-instance socket lock. To reload code/config, stop
  only the worker and let the loop respawn it.
- GPU-heavy live commands, resident replies, and ambient persona generation
  share `services/model_queue.py`: one running job, visible queue positions,
  one active job per user, duplicate-request suppression, and super-admin
  status/clear controls.
- The queue is process-wide. Offline maintenance scripts are not yet coordinated
  by the live queue, so long model builds should run with the bot paused or via
  a future cross-process worker.
- `~botpersona`, `~botmode`, `~botcontext`, and `~botchance` are live,
  channel-scoped, time-limited resident-persona controls. Direct messages,
  greetings, topic affinity, idle speech, reply threading, cooldowns, and a
  no-response streak cap are implemented.

## Identity And Archive

- User aliases normalize transitively through one canonical map. Search,
  context, rosters, profiles, semantic artifacts, and display names use it.
- Live display recency is separate from offline Twitch-ID lookup time. An
  offline resolver can no longer make an old alt become the displayed name.
- Imported rows retain `raw_author` while their searchable `author` is
  canonical. This preserves provenance without duplicating people at read time.
- Generated artifacts carry an alias-map signature. `~artifacts` and
  `scripts/artifact_status.py` warn about split identities, missing provenance,
  or a changed alias map instead of silently serving stale rankings.
- Chronological context windows dedupe alias-mirrored lines and refuse to
  invent conversation around author-only source logs.

## Data Shape

Different tasks intentionally use different evidence budgets:

| Surface | Current input policy |
| --- | --- |
| Authorship classifier | Up to 4,000 filtered, normalized-exact-deduped messages per person plus held-out evaluation |
| Semantic message index | 3,000 deterministic, channel-bounded merged utterances per person: 80% coverage, 20% high-information |
| Person vector | Mean of the unbiased coverage lane from that shared index |
| Text-IQ lexical features | Deterministic filtered history, capped at 15,000 utterances per person |
| Text-IQ semantic features | Up to 3,000 eligible indexed utterances per person |
| Regex claim bank | Up to 20,000 message-local rows per person; weak rows remain candidates |
| Verified profiles | Targeted deep-history retrieval by fact slot, followed by contextual model judgment |

The selector removes commands, repeated spam, unusable fragments, and exact
duplicates. It does not simply take one random block. Coverage selection is
stable across rebuilds; the high-information lane improves retrieval without
warping person-level averages.

## Current Quality Systems

- Persona RAG uses keyword and semantic evidence, merged utterances, recent
  conversation, source-aware context, copy checks, and output validation.
- Two valid persona candidates are reranked without another model call using
  target-authorship probability, distinctive markers, nonlinear prompt fit,
  target length distribution, and copy margin. Private-history replay removed
  first-candidate score ties and improved both target probability and prompt fit.
- `~askchat` fuses BM25 and dense author retrieval, keeps strong paraphrases,
  adds safe chronological context to receipts, and lets the local model
  synthesize only from labeled evidence. `raw`/`noai` remains receipts-only.
- The old regex fact bank is evidence storage, not truth. Exact repeats count
  once; confirmation requires independent days and fresh phrasings, while
  contradiction or cross-user echo blocks promotion. Its sidecar records the
  build budget and alias signature; stale rows fail closed.
- Verified profile memory uses contextual model judgment, archive-grounded
  copypasta rejection, plausibility labels, multi-day corroboration, and
  disputes. Profile v5 gates vague/non-self candidates before model work and
  normalizes accepted outputs through slot-specific schemas. Candidate judgment
  is batched, cached, and individually retried when a batch response omits an
  item. Slot-level atomic checkpoints make long builds resumable after interruption.
- `~irony` combines surface wording with community repetition, unusual literal
  claims, and confirmed-profile agreement/conflict. It remains experimental;
  zero-shot embeddings alone do not reliably recover intent.
- Text-IQ uses the median of each person's top 10% rather than ordinary-message
  averages. Reasoning combines semantic moves with direct clause/reasoning and
  question structure, rather than relying only on embeddings. It rejects long
  exact cross-user copypasta and stores auditable per-dimension receipts for
  `~iq why <user> [dimension]`.
- `~funny` uses before/after new-laugh deltas, excludes self/bot/noise reactions,
  collapses rapid same-author bursts, dedupes laughers across chats, and
  invalidates its cache as the archive grows.

## Maintenance

After aliases, filters, semantic units, or embedding models change, run:

```powershell
python scripts/rebuild_persona_artifacts.py --semantic-unit utterance --continue-on-error
```

The rebuild creates the 3,000-utterance message index first, then derives person
vectors and IQ semantic features from it so the same messages are not embedded
three times. Restart the worker afterward so process caches reload.

The same pipeline rebuilds the 20,000-row/person claim receipt bank. Verified
profile v5 is opt-in because it uses the chat model: add `--profile-roster 40`
during a model-idle maintenance window. A July 16 one-user benchmark judged 55
candidates in 4m14s, implying roughly 2h50m for 40 similarly dense users.

Utterance artifacts carry a chunking-version field. Version 3 groups each
person's short bursts independently per channel and collapses duplicate source
components while retaining their message IDs. Simultaneous posts in two
channels can no longer become one synthetic sentence.

Live `~iq` is cache-only. Missing or provenance-stale IQ data reports that a
maintenance rebuild is needed; it never starts the expensive offline builder
from a chat command.

Before a commit:

```powershell
python scripts/freshness_check.py
python -m unittest discover -s tests -v
git diff --cached --name-only
git diff --cached --unified=0
```

Never stage `.env`, `config.toml`, `_private/`, `data/`, databases, raw logs,
tokens, private aliases, or user-identifying generated reports.

## Known Limits

- A full profile v5 roster still needs a scheduled/dead-hours build. The live
  stale profile shell fails closed until that roughly three-hour pass completes.
- Emote semantics are versioned and crash-safe, but most long-tail emotes still
  have only 30 contexts. Targeted examples now use up to 160 without duplicate
  padding; the broader top-up remains an offline maintenance job.
- Message embeddings mostly measure topic/register. Reasoning and intent axes
  are weak signals and must stay receipt-auditable.
- Exact cross-user copypasta is filtered from IQ, but near-copy and unique
  pasted prose still need a stronger quotation/novelty detector.
- Archive QA can summarize evidence, but it cannot infer a stable belief from a
  few mentions. Opinion and culture claims need repeated, contextual evidence.
- Resident personas still use probabilistic heuristics rather than a trained
  response-volition model.
- IQ v5, fact v4, classifier/style, person vectors, and message indexes are
  rebuilt with the current identity signature. IQ depth remains a weak
  specificity proxy and should not gain weight without a reasoning benchmark.

## Next Work

The authoritative ranking is in [ROADMAP.md](ROADMAP.md). In brief: freeze and
review the held-out persona benchmark, validate the new reranker, schedule the
profile v5 and emote top-up builds, then improve archive QA and supervised
intent work. More LoRA training comes after those evaluation and data-shape
steps, not before them.
