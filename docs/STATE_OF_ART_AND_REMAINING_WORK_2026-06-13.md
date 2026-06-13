# State Of Art And Remaining Work - 2026-06-13

This is the compact map I should have written before scattering effort across
several useful but disconnected pieces.

## External State Of Art

The relevant modern pattern is not "fine-tune until it remembers." For a live
chat archive/persona bot, the best practice stack is:

1. **Evaluate retrieval and generation separately.** RAGAS frames RAG quality as
   retrieval focus, faithfulness to context, and answer quality, not just "did
   the model sound good." Source: <https://arxiv.org/abs/2309.15217>.
2. **Use better units than arbitrary lines.** RAPTOR-style systems show why
   long corpora need hierarchy/summaries in addition to flat chunks. For Twitch,
   the equivalent first step is merged utterances and source-aware context.
   Source: <https://arxiv.org/abs/2401.18059>.
3. **Add structured memory for private corpora.** GraphRAG's core lesson is
   entity/relationship/community structure on top of chunks for broad questions.
   For this project, that means claims, lore terms, emotes, relationships, and
   citations. Source: <https://arxiv.org/abs/2404.16130>.
4. **Agents need memory retrieval and reflection, not only style prompts.**
   Generative Agents is the useful model here: store observations, retrieve by
   relevance/recency/importance, then reflect/summarize. Source:
   <https://arxiv.org/abs/2304.03442>.

## Current Project State

Strong/live:

- Local SQLite + FTS archive, live capture, Chatterino imports, source-aware
  context windows, alias normalization, and exact/regex archive commands.
- One-shot personas (`~persona`, `~hyper`, `~generate`, Markov) with RAG
  evidence, copy checks, output filtering, model A/B hooks, and reaction logs.
- Stylometry: classifier, lexical markers, `~like`, `~twin`, `~whosaid`.
- Semantic layer: person vectors, message index, trait axes, `~why`, emote
  meaning vectors, IQ v2 cache.
- Eval pieces: held-out reply case sampler/generator/scorer and reaction
  feedback logs.
- Artifact hygiene: `~artifacts`, artifact status CLI, metadata for future
  classifier/style builds, background rebuild launcher.
- First memory layer: evidence-only fact-bank builder/query scripts.

Known weak points:

- Current semantic vectors/message index were rebuilt recently but predate the
  utterance-unit artifact change, so they are `missing-unit` until the next
  long rebuild.
- The fact bank is candidate extraction, not truth. It needs review/ranking,
  contradiction handling, and a QA surface.
- Persona quality still lacks a live reranker trained/evaluated against held-out
  replies and reaction feedback.
- Resident bot modes are still planned, not implemented.
- Intent/irony probes are under-labeled.
- IQ has better sampling than the early attempt, but still needs receipts and
  split-stability explanations to feel trustworthy.

## Remaining Work Ranking

1. **Archive-QA / lore command.** Highest immediate product value. It turns the
   archive, fact bank, emote meaning, and exact search into a single evidence
   surface. This is the next implementation target.
2. **Run full utterance artifact rebuild.** Necessary to make `~distinct`,
   `~vibes`, `~why`, semantic retrieval, and burst axes all come from the new
   chunking unit. Use the background launcher when a long run is acceptable.
3. **Held-out eval run, not just harness.** Freeze cases, generate with current
   persona pipeline, and save a baseline report before more RAG changes.
4. **Persona output reranker.** Score candidate replies for contextual fit,
   target voice, copy risk, and likely chat reaction before posting.
5. **Fact-bank v2.** Add contradiction grouping, review queues, confidence
   calibration, and decay/recency weighting.
6. **IQ receipts.** Add `~iqwhy`/report with component z-scores, confidence,
   split deltas, and driver utterances.
7. **Resident persona controls.** `~botpersona`, `~botmode`, `~botcontext`,
   `~botchance`, with per-channel state and timed revert.
8. **Intent/irony label loop.** More human labels, retrain probes, then decide
   whether they influence live generation/retrieval.
9. **Turing-test game.** Useful human feedback and fun, but lower priority than
   evaluator/reranker plumbing.
10. **Domain-adapted embeddings.** Potentially huge for emotes/slang/lore, but
    should wait for stable eval targets and labeled pairs.
11. **LoRA v3.** Only after context-pair export and held-out eval. Fine-tuning
    is style prior, not memory.
12. **Public anonymized similarity map/site polish.** Nice showcase work, lower
    live-bot impact.

## Decision

Implement **Archive-QA / lore command** first, because it uses the pieces that
already exist and directly answers the repeated project goal: ask the archive
organic questions with evidence instead of guessing from embeddings or a LoRA.
