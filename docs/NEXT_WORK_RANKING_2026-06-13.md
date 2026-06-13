# Next Work Ranking - 2026-06-13

This is the current ranked plan after reviewing the command surface, roadmap,
private work buckets, recent smoke tests, and current RAG/persona-memory
patterns. It is public-safe: no raw logs, private handles, or private results.

## Current Read

NoLifeChatter already has the hard foundation: archive search, alias-aware
personas, RAG evidence, semantic indexes, emote semantics, style classifiers,
and local LLM generation. The weakest layer is now not "one more prompt" or
"one more LoRA run." It is evaluation and memory shape:

- Do we know whether a persona change predicts real replies better?
- Do retrieved examples represent the right conversational moment?
- Can the bot answer evidence-backed questions about recurring user facts,
  lore, and claims?
- Can we separate voice/style from topic, social graph, and language?

Recent RAG/agent work points the same direction:

- RAG needs evaluation across retrieval focus, faithfulness, and answer quality,
  not only vibe checks ([RAGAS](https://arxiv.org/abs/2309.15217)).
- Long-corpus systems benefit from hierarchical summaries instead of only flat
  chunks ([RAPTOR](https://arxiv.org/abs/2401.18059)).
- Private-corpus discovery often wants extracted entities, relationships, and
  community summaries in addition to vector search
  ([GraphRAG](https://arxiv.org/abs/2404.16130)).
- Believable agents use memories plus reflection/retrieval, not only a style
  prompt ([Generative Agents](https://arxiv.org/abs/2304.03442)).

For this project, that translates to: build a real held-out reply benchmark,
then build evidence-backed memory/fact retrieval, then decide whether training
or resident autonomous behavior is worth turning up.

## Ranked Work

1. **Held-out reply eval harness.**
   Highest value because it answers whether persona/RAG changes actually improve
   against real hidden replies. It should sample real moments, hide the target
   user's reply, generate from prior context only, and score against reality.
   This is now implemented as `scripts/eval_heldout_replies.py`.

2. **Memory/fact bank prototype.**
   Extract recurring user facts, claims, lore terms, and habits with evidence
   rows, confidence, timestamps, contradiction handling, and decay. This is the
   bridge to questions like "has X ever said Y?" and "what does X believe about
   Z?" without pretending LoRA is memory.

3. **Archive-QA / lore command.**
   A local evidence-answering route for archive/emote/lore questions. It should
   retrieve exact quotes, semantic neighbors, emote facts, and eventually fact
   bank entries, then answer with citations. Do not solve this by fine-tuning.

4. **Persona output reranker.**
   The generator already samples candidates. The next version should score
   candidates for contextual fit, target voice, non-copying, and "does it land."
   Held-out eval and reaction feedback are the labels for this.

5. **Resident persona controls.**
   `~botpersona`, `~botmode`, `~botcontext`, and `~botchance` are still useful,
   but they should come after the eval harness because autonomous behavior will
   amplify quality problems if the evaluator is weak.

6. **Turing-test / real-or-AI game.**
   Fun and useful as human feedback, but best after held-out eval so the game
   can collect meaningful labels instead of only entertainment.

7. **Intent-axis labeling and retraining.**
   Still important, especially for irony/emote operators, but it needs human
   labels. Engineering should support it, not block on it.

8. **Another LoRA run.**
   Do this only after held-out eval and cleaner context-pair export. The last
   LoRA result showed that more training without better retrieval/eval can
   become bland or unreactive.

## Implemented In This Slice

`scripts/eval_heldout_replies.py` now:

- Samples held-out reply cases from valid chronological channel logs.
- Defaults away from one-speaker mirror logs so the context is real.
- Filters bot/noise authors and bot-command-shaped context rows.
- Writes private JSONL cases under `data/unsynced/`.
- Optionally calls the normal persona generator with `--generate`.
- Feeds historical context through `recent_override` instead of live chat.
- Excludes the hidden target reply from prompt evidence to avoid leakage.
- Scores generated-vs-real with normalized line similarity, topic-term
  precision/recall, length ratio, optional classifier attribution, and optional
  embedding cosine.

The next practical run is:

```powershell
.\.venv\Scripts\python.exe scripts\eval_heldout_replies.py --sample-only --max-cases 80
```

Then, with LM Studio running:

```powershell
.\.venv\Scripts\python.exe scripts\eval_heldout_replies.py --case-in data\unsynced\persona_heldout_eval_cases.jsonl --generate --classifier-score --embed-score
```
