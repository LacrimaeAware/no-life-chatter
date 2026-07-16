# Roadmap

Last refreshed: 2026-07-16.

## Verdict

The project already has the hard base: a large local archive, alias-aware
search/context, persona RAG, stylometry, semantic indexes, trait tooling,
emote semantics, resident personas, a shared live model queue, and first-pass
memory. The highest-value work is no longer another free-form axis or a larger
fine-tune. It is reliable evidence selection, measurable output quality, and
memory coverage.

## Recently Shipped

The July 15-16 audit landed systemic changes rather than per-user exceptions:

- Canonical identity provenance, display-recency separation, and stale-artifact
  detection across classifier, semantic, IQ, fact, and profile artifacts.
- One deterministic 3,000-utterance semantic index per person. Utterance v3 is
  channel-bounded and collapses duplicate imported components while preserving
  source message IDs.
- Current 40-person classifier, person vectors, message indexes, and IQ v5. IQ
  has complete embedding/judge coverage, fixed evidence receipts, atomic output,
  and fail-closed dependency checks.
- Fact v4 rebuilt over up to 20,000 rows/person. Extraction stops at discourse
  boundaries, rejects clause-shaped possession values, and unscoped QA cannot
  attach one person's self-claim to a different named subject.
- Verified profile v5 candidate precision and slot-aware value normalization.
  A measured one-user run judged 55 candidates in 4m14s; the full roster is an
  explicit dead-hours job, not an accidental interactive task.
- Persona candidate reranking now uses target-authorship probability,
  distinctive markers, nonlinear contextual fit, target length, and copy
  margin. On 113 recent private two-candidate events, score ties fell to zero
  and mean target probability rose from 0.173 to 0.210.
- Versioned, atomic emote semantics with diversified contexts. Three failure
  cases were refreshed; two reached 160 contexts and one honestly stopped at 92.
- Shared live model queue, cold/warm axis routing, category help, command bible,
  archive QA context, irony evidence hierarchy, and comedy cache/credit fixes.

Earlier geometry work remains active: ABTT-2 person-vector correction, Lowdin
trait-axis orthogonalization, BM25+dense RRF, behavioral `~style`, and
distributional contradiction flags. See [RESEARCH_TO_APPLIED.md](RESEARCH_TO_APPLIED.md)
and [GROUND_TRUTH.md](GROUND_TRUTH.md).

## Ranked Next Work

### 1. Freeze And Review The Persona Benchmark

Use the existing held-out-reply sampler across direct replies, topic
continuations, emote/slang reactions, and quiet-chat prompts. Add human review
labels for target voice, contextual fit, genericness, copying, and whether the
line lands. Automatic classifier and lexical similarity are diagnostics, not a
substitute for this review.

### 2. Validate And Extend The Persona Reranker

Compare first-candidate, old keyword selection, and the new ensemble on the
same frozen cases. Tune weights only against held-out labels. Then add reaction
outcomes as a weak retrospective feature and consider three candidates only if
the measured gain justifies the extra queue time.

### 3. Build Verified Profile V5 In Dead Hours

The measured estimate for 40 dense users is roughly 2h50m. Run it with the bot
paused, inspect confirmed/disputed values, and publish only after complete model
coverage. Before scheduling it regularly, consider mixed-slot batches to reduce
the current one-call-per-nonempty-slot overhead.

### 4. Broaden Emote Context Coverage

Top up common emotes from 30 toward 160 diversified contexts in bounded batches.
Track usable context count separately from raw hits and never pad with duplicate
lines. Evaluate meaning summaries on niche emotes where the model has no useful
global prior.

### 5. Improve Archive QA Beyond Fixed Slots

Add evaluated query expansion or a small retrieval planner for questions whose
wording does not overlap likely answers. Then add all-author dense retrieval and
evidence-level stance aggregation. Subject attribution and independent context
remain hard requirements before any mention becomes a belief.

### 6. Correct And Benchmark IQ Geometry

The complete IQ v5 build is auditable, but the review found that semantic
breadth/depth inherited a lexical prefilter and that current depth is closer to
embedding unusualness than sustained topic depth. Separate the full semantic
coverage lane, strengthen quotation detection, and validate any recurring-topic
depth metric before rebuilding. Do not tune rankings person by person.

### 7. Cross-Process Model Scheduling

The live bot has one shared queue, but offline scripts can still compete with
it. Add an inter-process lease or one maintenance worker so commands can report
maintenance state and model jobs cannot stampede LM Studio.

### 8. Train Intent And Resident Volition From Reviewed Labels

Use literal plausibility, community repetition, known-fact conflict, and
conversation response for supervised irony/intent work. Separately train a
cheap `reply / stay silent` resident gate from real outcomes while retaining
channel scope, direct-message boost, idle streak cap, and bot suppression.

### 9. Training And Product Surfaces

Only after benchmark and reranker gains plateau:

- export conversation-context pairs for LoRA v3;
- test domain-adapted embeddings against retrieval and emote prediction;
- consider distributional person models beyond mean pooling;
- expose selected diagnostics on an authenticated private website route.

An unlinked route is obscurity, not access control. Private data requires server
authorization, not a secret-looking URL.

## Operating Checklist

```powershell
python scripts/freshness_check.py
python -m unittest discover -s tests -v
```

After aliases, filters, semantic units, or embedding models change:

```powershell
python scripts/rebuild_persona_artifacts.py --semantic-unit utterance --continue-on-error
```

The pipeline includes the 20,000-row/person claim bank. Add
`--profile-roster 40` only in a model-idle dead-hours window. For unattended
work, use `scripts/start_rebuild_background.ps1`, inspect both logs, and restart
the bot worker only after the artifact reports complete coverage.
