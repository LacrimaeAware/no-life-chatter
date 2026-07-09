# Roadmap

Last refreshed: 2026-06-14.

## Recently Shipped (2026-06-14) — embedding-geometry pass

Driven by findings ported from the `structured-transform-discovery` research
repo; full write-up in [RESEARCH_TO_APPLIED.md](RESEARCH_TO_APPLIED.md).

- `scripts/eval_geometry.py` — the geometry dial (anisotropy, axis collinearity,
  axis-score entanglement, ABTT safety guard). Build the number before the fix.
- Trait axes decorrelated via **Löwdin** orthogonalization, unified across
  `~traits`/`~top`/`~top burst` (entanglement 0.483 -> 0.249).
- **ABTT-2** isotropy correction in `persona_embeddings._centered()` (k chosen
  empirically; the effect is non-monotonic).
- **RRF** bm25+dense fusion in `archive_qa` author QA (paraphrase recall, bm25
  kept as an independent lane).
- **`~style` command** (`utils/behavior_profile.py`) — data-driven behavioral
  personality read (how a chatter types vs the room). The structural half of
  personality; intent traits (irony/hostility) parked as not behaviorally
  measurable. Full trajectory + overturned claims in
  [INVESTIGATION_LOG.md](INVESTIGATION_LOG.md); design in
  [PERSONALITY_SYSTEM_DESIGN.md](PERSONALITY_SYSTEM_DESIGN.md).

## Verdict

NoLifeChatter has the hard base working: local archive, command discovery,
search/stats, persona RAG, Markov/LLM personas, stylometry, semantic artifacts,
emote semantics, first fact-bank extraction, archive-QA via `~askchat`, command
audits, and artifact-status reporting.

The main quality bottleneck is no longer "add one more axis" or "train one more
LoRA." The highest-value work is making retrieval/memory/eval reliable enough
that later autonomy or training does not amplify bad evidence.

The next operational dependency is the full utterance-unit artifact rebuild.
It should make `~distinct`, `~vibes`, `~why`, semantic persona retrieval, burst
trait evidence, and IQ embedding features line up with the newer chunking and
alias/filter rules. Check `~artifacts` or `scripts/artifact_status.py` before
trusting rankings.

## State Of Art For This Project

- RAG quality should be evaluated as retrieval focus, answer faithfulness, and
  final answer quality, not just "sounds good."
- Long chat corpora need better units than arbitrary single lines. For this
  project, merged same-author utterances and source-aware context windows are
  the immediate practical layer.
- Memory should be structured and cited. Fact-bank rows are candidate claims
  with evidence, not verified truth.
- Fine-tuning is a style prior. It is not memory. RAG and archive QA should own
  facts, lore, and old-message evidence.
- Resident/autonomous persona behavior should come after eval and retrieval
  hardening, because it multiplies whatever quality level exists.

## Current Strengths

- SQLite/FTS archive with live capture and imported Chatterino logs.
- Alias-aware search, stats, quotes, context windows, and user normalization.
- One-shot persona commands with archive evidence, copy checks, output filter,
  and local OpenAI-compatible LLM support.
- Stylometry commands: `~whosaid`, `~markers`, `~like`, `~twin`.
- Semantic commands/artifacts: `~vibes`, `~traits`, `~top`, `~distinct`,
  `~why`, `~iq`, message index, and emote semantics.
- First evidence-backed lore surface: `~askchat` plus offline `scripts/ask_chat.py`.
- First memory prototype: `scripts/build_fact_bank.py` and
  `scripts/query_fact_bank.py`.
- Held-out reply eval harness exists, but the baseline run still needs to be
  executed and compared against future changes.
- Command health is repeatable with `scripts/audit_commands.py`.

## Known Weak Points

- Generated artifacts can become stale after alias merges, embedding-model
  swaps, semantic-unit changes, or message-quality filter changes.
- Some archive sources are author-only mirrors. They cannot safely provide
  surrounding conversation unless another speaker is present in the same time
  window.
- Fact-bank claims are unreviewed candidates. They need contradiction grouping,
  confidence calibration, and review tooling.
- Persona generation still lacks a live reranker trained against held-out reply
  cases and reaction feedback.
- Resident persona controls are planned, not live.
- Intent/irony probes are under-labeled and should stay diagnostic until the
  label set is stronger.
- IQ is a roster-relative text-register/cognition estimate. It needs receipts
  (`~iqwhy` or a report) before users can audit why a ranking happened.

## Ranked Next Work

1. Finish or verify the full utterance-unit artifact rebuild, then restart the
   bot worker so live commands see fresh artifacts.
2. Run the held-out reply baseline on frozen cases with the current persona
   pipeline.
3. Improve archive-QA/lore v2: contradiction handling and answer synthesis.
   (Semantic evidence + RRF bm25/dense ranking now landed for the author path;
   profile slot-intent routing landed 2026-07-09; remaining: all-channel dense
   index, contradiction grouping, a rerank step, and **HyDE-style query
   expansion** — for arbitrary questions, have the local LLM write 2-3 phrases
   the answer would have been stated as first-person ("where does X live" →
   "i live in", "im from") and search THOSE; the profile slots' hand-written
   anchors are the manual special case of this.)
4. Build a persona output reranker for contextual fit, target voice, copy risk,
   and likely chat reaction.
5. Fact-bank v2 — the slot-profile core landed (`utils/user_profiles.py`,
   `scripts/build_user_profiles.py`): fixed profile slots
   (location/age/gender/occupation/relationship/pets/hobbies/languages/family),
   anchor-phrase retrieval over the alias group, an in-context
   sincerity/extraction judge on the local model (rejects jokes, quotes,
   copypasta), and multi-day corroboration — a value is only "confirmed" on
   >= 2 independent days; single sightings stay "candidate"; conflicting
   confirmed values become "disputed", never a silent guess. Judged messages
   are cached, so re-runs are incremental (dead-hours-batch friendly).
   Remaining: run the full roster on a schedule, wire `profile_line()` into
   persona generation as a fourth evidence section, and surface confirmed
   facts in `~askchat`.
6. Add IQ receipts: component z-scores, confidence, split deltas, and driver
   utterances. (Rarity contamination fixed in code 2026-07-09 — emote names,
   usernames, and confidently non-English utterances no longer count toward
   rare vocabulary (`persona_iq._rarity_exclusions`, `_non_english`); reasoning
   markers count DISTINCT markers so doubled lines can't fake a chain. A new
   self-supervised dial also exists: `scripts/eval_emote_prediction.py` masks
   the emote a chatter actually used and scores a model on picking it from a
   lineup — unlimited free labels for "does it understand emotes".
   **Next known contaminant, from the 2026-07-09 audit receipts: pasted
   text.** A pasted psych definition drove one chatter's causal-reasoning
   axis and the "new viewer here" copypasta drove another's syntax peak —
   quotes/copypasta score as smart. Fix: filter IQ input utterances with the
   fact-bank's archive-grounded copypasta check (`user_profiles._said_by_others`
   generalized), and note the cognitive-axis projections are tiny (~0.07-0.09
   cosine) — treat the embedding reasoning dims as weak signals.)
7. Implement resident persona controls from
   [GENERATE_AND_BOT_MODES.md](GENERATE_AND_BOT_MODES.md): `~botpersona`,
   `~botmode`, `~botcontext`, and `~botchance`.
8. Continue the intent/irony label loop, then decide whether probes should
   influence retrieval/generation. **Now has a concrete payoff target:** the
   measured charged-axis confound (`scripts/irony_confound.py`,
   `docs/GROUND_TRUTH.md` known limitations). The zero-shot "ironic" axis cannot
   detect deadpan/charged irony (an ironic chatter scores "sincere", yet scores
   mid-high on "menace" off joke lines), and a naive irony-discount does not work
   (irony↔menace corr +0.17, no signal to discount). Path: train a supervised
   irony/intent detector from the oracle queue labels, then use it to make
   charged-axis scoring (`~traits`/`~top` on menace/racism/misogyny-type axes)
   intent-aware so it stops reading bits as sincere belief.
9. Build the real-or-AI game after eval plumbing is stable enough to turn the
   game into useful feedback.
10. Consider domain-adapted embeddings only after stable eval targets exist.
    Geometry frontier (all gated on `scripts/eval_geometry.py` + a held-out
    baseline showing the cheap fixes plateaued, per
    [RESEARCH_TO_APPLIED.md](RESEARCH_TO_APPLIED.md) §5): a curved-orbit vs
    steering-vector replication study on text personas; a distributional person
    model (per-message clouds + 2-Wasserstein, replacing the mean-pool); and a
    reconstruction-pressure latent for genuine voice transfer. **First slice
    SHIPPED (2026-06-14): a self-contradiction reliability flag.**
    `persona_msg_index.contradiction_scores(axis)` measures both-pole occupancy
    from the per-message clouds, and `~traits` now marks a charged-axis lean ⚡
    when the chatter occupies both poles (a performativity proxy that needs no
    irony oracle). Diagnostic: `scripts/contradiction.py`. Remaining: extend the
    flag to `~top`/`~why`, and the full distributional model (2-Wasserstein) +
    the held-out reply benchmark to validate it. See `docs/RESEARCH_TO_APPLIED.md` §6.
11. Consider LoRA v3 only after context-pair export and baseline eval. Do not
    use LoRA as a substitute for memory.
12. Polish public anonymized similarity maps and site visuals after the live
    bot quality loop is healthier.

## Nightly / Return Checklist

```powershell
.\.venv\Scripts\python.exe scripts\freshness_check.py
.\.venv\Scripts\python.exe -m unittest tests.test_pure_functions
```

If aliases, filters, embedding model, or semantic units changed, run:

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_persona_artifacts.py --semantic-unit utterance --continue-on-error
```

For unattended Windows runs, use:

```powershell
.\scripts\start_rebuild_background.ps1 -SemanticUnit utterance
```
