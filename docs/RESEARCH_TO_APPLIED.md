# From research to production: porting latent-geometry findings into a live persona engine

This document traces a concrete line from a research project of mine —
[`structured-transform-discovery`](../../structured-transform-discovery), a
controlled study of how transformation *factors* are represented in frozen
neural encoders — into measurable changes in this bot's persona/embedding
stack. It is written as a portfolio artifact: the point is not just *what*
changed but *how the research framed the diagnosis* and *how each change was
verified before shipping*.

The short version: the research repo studies **steering vectors and curved
factor orbits in image encoders**. This bot scores chatters by projecting their
message embeddings onto **trait axes that are exactly steering vectors**. The
research predicted those axes would be entangled and mushy. They were. The fix
is geometric, and every step is backed by a number from a harness I built first.

---

## 0. Methodology transfer: build the falsifiable dial *before* touching code

The research repo's defining discipline is *falsifiability*: matched-random
nulls, pre-registered predictions, "report dimension-robust measures, not raw
rank," and a willingness to close an open question **in the negative**
(Experiment 22's a-priori-alpha predictors were tested and *failed*). Several of
its headline results are negative — a constant steering vector never transfers
across six factors and three encoders.

So the first thing built here was not a fix but a measurement instrument:
[`scripts/eval_geometry.py`](../scripts/eval_geometry.py). It computes, from
artifacts already on disk:

- **anisotropy** of the person-vector matrix (mean |off-diagonal cosine|, top-k
  singular-value share),
- **axis collinearity** (pairwise cosine of the trait axes),
- **axis-score entanglement** (do a person's trait z-scores correlate across the
  roster — the thing a user actually experiences as "the axes all feel the
  same"),
- an **ABTT safety guard** (how much of each axis survives a proposed isotropy
  correction).

Building the dial first immediately paid off: **it falsified two claims** from
the initial audit before they could become code (see §4). That is the whole
value of the research mindset — a plausible diagnosis is not a verified one.

---

## 1. The steering-vector pathology → Löwdin-decorrelated trait axes

**Research finding.** `structured-transform-discovery` Experiments 23–26: on a
frozen encoder, a factor is typically a **low-dimensional curved orbit**, and a
**constant mean-difference "steering vector" is a poor operator** on it — it does
not transfer across classes (transfer residual 0.49–1.05, never below the 0.25
"steerable" threshold). A corollary the repo notes: distinct factors built this
way share a dominant component — "one axis wearing two names."

**The bot's code, before.** A trait axis was
`v = mean(embed(positive_poles)) − mean(embed(negative_poles))`
([`persona_traits.py`](../utils/persona_traits.py)) — *literally* the steering
vector the research falsified. A person's trait score was their centered vector
projected onto `v`. Five such axes (menace, ironic, unhinged, professor, doomer).

**What the dial showed.** The raw axes share a "negativity" component, so the
five per-person scores were **0.483 mean |off-diagonal| correlated** across the
roster — knowing someone's menace score told you most of their doomer score. The
axes were not five dials; they were ~2.5 dials wearing five labels.

**The fix.** Replace the ad-hoc, order-dependent Gram-Schmidt that one code path
used with **Löwdin symmetric orthogonalization** —
`A_orth = (A Aᵀ)^{-1/2} A` — and route *every* scoring path through it
([`persona_traits.ortho_axis_vectors`](../utils/persona_traits.py)). Löwdin is
the orthogonal matrix **closest** to the original set (minimal Frobenius
rotation), so it shares the de-correlation evenly instead of privileging
whichever axis happens to be first in a list.

**Verified result.**

| axis geometry | score-correlation | doomer alignment with its own label |
|---|---|---|
| raw (steering vectors) | 0.483 | 1.000 |
| Gram-Schmidt (order-dependent) | 0.281 | **0.732** ← distorted (last in list) |
| **Löwdin (shipped)** | **0.302** | **0.917** ← every axis stays itself |

Gram-Schmidt scored marginally lower correlation but **collapsed the last axis
to 73% of its own meaning** purely from list order; Löwdin keeps every axis
0.92–0.99 aligned with its label. This also fixed a real inconsistency: three
code paths (`~traits`, `~top`, `~top burst`) had used *three different* axis
geometries and silently disagreed. They now share one source of truth.

---

## 2. Nuisance-subspace down-weighting → ABTT isotropy correction (and a
   non-monotonic surprise)

**Research finding.** The repo repeatedly finds that **removing or
down-weighting a nuisance subspace** helps held-out, de-confounded accuracy, but
**partial beats full** removal — the optimum is *interior* (Exp 18–19: an alpha
near 0.5–0.75 beats both full removal and no removal; full removal measurably
*hurt*, 0.908 → 0.874). The lesson: subspace surgery is real but must be tuned,
because a nuisance direction also carries some signal.

**Applied.** Add the "All-But-The-Top-k" isotropy correction (Mu & Viswanath,
ICLR 2018) to the centered person space: after mean-centering, project out the
top-k principal components. This is the text analogue of the repo's
nuisance-subspace removal.

**The non-monotonic surprise — exactly the repo's "interior optimum."** I
assumed k=1 (conservative) was safe and beneficial. The dial disagreed:

| k | person-similarity anisotropy | axis-score entanglement |
|---|---|---|
| 0 | 0.165 | 0.302 |
| **1** | 0.157 | **0.335 ← worse!** |
| **2** | **0.149** | **0.249 ← better on both** |

k=1 removes a component that carried *axis-discriminative* signal and **tangled
the trait axes worse**; the second component is a shared nuisance whose removal
improves *both* person similarity and trait decorrelation. **You cannot reason
your way to k=2 from theory** — the harness found it, which is precisely the
repo's point that subspace-removal strength is empirical, not a priori. Shipped
at k=2; every trait axis still retains ≥0.94 of its energy (doomer lowest at
0.943, unhinged 0.995 — the safety guard that confirms the dials aren't blunted).

**Combined effect of §1 + §2:** trait-axis entanglement **0.483 → 0.249**, a 48%
reduction in cross-axis redundancy, with person-similarity anisotropy also down
(0.165 → 0.149).

---

## 3. "Keep the strong baseline visible" + hybrid retrieval → RRF fact retrieval

**Research finding.** Two lessons recur in the repo's reviews: a **plain
discriminative baseline often matches or beats** an elaborate learned metric
(logistic regression was the strongest or near-strongest method across five
datasets in Exp 21), and **keep the strong cheap baseline reportable** so a fancy
method can never silently tank recall. Relatedly, the steering audit found that
where structure transfers at all it is via **late, token-level interaction**, not
a single pooled vector.

**The bot's code, before.** The question-answering path
([`archive_qa.build_report`](../utils/archive_qa.py)) ranked evidence by **bm25
keyword match only**. The dense embedding index that justifies embedding millions
of messages **never touched QA ranking** — so a question whose answer is phrased
differently from the query ("does X like cars?" vs. a message saying "my civic
rips") simply failed.

**The fix.** Fuse the bm25 lane with the dense semantic lane by **Reciprocal
Rank Fusion**: `score(line) = Σ_lanes 1/(k + rank_lane)`. RRF is *rank-only*, so
it needs no score calibration between bm25 and cosine — robust on an un-whitened
embedding space, and it can never rank below the better single lane. bm25 stays
the first, independently-visible lane (the "keep the baseline visible" lesson);
dense only *adds* paraphrase recall. Falls back to pure bm25 when the index or
embedder is unavailable, so it can never break QA.

**Verified result (real archive, paraphrase recall the keyword lane missed):**

| query | bm25 keyword lane | what the dense lane added |
|---|---|---|
| "video games" | repetitive emote spam ("NEXT VIDEO …") | *"i cant even find games now"* |
| "food" | — | *"why doesnt he just pay someone to eat food for him"* |
| "money" | — | *"money is a snowball mechanic"* |
| "girlfriend" | 5 strong keyword hits | nothing (correctly — RRF doesn't degrade a strong lane) |

---

## 4. What the research mindset *corrected* (the honest part)

Building the dial first caught two confident-but-wrong claims from the initial
LLM-assisted audit. Reporting them is the point, not an embarrassment — it is the
difference between a plausible story and a measured one.

- **"Centering leaves every cosine at ~0.99."** False. Mean-centering already
  takes mean |off-diagonal cosine| from **0.983 → 0.165**; the anisotropy that
  was called the deepest problem was *already solved in production*. ABTT is a
  refinement on an already-healthy space, not a rescue.
- **"menace ≈ doomer at +0.91."** The real cosine is **+0.64**. The 0.91 was a
  *stale code comment* from an earlier embedder build, read as if it were a live
  measurement. The genuine collinearity is moderate, and the experienced problem
  was the *score* entanglement (0.483), not the axis-vector cosine.

Both corrections came from the same harness, in minutes, before any code shipped.

---

## 5. Open frontier (next, evidence-gated)

These follow directly from the research and are queued behind the dial — each
ships only when the harness shows the cheap wins have plateaued:

- **Curved-orbit vs. steering-vector replication on text personas.** Port the
  repo's Exp 23–26 go/no-go to text traits: build each axis as both a linear
  shift and an RBF kernel-ridge readout, report per-axis transfer residual
  against the 0.25 threshold. Either outcome is publishable — a clean replication
  (or refutation) of an image-domain result in a new modality.
- **Distributional person model.** The person vector is currently a mean-pool —
  it collapses a bimodal "shitposter + analyst" to a meaningless midpoint. Keep
  the per-message cloud and compare people as Gaussians via the **2-Wasserstein
  (Bures) metric**. The clouds are already on disk; only the consumer is missing.
- **Reconstruction-pressure latent for voice transfer** (Exp 27 port): the
  repo's sharpest counterintuitive result is that a *reconstruction*-trained
  autoencoder made a factor steerable where a *classifier*-supervised one (probe
  R²=1.00) did not. That is the only credible path to genuine vector-arithmetic
  voice transfer; the naive `embed(B) + (μ_A − μ_global)` is the exact construct
  the audit falsified.

---

## 6. Findings worth keeping, and questions this data can answer

Three findings from this pass that generalize beyond the bot:

1. **Mean-centering already buys most of the isotropy; the heavyweight fix
   didn't earn its reputation here.** Centering took anisotropy 0.983 → 0.165;
   ABTT only refined it to 0.149. The lesson is the methodology, not the
   technique: a measurement falsified the assumption that anisotropy was the
   bottleneck before any code shipped.
2. **ABTT is non-monotonic in k.** k=1 was *worse* than both k=0 and k=2 on axis
   entanglement (0.302 → 0.335 → 0.249). Subspace-removal strength is empirical,
   not analytic — a clean small-scale echo of the source repo's "interior
   optimum" result (Exp 19–22).
3. **Order-dependent orthogonalization silently distorts the last basis
   vector.** Gram-Schmidt left "doomer" only 0.73 aligned with its own label
   purely because it was last in a hardcoded list; Löwdin keeps it at 0.92. Any
   system that z-scores projections onto a hand-ordered axis set inherits this
   bug invisibly.

A fourth finding, and the cleanest verify-first save of the project: **the
obvious fix for the "ironic edgelord tops the racism axis" complaint is wrong.**
The intuitive fix is to down-weight a person's ironic messages when scoring a
charged trait axis. Measured (`scripts/irony_confound.py`), it fails twice: (a)
person-level irony↔menace correlation is only +0.17, so discounting barely moves
the ranking; and (b) more fundamentally, the zero-shot "ironic" trait axis
*cannot detect the relevant irony* — a highly-ironic chatter's per-message ironic
projection spans only ≈0.20 and sits near zero, so there is nothing reliable to
discount. The charged axes read surface words; the system has no working intent
signal. The honest conclusion is that this needs a **supervised** irony detector
(the irony oracle review queue), not a discount hack — and that conclusion only
exists because the discount idea was measured before it was built.

One genuinely testable question this dataset is well-posed for (beyond the §5
frontier): **is mean-pooling destroying recoverable signal?** It is checkable
now, with no new data — compute the within-author variance spectrum from the
per-message clouds already on disk. If a meaningful fraction of authors are
bimodal (high within-author spread), the single-centroid person vector
misrepresents them, and a distributional model (2-Wasserstein over the clouds)
should beat centroid cosine on the held-out reply benchmark. That is a clean,
falsifiable hypothesis with a pre-committed metric — exactly the experiment shape
the research repo is built around, and a good portfolio result either way.

**First measured answer (2026-06-14, `scripts/contradiction.py`): yes, and it has
a product use.** A chatter known to be performative/ironic appears at *both*
poles of the menace axis at once (edgy shitpost lines *and* wholesome lines), yet
their mean projection is a bland −0.026 — mean-pooling erases the very structure
that characterizes them. A "contradiction" score (mass above the global 90th
percentile pole × mass below the 10th, geometric mean) recovers it and ranks them
in the top ~18% of the roster. This is the user's own hypothesis — *a deeply
ironic person holds contradictory traits ("high in both feminism and misogyny"),
impossible as one axis value but ordinary in one person's data* — and it needs no
irony oracle, no single-message intent call, and no whole-conversation context.
It is a *performativity* proxy, not a proven irony detector (topic variety can
also produce range), but it is the honest no-oracle read of a charged axis, and
its product form **shipped the same day**: `~traits` now marks a charged-axis
lean ⚡ when the chatter occupies both poles (via
`persona_msg_index.contradiction_scores`), so the bot stops asserting a confident
sincere score for a person who is performing both ends — e.g. a chatter who reads
2.2σ "menace" but also lives at the wholesome pole is flagged unreliable rather
than branded a menace.

## 7. The biggest finding: the embedder is a topic machine, not a personality one

Discovering axes from the data instead of imposing them (`scripts/discover_axes.py`,
unsupervised PCA/ICA over the person vectors) revealed *why* the trait axes never
felt right: **the directions the data actually varies along are topic and
language, not personality.** The top components are gaming-vs-coding talk,
English-vs-Spanish/German, and emote-density — only sentiment (friendly vs
hostile) is personality-ish, and it is weak. BGE-M3 encodes *what* people talk
about; personality is the faint residual after topic. That is the ceiling on the
whole embedding-based persona approach, and it is why "menace" collapsed into
"negativity" — sentiment is one of the few non-topic signals that survives
embedding.

The fix is to measure personality where it actually lives: **behavior, not
vocabulary** (`scripts/behavior_axes.py`). Eleven topic-free counted features
(words/msg, caps rate, emote rate, @mentions, profanity, vocab richness,
message-doubling, …) → PCA → interpretable axes that match human reads:
emote-spam-hype (28% var), crude-vs-clean (21%), talks-at-people-vs-broadcasts
(12%). The behavioral axes are **mostly independent of the topic axes** (mean
\|correlation\| 0.22 across all pairs; they overlap only on emote-density). The
behavior space *knows who people are* — it puts the no-emote/rich-vocab chatter,
the cleanest-mouthed one, and the always-@ing-people one exactly where a human
would — in a way the topic space never did. A real data-quality catch fell out of
this: whole-message doubling ("X X") varies 0–31% by person and inflated every
per-message count until de-doubled. This behavioral space, not more embedding
geometry, is the foundation for the "better axes" the owner asked for.

## 8. Cross-repo pass (2026-06-18): the question-encoding fix and a falsified value model

A second sweep, pulling ideas from two *other* repos — the win-rate model in
`kaggle-fun/pokemon-tcg-ai-battle` and the concept-geometry instrument in
`structured-transform-discovery`. Both repos converge on the same two lessons:
*(i) the quality of any value model is capped by how you encode the state/question,
and (ii) benchmark against a dumb prior / close the loop, because clever geometry
lies in-sample.* Applied here, three results:

- **Asymmetric query encoding — FALSIFIED for our model.** The question-answering
  path (`~askchat`/`archive_qa`) is far cruder than the trait geometry: bag-of-words
  FTS plus a single *symmetric* mean-pool of the raw question. The obvious "quick
  win" was a BGE-instruction query prefix (asymmetric retrieval). Measured on
  bge-m3: it just lowers every cosine and reshuffles noisily — wash-to-worse. Not
  shipped. (bge-m3 does not use the bge-v1.5 instruction convention.)

- **The lexical gate was the real bug — FIXED and validated.** The dense lane
  exists to find *paraphrases* (no shared keyword), but `_matches_focus` hard-dropped
  any hit lacking a literal query-term overlap — measured: **~75% of the top dense
  hits discarded, including the literal best answers** ("he plays apex" for "what
  video games", killed because "playing" ≠ "play"). Replaced with the two-tier
  cosine floor the config already defined but the code never wired in (anchored
  ≥0.50, unanchored paraphrase ≥0.62). Held-out: **dense recall ~doubled (7→13 of
  32)**, recovering the best paraphrase answers.

- **The "did-it-land value model" — FALSIFIED before building it.** The exciting
  idea (owner's framing: "winning players = the real person's messages; emulate
  the moves that landed") is win-rate prediction applied to persona quality. The
  dial: [`scripts/landing_probe.py`](../scripts/landing_probe.py) labels real
  messages by whether they sparked a fresh laugh (9.5% do), then asks whether that
  is predictable from message *content*. Held-out AUC: embedding contrast-axis
  **0.533**, full-embedding logistic regression **0.489**, best dumb baseline
  (has-emote) **0.545** — i.e. **content does not predict landing; no model beats
  the dumb prior.** Whether a line lands is *context and timing*, not the words.
  This is exactly where the Pokémon analogy breaks: a good *move* transfers to
  similar game-states, but a line that *landed* did so for its moment and does not
  transfer to a new context. So a content-based persona value model is not worth
  building. What survives: keep retrieving the person's real lines + LLM (already
  done), and treat the structured-transform geometry as *diagnostic rigor*
  (contrast-pair SVD axes, shrinkage whitening, bootstrap stability) — the place
  both source repos agree these methods actually pay off.

## Showcase note

This is one of the better portfolio stories in the project because it is the full
loop: **independent research → a falsifiable diagnosis → a measured production
change → an honest account of what the measurement overturned.** It shows
(a) reading one's own research as an instrument rather than a trophy, (b)
building the dial before the fix, and (c) the empirical humility to ship k=2 over
the "obvious" k=1 because the data said so. The reproducible numbers all come
from [`scripts/eval_geometry.py`](../scripts/eval_geometry.py); the research
lineage is in [`structured-transform-discovery`](../../structured-transform-discovery).
