# Chat personality research notes

Exploratory plan for using Twitch chat archives to study stable user style,
behavior, and persona dimensions. This is not a clinical or diagnostic system.
The useful version is a private research/art tool first, with public output only
after consent and careful anonymization.

## Research questions (2026-06 — what we actually want to know)

The token-statistics era (log-odds markers, TF-IDF stylometry) is built and
live; these are the questions it raised. Most need embeddings (a local
embedding model is already served next to the chat models) because they're
about MEANING, where token overlap is structurally blind.

1. **Style vs topic vs language.** A bilingual chatter's strongest "marker" is
   their other language; a person's hobby dominates their vocabulary. Can we
   factor a chatter into independent style / topic / language components, so
   "who writes like X" stops conflating "who also speaks German"? (Embedding
   spaces make language a direction that can be projected out; token stats
   can't.)
2. **First-order words vs second-order traits.** Two people never share a
   single catchphrase yet read as the same KIND of person (both contrarian,
   both sincere-poster, both doomer). Define trait axes by embedding example
   sentences for each pole, project chatters onto them — do the axes recover
   what chat intuitively knows? Which traits are even recoverable from text?
3. **Alt detection, properly scored.** Token profiles already surface alts
   with shared-catchphrase evidence. Build a labeled set (known alts from
   config aliases), measure precision/recall of token-similarity vs
   embedding-similarity vs both. Which finds alts that changed their
   vocabulary on the new account?
4. **Temporal drift.** "My markers are polluted by what I spammed 5 years
   ago" — measure how much a person's profile moves per year. Is voice
   stable while topics churn? Do people converge on a community's voice the
   longer they stay (and can we measure who assimilates vs who stays
   distinct)?
5. **Community fingerprints.** The same person scoped to two channels yields
   two visibly different profiles. How much of a person is portable vs
   channel-induced? Does each channel have an "accent" everyone there picks up?
6. **Personality clusters/maps.** Cluster person-vectors; do the clusters
   match the social groups, or cut across them (the interesting case)? A 2D
   map of the community is the flagship deliverable — people love seeing
   where they land.
7. **The judge problem.** The authorship classifier doubles as a persona
   metric ("does the generated line read as them"), but it rewards lexical
   tics, so an unreactive model that spams catchphrases can score well.
   Does an embedding-based judge (semantic similarity to their real replies
   in similar moments) track human funniness judgments better?

## Second-order semantics: the irony problem (user musings, 2026-06-12)

Live test result that frames everything: the zero-shot "ironic" axis detects
sarcastic SURFACE FORM, not intent. Marked sarcasm ("almost like you
aren't...") scores ironic — but as the user notes, marked sarcasm is barely
irony at all; it is a direct claim in roundabout clothing. Deadpan irony
("I will laugh at the children extra hard just for you") scores as the most
SINCERE line in its conversation, because deadpan is sincere-looking by
construction. Emote-stripping deletes tone markers (FeelsOkayMan) on top.

The user's theory of how humans actually do it, as design directions:

1. **First-order vs second-order meaning.** First order = what the words
   literally say. Second order = meaning conditioned on (a) the speaker's
   known values, (b) the in-group's acceptable values, (c) delivery register.
   Irony is a large first/second-order GAP: a statement orthogonal to the
   speaker's or group's true values, delivered casually. Operationalization:
   literal extremity (projection on harm/moral axes) x casual delivery x
   distance from speaker prior = deadpan-irony signal. A morally harmful
   opinion stated casually in a friendly conversation is usually a bit.
2. **Speaker prior / value profile.** Requires a per-person stance model —
   we have person vectors and per-message indexes; "how unusual is this
   content FOR THIS PERSON" is computable as a percentile within their own
   message distribution.
3. **Frequency as a light proxy for masking.** If someone "ironically" makes
   the same joke about X constantly, revealed preference says it is partly
   masking. Per-person topic-frequency conditions the playful-vs-masking
   call. (User: "even if someone claims something is ironic, if they
   constantly joke about X it can be somewhat masking.")
4. **Utterance merging.** "I wish" is an unreadable fragment alone — and it
   was sent seconds before "It would be a great day to cause suffering to
   children" by the same person. Messages should be merged into utterances
   by author + temporal proximity before any semantic analysis. Fixes
   fragments, emote spam, and a chunk of the context problem at once.
5. **Conversation-level effects (the sixth-person question).** Sort by
   temporal spacing, not just user: treat a conversation as the unit, measure
   its axis profile over time, and measure how the ARRIVAL of person X shifts
   it. "Does the conversation become more ironic / more race-oriented when X
   joins?" — a per-person INFLUENCE vector, possibly more insightful than
   their own message profile. Needs conversation segmentation (which #4
   gives). Confound to handle: time-of-day and topic seasonality.
6. **Learning without the oracle on novel messages.** Oracle labels bootstrap
   the supervised head; the second-order features (1-3) are what let it
   GENERALIZE to novel messages — the model learns "extreme content + casual
   register + off-prior = ironic" as a rule, not a lookup. Reaction-tracker
   laughter is a free weak label correlated with playful irony.

## Irony oracle v1 result: use axes, not one overloaded class

The first 60-item irony oracle pass is complete and was converted into a
private multi-axis dataset:

`data/unsynced/oracle/irony_v1_multi_axis.jsonl`

The main lesson: **hyperbole is not irony**. Hyperbole often preserves the
speaker's intended direction while exaggerating magnitude. "Worst ever" can be
a sincere negative judgment with comic magnitude, not a reversal of meaning.

Going forward, the model should predict separate axes:

- **Validity**: human utterance vs bot notice, pure link, pasted log,
  moderation line, or other non-semantic junk.
- **Literal-intended alignment**: aligned, divergent, unclear, not-applicable.
- **Magnitude distortion**: normal/literal, overstated, understated.
- **Play frame**: earnest/low-play, playful, masking-play.
- **Masking / facework**: absent, possible, present.
- **Hostility**: none, mild/mock, present.
- **Shock / attention seeking**: absent vs present.

The v1 class labels (`sincere`, `playful-sincere`, `hyperbolic-sincere`,
`playful-ironic`, `masking-ironic`) should be treated as a bridge from the
old review UI, not the final ontology. The next queue should ask these axes
directly, and the downstream model should be multiple small probes rather than
one "irony" classifier.

## Intent probe v0

`scripts/train_intent_probes.py` now trains those small probes from the private
multi-axis dataset. The default run writes ignored artifacts:

- `data/unsynced/intent_probes.pkl`
- `_private/INTENT_PROBES_REPORT.md`

The first run used the local bge-m3 embedding model with context and emote tags
expanded. It is only a seed model because 60 rows is tiny, but it already gives
a useful read on which axes are learnable:

- **Magnitude distortion / hyperbole**: modest real signal.
- **Play frame**: modest real signal.
- **Hostility**: modest real signal.
- **Literal/intended alignment**: weak; likely needs better second-order
  features, not just frozen text embeddings.
- **Masking / facework** and **shock / attention**: not enough positive labels.

That means the next queue should not be "more irony labels" in the old sense.
It should ask the axes directly and intentionally collect more rare positives,
especially masking and shock.

## Emote meaning — the five-source architecture (2026-06-12 night)

An emote's meaning is assembled from up to five signals, each covering the
others' blind spots. A learned/weighted combination (trained as labels
accumulate) should weigh them per-emote:

1. **Usage context** (built: emote_semantics.pkl) — mean embedding of messages
   the emote appears in, emote removed. The only source that works for DEAD
   emotes (old logs), misleading names, and fake personal emotes.
2. **7TV tags** (built: emote_registry.json, 1713/3087 tagged) — author-given
   topic words (SUSSY -> sus/suspicious). Cheap, often precise; can be
   ironic/gamed, which training should down-weight.
3. **Alias name** (built) — what people TYPE. Aliased emotes (507 in our
   registry) are the 'double awareness' case: the alias-word carries one
   meaning, the underlying image another, and the CLASH is usually the joke
   (a friend's name aliased onto a dancing-animal emote, said after an
   on-theme remark). Representable only once image semantics exist.
4. **Original name** (built) — the canonical emote name; weak (names lie).
5. **Image caption** (parked, spec in buckets) — VLM caption of the emote
   image, embedded into our space. Image URLs already stored in the registry;
   only available for CURRENT emotes (dead ones rely on #1).

Detection (is-this-an-emote): shape heuristic UNION the known sets (registry),
plus a learned classifier trained on the emote-suspect oracle queue for tokens
outside every list. Word-emotes ('ok') are near-1:1 image-to-word and low
value — intentionally not chased.

## The disgust-emote case study (the motivating example for emote semantics)

A real line: `DansGame women dont have dicks`, scored by the gay axis as the
NEGATIVE extreme. The pipeline strips emotes before embedding, so the model
read only "women dont have dicks" — an anti-pole statement. But DansGame
means DISGUST: the full reading is "I am disgusted that women don't have
dicks" -> "I wish they did" -> the OPPOSITE pole of what we scored. The emote
carries the entire semantic inversion, the way tone of voice does in speech.

Implications, in order of force:
1. Emotes are not decoration — they are sentiment/stance OPERATORS applied to
   the proposition. Stripping them deletes the operator. Naive inclusion
   (nomic/bge embedding of the emote token) doesn't capture the operator
   semantics either — the embedder has never seen DansGame used correctly.
2. This is THE argument for the domain-adapted embedder trained on our own
   logs, where DansGame co-occurs with disgust-context thousands of times.
3. Training must use CONTEXT, not isolated lines (user's point): an emote's
   meaning is learned from what it gets attached to across neighboring
   messages. The utterance-merging + msg-index infrastructure exists; the
   training pairs should be (message-with-emotes, surrounding context).
4. Until then: every axis/why/irony output that involves emote-heavy lines is
   systematically blind to inversions like this one.

## Open methodology questions (2026-06-12, second batch)

1. **Intellectualism vs intelligence.** The ~iq estimator measures EXPRESSED
   cognition; some people express their intelligence less by choice (good at
   math, never discusses it). The estimator is honest about being a register
   measure — but the gap is itself interesting: expression bias varies per
   person. Heuristically the two correlate (user's point, in caps), but
   per-person divergence between capability signals and expression frequency
   could be its own measurement.
2. **Predictive evaluation against reality.** The persona engine is never
   benchmarked the way it's conceptually trained: take a real moment where X
   spoke, hide X's actual message, generate X's reply from the context,
   compare generated vs actual (embedding similarity, classifier, eventually
   human/funniness). "Isn't that the whole point of predictive training?" —
   yes, and we don't do it. Build the held-out-reply eval harness; it also
   gives a per-persona quality score that doesn't depend on tics.
3. **Are the five built-in axes the right basis?** They were chosen by
   judgment, not data. Alternatives: PCA/ICA over person vectors to find the
   community's ACTUAL principal personality dimensions, then name them
   (LLM-label the extremes). Orthogonalization order also embeds judgment
   (menace-first means doomer = pessimism-beyond-hostility, not vice versa).
4. **Identity should be id-dominant.** Names change; Twitch ids don't. The
   author_ids table + live id capture make future merges factual; dead old
   logins (pre-capture) remain inference territory (temporal handoff + NCD).

## What this is trying to measure

The user is interested in whether high-volume chat logs can reveal stable,
replicable "personality-like" dimensions in the same broad spirit as classic
psychometrics: dimensions that survive new samples, new months, and maybe new
channels.

That does **not** mean forcing chatters into Big Five labels. Without real
questionnaire labels, we cannot honestly claim "this user is high openness" or
"this replicates Big Five." What we can do is discover chat-native latent
dimensions and test whether they are stable.

Possible dimensions:

- Joke/irony density.
- Aggression vs warmth in chat register.
- Question-asking vs claim-making.
- Topic breadth vs narrow obsession.
- Lore-reference rate.
- Emote reliance and emote variety.
- Message burstiness and double-texting.
- Caps/punctuation intensity.
- Night-owl or schedule rhythm.
- Direct social engagement vs ambient posting.
- Signature phrase concentration vs flexible vocabulary.

## Keep personality separate from social graph

A major trap: clustering people by who talks to whom mostly finds friend groups,
not personality. That is still useful, but it is a different object.

Keep three feature families separate until the analysis stage:

- **Style/language features:** message length, punctuation, caps, emotes,
  phrase entropy, slang, question marks, pronouns, repeated catchphrases,
  short-vs-long response habits.
- **Semantic/topic features:** embeddings of messages, topic distributions,
  games/media/person references, lore terms, channel-specific memes.
- **Interaction graph features:** who replies to whom, mentions, co-presence,
  timing around other users, thread participation.

For "personality" maps, start with style/language and semantic features. Use
interaction graph features as a separate overlay, not the main input. If a
cluster only appears when interaction features are included, call it a social
cluster, not a personality cluster.

## Candidate feature table

Build one row per author, optionally one row per author per month.

Basic filters:

- Minimum messages: 1,000 for a serious profile, 200 for a noisy preview.
- Drop commands, links-only lines, one-emote lines, and bot messages.
- Merge known alt accounts with `[archive.user_aliases]` before computing
  author-level metrics.

Interpretable numeric features:

- Median/mean message length and token count.
- Fraction of one-word, short, medium, and long messages.
- Caps ratio, punctuation ratio, question-mark rate, exclamation rate.
- Emoji/emote/token-like rate.
- Unique phrase ratio, repeated n-gram concentration, vocabulary entropy.
- Fraction of messages that mention another user.
- Fraction of messages that answer a question within a short time window.
- Burstiness: messages per active minute, multi-message streak length.
- Activity rhythm: hour-of-day and day-of-week distribution.
- Topic breadth: number of recurring semantic clusters.

Embedding features:

- Mean embedding of sampled messages.
- Mean embedding after subtracting the channel/month baseline.
- Topic centroids per author.
- Contrastive signature: what this author says unusually often compared with
  the same channel at the same time.

## Methods worth trying

Start simple and interpretable:

1. Export the per-user feature table to CSV/Parquet.
2. Z-score features within channel and month to reduce channel meta effects.
3. Run PCA or factor analysis on interpretable features.
4. Run UMAP only for visualization, not as the evidence by itself.
5. Use HDBSCAN or Gaussian mixtures for tentative clusters.
6. Compare clusters with and without interaction-graph features.
7. Generate a short report for each dimension/cluster with top features and
   example snippets kept private.

For embeddings:

1. Embed sampled messages locally or through a cheap embedding model.
2. Average by author and by author-month.
3. Subtract channel/month baselines to reduce "everyone is talking about the
   same stream moment" effects.
4. Cluster on the residual vectors.
5. Ask whether clusters are still visible in interpretable features.

## Replication/stability tests

This is the part that makes it interesting instead of just a pretty map.

- Split-half reliability: random half of each author's messages vs the other
  half.
- Time reliability: first half of the year vs second half.
- Month-to-month stability: author vectors should not jump wildly unless their
  behavior genuinely changed.
- Channel holdout: if an author appears in multiple logged channels, train the
  profile on one channel and test similarity in another.
- Baseline subtraction: compare raw clusters with clusters after removing
  channel/month averages.
- Bootstrap confidence: resample messages per author and report uncertainty.

A dimension is worth naming only if it survives some of these tests.

## Privacy and ethics

Private exploration over personally collected logs is one thing. Public
research, a paper, public charts, or identifiable scores are another.

Rules for anything public-facing:

- Ask included users for consent before using identifiable logs or examples.
- Prefer aggregate or anonymized results.
- Do not publish raw edgy/private quotes without permission.
- Avoid clinical labels or claims about mental health.
- Present scores as "chat behavior in this archive," not as a total description
  of a person.

## Useful deliverables

- Offline notebook: exports feature tables and renders maps.
- Private `~psyche <user>` command: fun stats, uncertainty, and nearest style
  neighbors.
- Cluster report: "what defines this cluster" with top features and private
  evidence snippets.
- Alias-aware profiles: known same-person accounts are merged before analysis.
- Drift report: how a user's chat style changes over months or years.

## First implementation slice

Small, useful, and not overbuilt:

1. Add `scripts/export_personality_features.py`.
2. Read from `data/unsynced/chat_archive.db`.
3. Output `data/unsynced/personality/features.csv`.
4. Include only interpretable features first.
5. Add a second script/notebook for PCA/UMAP plots.
6. Add embeddings only after the baseline table is sane.
