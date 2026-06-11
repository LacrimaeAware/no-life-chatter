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
