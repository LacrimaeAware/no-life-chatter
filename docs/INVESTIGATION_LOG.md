# Investigation Log — persona / personality system

For any model **auditing** this work: this is the full trajectory, including the
claims we **overturned**. Read it before trusting any single finding, so you
don't re-introduce a dead claim or repeat a falsified step.

## Language discipline (apply this when writing here)

- This work **empirically found associations** and **measured** things. It has
  **not "proved"** anything. Do not use "proved / proves / proof / demonstrates
  conclusively."
- On small N (e.g. 34 chatters, or a 12-person hand-labeled group), a result is a
  **descriptive lean / heuristic**, not a trained model and not a proof.
- Tags below: **FOUND** = an empirical observation on stated data · **OVERTURNED**
  = an earlier claim this work falsified (do not repeat it) · **CORRECTED** = a
  mistaken interpretation we fixed · **LIMIT** = a known wall / parked frontier.

## 1. Embedding geometry

- FOUND: raw person-vector cosines are anisotropic (mean |off-diagonal| ≈ 0.983).
- FOUND: mean-centering already reduces that to ≈ 0.165.
  **OVERTURNED** the audit claim that "centering leaves every cosine ≈ 0.99."
- FOUND: an all-but-top-k (ABTT) isotropy correction is **non-monotonic in k** —
  k=1 was *worse* than both k=0 and k=2 on trait-axis entanglement.
  **OVERTURNED** the assumption that k=1 is the safe/obvious choice. Shipped k=2.
- FOUND: trait axes are mean-difference "steering vectors"; their per-person
  z-scores were ≈ 0.48 associated across the roster. Löwdin (symmetric)
  orthogonalization + ABTT lowered that to ≈ 0.25.
- **OVERTURNED**: "menace · doomer = 0.91" — that figure was a **stale code
  comment** from an earlier build; the measured cosine is ≈ 0.64.

## 2. Irony / intent

- FOUND: the zero-shot "ironic" axis fires on *marked* sarcasm but not on
  deadpan/charged irony; a chatter the owner calls highly ironic scored as
  "sincere" on it, with a tiny dynamic range.
- **OVERTURNED**: the intuitive fix "down-weight a person's ironic messages when
  scoring a charged axis." Person-level irony↔menace association was only +0.17,
  and there was no reliable irony signal to down-weight in the first place.
- LIMIT: irony is not readable from a single message — it needs whole-conversation
  context, the speaker's prior, and **who the line is about** (subject attribution).

## 3. Contradiction / performativity

- FOUND: chatters who occupy BOTH poles of a charged axis (the "contradiction"
  signal) tend to be performative; it matched the owner's read on a clear
  true-negative and true-positive.
- **CORRECTED**: this is a *performativity / range* proxy, NOT an irony detector.
  It false-positives on a sincerely-moody (genuinely bidirectional) person. The
  in-chat flag wording was softened from "performative" to "unreliable read,
  could be irony or just range."

## 4. Topic vs. behavior

- FOUND: unsupervised axes over the person vectors are dominated by **topic**
  (e.g. gaming-vs-coding) and **language**, not personality — the embedder
  encodes *what* people talk about. This is the ceiling on embedding-only personas.
- FOUND: topic-free **behavioral** features (emote rate, verbosity, caps,
  @mentions, profanity, vocabulary) give interpretable axes that match the human
  read, and are ≈ 0.22 associated with the topic axes (mostly independent).
- **CORRECTED**: a "doubles" feature (whole-message "X X") is largely a
  **logging/import artifact**, not a behavior. It was removed from the `~style`
  read. (Per-message de-doubling is retained — it cleans the *other* counts.)
- CAUTION: **language written ≠ nationality.** Two misreads happened (a line in
  Czech, a line in Portuguese, each mis-attributed to the speaker's origin or
  intent). Subject attribution is unsolved and is the top forward capability.

## 5. The core conclusion: structural vs. intent traits

- FOUND: personality traits split into two kinds.
  - **Structural** — *how* someone types. Behaviorally measurable; matches the
    human read. This is what `~style` reports.
  - **Intent / disposition** — *what they mean / who they are*: irony, hostility,
    sincerity, masking, Schadenfreude. **Not recoverable from surface features.**
- FOUND: a keyword **hostility** proxy ranked a chatter whose edge is an ironic
  *bit* at the top, above the chatters the owner labels genuinely cruel. The proxy
  ranks delivery, not intent — the same wall as irony.
- FOUND (association, N≈12 hand-labeled): chatters the owner labels "over-the-top"
  lean higher than "low-key" ones on caps (≈ +1.15 z), exclamation (≈ +0.90),
  emotes (≈ +0.91), and @mentions (≈ +0.76). A descriptive lean, not a model.

## 6. Integrated vs. parked (current state)

- LIVE in the bot: `~style` (behavioral / structural), `~traits` (old hand-picked
  axes, now Löwdin-decorrelated, with the ⚡ both-poles flag), `~top`, and RRF
  bm25+dense retrieval in `~askchat`.
- PARKED (genuinely hard): the intent/disposition traits. They need subject
  attribution, a speaker-prior model, and **labeled messages** — not just labeled
  people. The 34 person-labels in `_private/PERSON_LABELS.md` are **too few to
  train** a model (it would be a glorified mean); they are an **answer key /
  association source**, used so far to validate the structural reads and to show
  the intent traits are *not* behaviorally recoverable.

## 7. `~style` build + emote detection (a worked example of the discipline)

- FOUND: behavioral features surface per-person as `~style`, z-scored vs the
  roster; the reads match the owner's labels (e.g. the emote-spammer, the
  cleanest-mouthed, the @-everyone chatter land where expected).
- CORRECTED — emote detection, **twice, each caught only by reading real
  messages**:
  - A shape-only heuristic UNDER-counted (it missed Capitalized-first emotes:
    Sadge/Pog/Lemon) — read as near-zero for a heavy emote user.
  - A case-insensitive registry match then OVER-counted — it flagged common
    words ("there", "omg") that collide with 7TV emote names, and an all-caps
    rule flagged every SHOUTED word ("HELP", "YOU").
  - Current rule: require MIXED case, then camelCase OR exact-case registry. This
    uses the two real signals the owner pointed out — **capitalization** ("Pain"
    the emote vs "in pain" the word) and **registry membership**. Residual:
    sentence-initial capitalized words; all-caps emotes (KEKW/OMEGALUL) dropped to
    avoid flagging shouting (most emotes are NOT all-caps, so this loss is a
    minority — a small allow-list could recover them). Emote rate is an
    approximate **lower bound**, not exact.
  - LESSON: each error was invisible in aggregate and obvious in the messages.
    Verify feature extractors against real messages, not intuition.
- CORRECTED: a "doubles" feature (whole-message "X X") was a logging/import
  artifact, not a behavior — removed from `~style`.
- FOUND (a genuine use of the oracle): the 12 over-the-top/low-key labels define
  a behavioral direction; scoring all 34 on it **generalizes the labels** — it
  surfaced un-labeled candidates and flagged disagreements (the owner's two
  accounts split his behavior across eras). Useful, honest — not a trained model.

## Do-not-repeat list (for auditors)

- Do not reintroduce "centering leaves cosines at 0.99" or "menace~doomer 0.91."
- Do not claim a behavioral "irony" or "hostility" score — it ranks the loudest
  ironist as the cruelest.
- Do not call the 34 person-labels a training set, or any of this "proof."
- Do not trust an emote/feature extractor without checking real messages, and do
  not call emote rate exact — it is an approximate lower bound.
- Do not declare something "fundamentally impossible" when capitalization,
  the registry, or token position carry real signal.
- Do not infer a chatter's nationality/identity from the language a message is in.
