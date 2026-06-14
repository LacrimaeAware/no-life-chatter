# Personality System — Design & Synthesis

The coherent plan behind the scattered experiments. Read this to know what the
personality/trait system IS becoming and what we keep vs. replace. Public-safe:
no real handles, no human descriptions of real people (those live privately in
`_private/PERSON_LABELS.md`).

## Goal

Measure a chatter's personality from their chat history in a way that matches how
a human who knows them would describe them — not a vibe that breaks immersion.

## What we learned (and why the old axes feel wrong)

1. **The five hand-picked trait axes are the wrong instrument.** menace, ironic,
   unhinged, professor, doomer were chosen by judgment ("randomly"), built as
   constant mean-difference *steering vectors* on a frozen encoder (the exact
   construct the `structured-transform-discovery` research falsified), and they
   are **mislabeled**: "menace" actually measures NEGATIVITY (dislike / sadness /
   doom), and "wholesome" measures mundane positivity. (`scripts/eval_geometry.py`)
2. **The embedder is a topic machine, not a personality one.** Discovering axes
   straight from the data (`scripts/discover_axes.py`) shows the strongest
   directions are TOPIC (gaming vs coding) and LANGUAGE (English vs Spanish/
   German), not personality. Personality is a faint residual after topic. That is
   the ceiling on any embedding-only persona approach.
3. **Irony can't be read from a single message.** It needs whole-conversation
   context, the speaker's history/values, and WHO the line is about (third person
   vs. to a friend's face). The zero-shot ironic axis cannot see deadpan/charged
   irony. (`scripts/irony_confound.py`)
4. **Personality lives in BEHAVIOR, not topic.** How someone types — caps rate,
   emote rate, message length, @mentions, profanity, repetition, vocabulary — is
   topic-free and matches human reads. The behavioral axes are ~independent of the
   topic axes (mean correlation 0.22). (`scripts/behavior_axes.py`)
5. **Human labels on PEOPLE are the ground truth we were underusing** — the
   "oracle on people, not messages."
6. **SUBJECT ATTRIBUTION is the single biggest missing capability.** A message's
   sentiment and identity must be assigned to *the right referent*: the speaker,
   the person addressed, or a third party — and the system must recognize when it
   is genuinely ambiguous. "they're not a bot, just [from country X]" is the
   speaker mocking a third party, not the speaker's own nationality; "shit bait
   hating on everyone" can be a person *quoting* someone else's behavior, not
   being hateful themselves. Reading sentiment/nationality/trait off the words
   without resolving the referent is the root of repeated misreads. This is the
   top forward capability.
7. **Behavior is dataset/chat-and-era dependent.** The same person looks very
   different across chats: in a big streamer's chat *everyone* emote-spams, so the
   emote-spam axis is largely a chat-CULTURE artifact, and an account from an
   older era can behave unlike the same person's newer account. Behavioral
   training must be scoped to a chosen dataset (e.g. the two home chats), not
   pooled blindly across every channel and era.

## The core split: structural traits vs. intent/disposition traits

The single most important finding from labeling the whole roster and testing
behavioral proxies against the human read: **personality traits fall into two
kinds, and only one of them is measurable from data we have.**

1. **Structural traits — HOW someone types.** Emote-reliance, verbosity, caps/
   intensity, deadpan (low exclamation), attention-via-@mentions, copypasta/
   doubling. These ARE captured by topic-free behavioral features
   (`scripts/behavior_axes.py`) and match the human read.
2. **Intent / disposition traits — WHAT someone means / WHO they are.** Irony,
   sincerity, hostility, Schadenfreude, masking, performative-vs-genuine. These
   are NOT recoverable from surface features: the same words carry opposite
   intent. Measured directly — a chatter whose edgy lines are an ironic *bit*
   ranked #1 on a hostility keyword proxy, above the chatters the human calls
   genuinely cruel; the proxy ranks delivery, not intent. Same wall as irony.
   These need context + the speaker's prior + who the line is aimed at
   (subject attribution), and the **human oracle is their only ground truth.**

Design consequence: build the structural half from behavior now; treat the
intent/disposition half as the hard frontier that depends on subject-attribution
and a speaker model, not on more surface features. Do not promise a behavioral
"hostility" or "irony" score — it will rank the loudest ironist as the cruelest.

## The design: two layers and a bridge

**Layer 1 — Behavioral axes (the measurable personality space).**
Topic-free counted features → discovered axes, named with a human. These dodge
the topic problem and are interpretable. Examples found so far: emote-spam-hype,
crude-vs-clean, talks-at-people-vs-broadcasts. Extend with timing (night-owl),
reply latency, and the target/relationship signal (third-person vs to-face).

**Layer 2 — Human person-oracle (the ground truth).**
A structured human read per chatter: irony type (none / unidirectional /
bidirectional), hyperbole, sincere vs performative, intense, attention-seeking,
moody, deadpan. Stored privately (`_private/PERSON_LABELS.md`). This is the
oracle — about people, built once, reused forever.

**The bridge — a small classifier.**
Train: behavioral features (+ sentiment, contradiction, emote-stance signals) →
predict the human labels. When it predicts the held-out labels well, it has
generalized the human oracle to the whole roster. THIS is the real `~traits`
replacement — personality the way a human would call it, grounded in labeled
truth, not in arbitrary pole sentences.

## Keep, replace, or demote

| Component | Decision |
|---|---|
| 5 topic-embedding trait axes (menace/ironic/…) | **Demote.** Keep as a topic/sentiment "vibe" read (fine for `~vibes`); stop treating them as personality truth. |
| Löwdin decorrelation, ABTT, the ⚡ contradiction flag | **Keep.** They make the existing axes honest in the meantime. |
| Behavioral axes (`behavior_axes.py`) | **Promote to the personality foundation.** |
| Human person-oracle (`PERSON_LABELS.md`) | **Promote to ground truth + classifier target.** |
| Topic embeddings (person vectors, `~like`/`~twin`) | **Keep for what they're good at** — topic/voice similarity and alt-detection — just not personality. |

## Roadmap (phased — this is "what we're doing from here")

1. **Name the behavioral axes** with example messages per pole. *(in progress)*
2. **Finish the person-oracle:** label all 34 chatters on the dimensions above,
   stored privately. *(~19/34 done)*
3. **Validate:** do the behavioral axes actually predict the human labels?
   Measure correlation per dimension. Drop features that don't.
4. **Bridge:** train the behavior→label classifier; report held-out accuracy.
5. **Ship:** a `~personality`/`~style` read from the validated system; keep the
   old axes only for topic-vibe.
6. **Subject-attribution layer (high priority, not "later"):** resolve who each
   message is about (speaker / addressee / third party) and flag ambiguity, so
   sentiment and traits attach to the right person. Start with cheap signals
   (@mentions, second-vs-third-person pronouns, quotation framing) and a "can't
   tell" class; this is the prerequisite for trustworthy irony and trait reads.
7. **Dataset scoping:** build the behavioral/personality layer on a chosen,
   coherent dataset (the home chats), not pooled across every channel/era, since
   chat culture and era dominate raw behavior.
8. **Candidate axis to test:** energy-expression vs. social-conformity — how much
   a person prioritizes expressing their own energy vs. going along with the
   room. Likely correlates with the over-the-top/attention dimension but the
   owner believes it is distinct; measure it.

Note on the "professor" axis: it measures *presentation* (long, articulate
sentences), not intelligence — a high scorer can read as low social-intelligence
while an actual physicist scores high for different reasons, and professor and
brainrot co-occur in the same person. Do not equate any single axis with a value
judgment.

## Honestly not solvable cheaply

- Per-message irony (needs context + speaker prior + target).
- Sincere-bidirectional vs ironic-bidirectional (needs intent; the ⚡ flag only
  says "both poles, unreliable," not which).

The throughline: **stop forcing personality out of a topic embedder; measure it
from behavior, anchor it to a human oracle, and let a classifier generalize it.**
