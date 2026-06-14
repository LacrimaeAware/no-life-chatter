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
6. **Hard, later:** target/relationship + whole-context features for irony — the
   part single messages can't do.

## Honestly not solvable cheaply

- Per-message irony (needs context + speaker prior + target).
- Sincere-bidirectional vs ironic-bidirectional (needs intent; the ⚡ flag only
  says "both poles, unreliable," not which).

The throughline: **stop forcing personality out of a topic embedder; measure it
from behavior, anchor it to a human oracle, and let a classifier generalize it.**
