# Project Audit - 2026-07-15

This is the dated evidence behind the July 15 state/roadmap refresh. It records
systemic findings only; private identities, raw receipts, and local config stay
outside tracked docs.

## Scope

- Runtime lifecycle, command discovery, permissions, and model scheduling.
- Archive schema, alias normalization, context, deduplication, and provenance.
- Classifier, person vectors, message indexes, axes, IQ, comedy, and irony.
- Persona retrieval/generation and output validation.
- Fact extraction, verified profiles, and `~askchat` synthesis.
- Public help, command documentation, artifact checks, and maintenance scripts.

The command auditor found 56 live command modules with 56 valid handlers and no
missing command-bible entries. The focused unit suite finished with 135 passing
tests after the changes below and the July 16 continuation audit.

## Principal Findings

### 1. Identity was normalized at read time but not governed as artifact input

Runtime alias chains worked, yet generated files could retain both sides of a
later merge. Display-name selection also treated an offline ID-resolution check
as recent live activity, allowing maintenance order to choose an old alias.

Resolution:

- preserve raw imported names separately from canonical searchable names;
- track real live recency separately from resolver timestamps;
- fingerprint the alias map in identity-sensitive artifacts;
- warn on split canonical groups, missing fingerprints, or changed maps;
- give the claim bank versioned identity/build metadata too;
- rebuild artifacts after identity changes.

### 2. Sampling was inconsistent and repeated expensive embedding work

Different semantic builders sampled independently, so person averages, message
receipts, and IQ could describe different slices. Some global sampling used SQL
randomness, making rebuild comparisons noisier.

Resolution:

- one deterministic 3,000-utterance index per person;
- 80% stable coverage and 20% high-information retrieval lanes;
- channel-bounded utterance v3, including duplicate-component collapse,
  instead of joining simultaneous posts from unrelated channels or repeating
  alias-mirrored evidence;
- person vectors mean-pool only coverage rows;
- IQ reuses eligible indexed rows and embeds only as fallback;
- deterministic corpus-frequency sampling;
- normalized exact dedupe before authorship train/test splitting and long
  cross-person copypasta rejection for IQ.

The 3,000 cap is not universal. Authorship keeps 4,000 filtered messages, IQ
lexical features can inspect 15,000 filtered utterances, and fact retrieval uses
deeper targeted history. One sample policy for every task would be simpler but
less valid.

### 3. IQ had useful aggregation but poor auditability

The score already used the median of each person's top 10%, not a mean over
ordinary Twitch filler. The larger failures were opaque embedding axes, stale
identity inputs, non-deterministic corpus sampling, and pasted text receiving
credit.

Resolution:

- preserve top-tail aggregation;
- raise semantic evidence to 3,000 deterministic utterances;
- filter exact long copypasta shared by canonical people;
- blend semantic reasoning moves with direct clause/reasoning and question
  structure instead of making the default reasoning score embedding-only;
- store median/top examples for lexical and semantic components;
- expose strongest/weakest or per-dimension receipts through `~iq why`.

Remaining risk: unique pasted prose and near-copies can still score. Embedding
reasoning axes are register resemblance, not verified reasoning, and should not
gain weight without a held-out task.

### 4. Archive QA retrieved evidence but discarded some paraphrases and context

Dense retrieval could find a valid paraphrase and a later lexical focus gate
could remove it. Even surviving receipts were shown to the answer model as
isolated lines, despite existing source-aware context machinery.

Resolution:

- preserve unanchored dense hits only above a stricter cosine floor;
- retain BM25 as an independent RRF lane;
- attach chronological context windows to answer evidence;
- use the archive's alias dedupe and author-only-source guard;
- keep raw/receipts mode and suppress weak regex claims from synthesis.

Remaining risk: arbitrary questions still need evaluated query expansion and
all-author dense retrieval. A few mentions cannot establish a stable opinion.

### 5. The old fact bank confused extraction confidence with truth confidence

Regex matches could become confident through repeats, merged bursts could make
the captured tail absorb unrelated lines, and contradictions were not explicit.

Resolution:

- extract from message-local rows;
- count exact evidence once;
- separate extraction confidence from evidence confidence;
- require independent days and fresh phrasings for promotion;
- block promotion on cross-user echo or opposing claim forms;
- scan up to 20,000 rows because facts need deeper history than personas.

The regex bank is now a receipt/candidate source, not a truth database.

### 6. Verified memory was better designed but too expensive to cover the roster

The contextual slot-profile system already rejected jokes/copypasta and required
multi-day support, but only a small number of people had been built because it
used one model call per candidate.

Resolution:

- add ordinary/unusual/impossible plausibility labels;
- require three independent days for unusual claims;
- diversify retrieval across days and increase its cap;
- judge multiple context-marked candidates per model call;
- retry only missing batch items and preserve the incremental cache.
- checkpoint each completed slot with valid metadata so interrupted long builds
  really resume instead of discarding paid-for judgments.

Shared artifacts are written by atomic replacement. Live readers therefore see
the previous complete artifact or the next complete artifact, never a partial
pickle/JSON file during background maintenance.

Remaining work is operational: run and review a full active-roster build, then
schedule freshness by active user.

### 7. Irony needs evidence hierarchy, not a larger zero-shot axis

Surface sarcasm/extremity vectors cannot determine whether a literal statement
is a meme, implausible bit, or contradiction of a person's known history.

Resolution:

- keep surface axes as weak evidence;
- add community near-copy/echo counts;
- parse a small set of high-precision literal claims;
- compare claims with confirmed profile facts;
- explain which evidence changed the read.

This remains experimental until reviewed intent labels support a supervised
model. The fix is not a per-user exception list.

### 8. Comedy scoring had cache and causal-credit leakage

Results stayed cached for the process lifetime, bot/noise laughers could enter
windows, rapid fragments could each receive one later laugh, and breadth summed
the same person again across chats.

Resolution:

- invalidate on archive growth plus a TTL;
- remove bot/noise accounts from both before and after windows;
- collapse rapid same-speaker fragments into one setup;
- aggregate laugher identities across chats before breadth/effective-N scoring.

### 9. Documentation described several shipped systems as future work

Resident personas and queue feedback were live while active state/roadmap docs
still called them planned. Artifact status printed WARN rows but its script
returned success, making the wrapper label the step OK.

Resolution:

- refresh state, roadmap, command bible, bot-mode spec, and README;
- support category-aware `~help` while keeping gated commands separate;
- mark experimental commands compactly;
- return nonzero from artifact status whenever any artifact warns or is missing.

## Ranked Verdict

The July 16 continuation completed the identity-sensitive IQ and claim builds,
then moved the next effort to persona quality instead of starting another IQ
run. It also added a model-free multi-candidate persona reranker. In replay over
113 recent multi-candidate events, the reranker removed all exact score ties,
changed 47 selections, raised mean target-classifier probability from 0.1730 to
0.2096, and raised the voice/context proxy scores without adding a model call.

Verified-profile candidate filtering and slot-aware normalization were tightened
before spending further model time. A one-user isolated benchmark judged 55
candidates in 4 minutes 14 seconds, implying about 2 hours 50 minutes for 40
dense users at the same workload. That full run was deliberately deferred for
dead hours; the live profile reader fails closed while the old artifact is
stale. A targeted emote build also raised the three audited emotes to 160, 160,
and 92 honest distinct contexts, without fabricating missing evidence.

The IQ build itself completed in 31 minutes with all 40 users receiving both
semantic and judge coverage. The audit found one unresolved validity issue:
semantic rows still pass through a rare-word/reasoning signal filter, and the
current niche-depth feature measures distance from the corpus center more than
sustained topic depth. A recurrence-clustering prototype emphasized repeated
chat rituals, so it was rejected rather than promoted into another rebuild.

1. Freeze and review held-out persona benchmark cases.
2. Validate and extend the shipped persona reranker.
3. Run and inspect verified profiles v5 during dead hours.
4. Top up broadly used emotes with honest context coverage.
5. Add evaluated query planning and broader dense QA.
6. Correct and benchmark IQ semantic sampling and depth geometry.
7. Train intent/irony only from reviewed labels.
8. Coordinate offline model jobs with the live queue across processes.

Fine-tuning is downstream of these items. More examples help only after the
examples are correctly shaped, deduped, contextualized, and measured.
