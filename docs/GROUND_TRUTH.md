# NoLifeChatter — Ground Truth

Last verified: 2026-06-14 · HEAD `98584a3` + uncommitted 2026-06-14 geometry pass

**Read this first.** This is the canonical, terse source of truth for operating
rules and verified numbers. Two hard rules govern it:

1. **Every number in the ledger below is re-checkable by a command.** Do not
   trust a number in prose — run its check. Numbers in documents rot silently
   (this repo once carried `menace~doomer = 0.91` in a code comment long after it
   was `0.64`; a downstream audit repeated the stale value as fact). The fix is
   not "be careful," it is `scripts/ground_truth_check.py`.
2. **If you change something, update the ground truth in the same commit.** A
   change that moves a verified number must update `EXPECTED` in
   `scripts/ground_truth_check.py` *and* the ledger here, together.

This file is intentionally short. It does not duplicate the long docs; it points
to whichever doc *owns* each truth (see the ownership map) and records only the
rules and numbers that, if wrong, cause real mistakes.

---

## Operating rules (non-negotiable)

**Runtime / live bot**
- The bot runs `run-background.vbs` → `_bot-loop.bat` → `chatbot.py`. The loop
  respawns the worker on exit. `chatbot.py` holds a single-instance lock on
  `127.0.0.1:48917`.
- **Never launch `chatbot.py` manually** to "test." A second launch exits on the
  lock. To reload code, kill the `python … chatbot.py` worker and let the loop
  restart it; use `stop-bot.bat` for an intentional stop.
- **Never edit `_bot-loop.bat` while the loop is running.**
- Code changes in `utils/`, `services/`, `commands/` only reach the live bot
  after a worker restart.

**Privacy boundary**
- `_private/`, `config.toml`, `.env`, `data/synced/`, `data/unsynced/`, logs,
  token files, and model artifacts are gitignored. Never stage them.
- **Before any commit, grep the staged ADDED lines for real handles/channels**
  and token-shaped strings. Tracked docs and the public site must stay
  anonymized — no real usernames, rosters, aliases, or raw chat lines.

**Working method**
- **Verify before claiming.** Any quantitative claim about embeddings, geometry,
  retrieval, or model quality must come from a command that can be re-run, not
  from memory or a previous doc. Prefer building the dial over asserting the
  result. (See `docs/RESEARCH_TO_APPLIED.md` §0.)
- **Persona output:** every posting path is `generate → output_filter → send`.
  Generated text must stay visibly bot output. Judge persona quality by
  funniness / in-character behavior, not token-similarity stats.
- Do not add `Co-Authored-By` trailers to commits in this repo.

---

## Verified numbers ledger

Re-verify all of these at once:

```powershell
.\.venv\Scripts\python.exe scripts\ground_truth_check.py
```

Geometry numbers (`*` needs LM Studio up for the axis rows):

| Metric | Value | Meaning / why it matters |
|---|---|---|
| `ABTT_K` | **2** | All-but-top-2 isotropy correction in `persona_embeddings._centered()`. Chosen empirically (the effect is non-monotonic; k=1 is worse). |
| person vectors | **34** | Roster size in `data/unsynced/persona_embeddings.pkl` (grows on rebuild; informational). |
| person anisotropy | **0.149** | Mean \|off-diagonal cosine\| of the production centered+ABTT person matrix. Raw was 0.983; **centering already fixes most of it** — ABTT is a refinement, not a rescue. |
| axis-score entanglement | **0.249** | Mean \|off-diagonal\| correlation of per-person trait z-scores (`traits_for`). Was **0.483** with the raw steering-vector axes; Löwdin + ABTT cut it ~48%. This is the real "axes feel the same" number. |
| menace~doomer axis cosine `*` | **0.64** | Cosine of the two most-collinear raw trait axes. **NOT 0.91** — that was a stale comment. The genuine collinearity is moderate. |
| doomer Löwdin alignment `*` | **0.917** | How aligned the Löwdin-orthogonalized doomer axis stays with its raw direction (Gram-Schmidt collapsed it to 0.73). Confirms axis labels remain valid after decorrelation. |

Lineage and the full before/after tables: `docs/RESEARCH_TO_APPLIED.md`. The dial
itself: `scripts/eval_geometry.py`.

---

## Known limitations (do NOT claim these are solved)

These are measured, recurring failure modes. They break user immersion and a
model working here must not paper over them or claim the trait/attribution
commands are accurate. Re-measure with `scripts/irony_confound.py`.

- **Charged trait axes read words, not intent.** `~traits`/`~top` on a charged
  axis (menace, or custom racism/misogyny axes) scores *surface content*. Someone
  who says edgy/charged things **ironically, as a bit** scores like someone who
  means it; the man who sincerely means it may not top the list. Measured: a
  highly-ironic chatter's most-"menace" lines are crude jokes (*"my piss could
  annihilate any poop"*), and the **zero-shot "ironic" axis cannot detect this
  kind of deadpan/charged irony** — its per-message projection range for that
  chatter is only ≈0.20 wide (p95−p5) and centered near zero, so there is no
  reliable irony signal to discount. Person-level irony↔menace corr is only +0.17, so
  "discount ironic messages" is **not** a working fix. A true intent fix needs a
  **supervised** irony detector (the irony oracle queue), which is blocked on
  labels and cannot work per-message anyway (irony needs whole-chat context +
  speaker history). The no-oracle partial read that *does* work today is
  **self-contradiction** (`scripts/contradiction.py`,
  `persona_msg_index.contradiction_scores`): a performative chatter occupies both
  poles of a charged axis at once. **`~traits` now ships this** — a charged-axis
  lean is marked ⚡ when the chatter also lives at the opposite pole, so the bot
  reports "unreliable" instead of a confident sincere score. It is a
  performativity/range proxy, **not** an irony detector, and it **cannot separate
  ironic-bidirectional from sincere-bidirectional** — a genuinely moody chatter
  (really angry sometimes, really happy other times) trips it the same as an
  ironic shitposter. So ⚡ means "this lean is unreliable," not "this person is
  insincere." User-validated 2026-06-14: correctly leaves a rarely-ironic chatter
  unflagged, but flags a sincerely-moody one as a false positive
  (`_private/CONTRADICTION_VALIDATION.md`).
- **`~whosaid` is authorship, not human-association.** On one or two words it
  answers "which model class maximizes likelihood," which is noisy and can name a
  statistically-plausible author over the one every human in chat would
  instantly guess (the person whose *bit* the phrase is). These are two different
  questions; the classifier owns the first, not the second. Short inputs are
  inherently noisy — treat low-token attributions as weak.
- **Multilingual confound.** The embedder (BGE-M3) is multilingual; a chatter who
  writes in another language (e.g. Brazilian Portuguese) can land near others by
  language as much as by voice/topic. Watch for this in `~like`/`~whosaid`.
- **`~style` measures STRUCTURE, not intent; emote rate is approximate.** `~style`
  reads *how* a chatter types (the part of personality that IS measurable). It
  cannot read irony/sincerity/hostility — those are intent (see
  `docs/INVESTIGATION_LOG.md` §5). Its emote rate is an approximate lower bound:
  emote detection from archived text is hard because 7TV emotes are named after
  common words; the current rule uses capitalization + the registry and drops
  all-caps emotes to avoid flagging shouting. σ in `~style`/`~traits` is relative
  to THIS roster (an emote-spam-heavy chat), not to humans generally.

These are honest open problems, not bugs to hide. The trait/attribution/style
commands are a fun mirror, not a measurement of what a person sincerely is.

---

## Document ownership map

Each truth has exactly one owning document. When sources disagree, the owner
wins; fix the others.

| Truth | Owner | Notes |
|---|---|---|
| Operating rules + verified numbers | **this file** (`docs/GROUND_TRUTH.md`) | Read first. |
| Current runtime state, return-to-work map | `docs/STATE_OF_OPERATION.md` | First read after a break. |
| Ranked next work | `docs/ROADMAP.md` | Recently-shipped + frontier. |
| Live command bible | `docs/COMMANDS.md` (+ `README.md` table) | Kept in sync by `scripts/check_readme_commands.py`. |
| Embedding/persona geometry rationale + research lineage | `docs/RESEARCH_TO_APPLIED.md` | Porting `structured-transform-discovery` findings. |
| Personality-system design (what we keep/replace + the plan) | `docs/PERSONALITY_SYSTEM_DESIGN.md` | The coherent synthesis: behavioral axes + human oracle + bridge classifier. |
| Personality-research questions + ethics | `docs/CHAT_PERSONALITY_RESEARCH.md` | Private/art-first; no public clinical claims. |
| Archive/FTS schema + query rules | `docs/CHAT_ARCHIVE.md` | FTS-first `CROSS JOIN` performance rule lives here. |
| Future ideas (not tasks) | `docs/IDEA_BANK.md` | Ideas graduate to ROADMAP when chosen. |
| Cross-repo portfolio synthesis | `master-organizer/docs/REPO_PORTFOLIO/project-bibles/nolifechatter.md` | The portfolio "bible"; keep it in sync (see contract). |

---

## The update contract

When you do X, update Y **in the same commit**:

| You changed… | You must update… | And run… |
|---|---|---|
| Embedding geometry, axis math, or rebuilt `persona_embeddings.pkl` | `EXPECTED` in `scripts/ground_truth_check.py` + the ledger above | `scripts/ground_truth_check.py` |
| A command (added/renamed/removed/help text) | `README.md` table + `docs/COMMANDS.md` | `scripts/check_readme_commands.py`, `scripts/audit_commands.py` |
| Current runtime behavior or next-work priority | `docs/STATE_OF_OPERATION.md`, `docs/ROADMAP.md` | `scripts/freshness_check.py` |
| Anything that changes "current truth" of the repo | the master-organizer bible + audit (`PROJECT_DOC_SYNC_PROMPT`) | the master-organizer dashboard refresh |

One command checks the whole repo's doc/number/command freshness:

```powershell
.\.venv\Scripts\python.exe scripts\freshness_check.py
```

It runs: docs-layout sanity, command audit, README↔command sync, artifact status,
**ground-truth number check**, rebuild-log status, and git dirtiness. A
ground-truth `DRIFT` is a WARN (a legitimate rebuild moves the numbers) — it is a
prompt to re-baseline `EXPECTED` and this ledger after confirming the new values
are sane, never to silently widen a tolerance.
