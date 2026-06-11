# Persona bot — roadmap

The fun-features arc for NoLifeChatter: turn the chat archive into per-user
**personas** the bot can roleplay as — an *accurate* one that genuinely
emulates a chatter's tendencies, vocabulary and rhythms, and a *hyperbolic*
one tuned for comedy — plus rare spontaneous in-character reactions, playful
psychometrics, and trivia.

**Status: roadmap — phases 0–1 are prerequisites and unglamorous; everything
fun builds on them.** Companion docs: [CHAT_ARCHIVE.md](CHAT_ARCHIVE.md) (the
data layer), [IDEA_BANK.md](IDEA_BANK.md) (smaller ideas, parked things).

## Feasibility verdict

Very feasible. Nothing here is research-grade; it's plumbing plus careful
prompting. The bot framework (command auto-discovery, SQLite, async send paths)
already exists, which removes most of the usual startup cost. Honest effort
estimates, assuming evenings/weekends:

| Phase | What | Effort | Needs LLM? |
| --- | --- | --- | --- |
| 0 | Chat archive (ingest + live capture) | an afternoon | no |
| 1 | Archive Q&A commands (`~said`, `~quote`, stats) | an evening | no |
| 2 | Persona cards (accurate + hyperbolic, offline batch) | a weekend | yes, offline |
| 3 | Runtime roleplay (`~as <user>`, rare random reactions) | an evening | yes, online |
| 4 | Playful psychometrics (Big-5-flavored profiles) | an evening+ | yes, offline |
| 5 | Trivia (quote-attribution game, film trivia) | an evening | no (optional) |

Phases 2–4 are independent once 0 exists; build in any order after 1.

## Phase 0 — Chat archive

Everything reads from one SQLite database: historical Chatterino logs ingested
once, live messages appended by the bot. Full design (format spec, schema,
ingest CLI, query CLI) in [CHAT_ARCHIVE.md](CHAT_ARCHIVE.md).

## Phase 1 — Archive Q&A

The "almost like an LLM" database questions need no LLM at all: FTS5 full-text
search answers "did user X ever say Y?" exactly and instantly. Ship `~said`,
`~quote`, `~firstseen`, `~chatstats` as ordinary commands. This phase also
shakes out parser bugs before personas consume the same data.

## Phase 2 — Persona cards

An offline batch job (`scripts/build_persona.py <user>`) that distills a
user's history into two reusable **persona cards** (JSON + a human-readable
markdown render):

1. **Stats block — pure counting, no AI.** Message count and date range; top
   non-stopword vocabulary; top emotes; average/median message length;
   capitalization and punctuation habits; busiest hours; favorite targets of
   @mentions; catchphrases (high-frequency n-grams that are rare in everyone
   else's chat — TF-IDF against the rest of the archive).
2. **Exemplar sample.** ~150–300 messages: the most-characteristic (containing
   their signature n-grams) plus a random spread across time so the card isn't
   hostage to one loud week.
3. **LLM distillation.** Feed stats + exemplars to a model with two prompts:
   - *Accurate card:* register and tone, sentence shapes, typical topics,
     emote usage patterns, things they never do (e.g. never uses punctuation,
     never types more than 6 words) — written as **instructions for an
     impersonator**, with verbatim example lines.
   - *Hyperbolic card:* same inputs, instructed to caricature — amplify the
     three most recognizable tendencies to absurdity while staying recognizably
     *them*. (The comedy card.)

Cards are cached at `data/unsynced/personas/<user>.json` (gitignored — this is
personal data about real people and never ships). Regenerate on demand;
include the source-message count and build date inside the card so staleness
is visible. Quality lever: the *exemplar selection* matters more than the
prompt — invest there first when a persona feels off.

## Phase 3 — Runtime roleplay

Two entry points, one engine:

- **Command:** `~as <user> [topic]` — the bot answers in persona.
- **Random reaction:** every eligible message rolls `reaction_chance`
  (default **1/1000**); on a hit, the bot picks a persona (the message's
  author, or random among built cards) and reacts *to the recent
  conversation*, not just the trigger line.

Engine: persona card (system prompt) + the last ~15 archive rows from that
channel (context) + the trigger → one short completion → channel send.

Guardrails (all config, `[persona]` section in `config.toml`):

- `enabled` master switch; per-channel allowlist (default: off everywhere).
- Cooldown (e.g. ≥10 min between random reactions per channel) on top of the
  roll, so an active chat can't get spammy even at 1/1000.
- Never roll on: commands, the bot's own messages, users without a built card,
  users on the opt-out list.
- **Output is always visibly the bot.** Format: `🎭 <user>-bot: <text>` — fun
  dies the moment someone thinks real messages are being faked.
- Length cap (~200 chars) and the existing send-failure logging.

Cost at hobby scale: a reaction is one ~1.5k-token-in / ~100-token-out call.
On a cheap fast model (Claude Haiku 4.5: $1/M in, $5/M out) that's
**~$0.002/call — even 10/day is ~$0.60/month.** Cost is not a factor.

## Phase 4 — Playful psychometrics

For the psychology-nerd itch: a per-user profile in the Big-5 *style* —
guesstimated trait sliders with a one-line justification each, plus fun
derived stats (positivity, question-asking rate, night-owl score, emote
dependency). Two honest constraints, stated up front in the output:

1. **This is entertainment, not assessment.** Chat behavior in a comedy-adjacent
   Twitch chat is a performance, not a clinical sample; real Big-5 inventories
   are validated questionnaires, not text vibes. Label every profile
   `for-fun guesstimate`.
2. Method: blend *lexical signals* computed locally (pronoun rates,
   positive/negative word ratios, exclamation/caps density, social vs solo
   topics — the LIWC-style tradition, which does have real research behind it)
   with an LLM read of the exemplar sample. Cache next to the persona card.

Delivery: `~psyche <user>` whispers the requester by default; posting publicly
requires the profiled user to have opted in (see consent below).

## Phase 5 — Trivia

- **Who-said-it:** the archive is a perfect quote bank. Bot posts a real
  (anonymized) message from a channel regular; chat guesses the author;
  first correct answer wins the point. Needs only Phase 0 + a scoreboard table.
- **Film trivia / Letterboxd:** see [IDEA_BANK.md](IDEA_BANK.md) — the
  official API is closed, but per-user RSS feeds are sanctioned and verified
  working, and TMDB's free API covers film metadata for trivia.

## LLM backend: hosted vs local ("do I need an AI GPU?")

**You don't need a GPU for any of this.** Recommended setup: a hosted
cheap-fast model behind a tiny adapter (`services/llm.py`, mirroring how
`services/translators.py` abstracts DeepL/Google), so the backend is swappable.

- **Hosted (recommended start):** Haiku-class models cost ~$0.002 per persona
  reaction (math above). Persona *builds* are bigger (one ~50–100k-token job
  per user) but one-time — a full friend-group's personas is cents to a few
  dollars. Quality for "imitate this writing style" is meaningfully better
  than small local models.
- **Local (the tinkering path):** an 8 GB-VRAM card runs 7–9B instruct models
  at Q4 quantization (Qwen3-8B, Llama 3.1 8B, Mistral 7B) via llama.cpp /
  Ollama — on AMD cards that means the Vulkan backend (no CUDA), which works
  but is the less-paved road. 12–16 GB unlocks 12–14B models. Verdict: fine
  for experiments and total privacy, noticeably weaker at voice-mimicry than
  hosted frontier-cheap tiers. Don't buy hardware for this project; if you buy
  one anyway for general AI tinkering, 16 GB+ is the comfortable tier.
- **No-LLM fallback that's still funny:** a per-user Markov chain generator
  (order-2 word chains from their messages) is zero-cost, fully local, and
  produces surreal-but-recognizable output — a good Phase-2.5 toy and a
  fallback when the API is down.

## Consent & privacy (read before building Phase 2+)

Personas and psychometrics are **profiles of real people built from things
they typed**, even if publicly visible at the time. House rules baked into the
design, not left to discretion:

- Persona data and the archive never leave the machine (`data/unsynced/`,
  gitignored). The public repo gets code and docs only.
- Build personas only for your friend group / regulars who'd be in on the
  joke; `~optout` permanently excludes a user (delete card + skip in rolls).
- Psychometrics are whisper-first and opt-in for public display.
- Random reactions are off by default and enabled per channel by a
  super-admin — the streamer should know their chat has a roleplay gremlin.
- Bot output is always marked as bot output (`🎭` prefix).

## Open questions (decide when building)

- Reaction persona choice: always the triggering author, or weighted random
  among all cards? (Author-reactive is funnier; random is more chaotic.)
- Should live capture store *all* joined channels or only allowlisted ones?
  (Storage is cheap; consent is the real question.)
- Persona refresh cadence — manual rebuild vs auto-rebuild every N new
  messages.
- Whether Phase 4 lexical metrics deserve their own `~chatstats` expansion
  even without the LLM layer.
