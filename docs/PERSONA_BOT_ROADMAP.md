# Persona bot — roadmap

The fun-features arc for NoLifeChatter: turn the chat archive into per-user
**personas** the bot can roleplay as — an *accurate* one that genuinely
emulates a chatter's tendencies, vocabulary and rhythms, and a *hyperbolic*
one tuned for comedy — plus rare spontaneous in-character reactions, playful
psychometrics, and trivia.

**Status: phases 0–1 built** (the archive and its query commands are live);
phases 2–5 are roadmap. Companion docs: [CHAT_ARCHIVE.md](CHAT_ARCHIVE.md)
(the data layer), [IDEA_BANK.md](IDEA_BANK.md) (smaller ideas, parked things).

## Feasibility verdict

Very feasible. Nothing here is research-grade; it's plumbing plus careful
prompting. The bot framework (command auto-discovery, SQLite, async send paths)
already exists, which removes most of the usual startup cost. Honest effort
estimates, assuming evenings/weekends:

| Phase | What | Effort | Needs LLM? |
| --- | --- | --- | --- |
| 0 | Chat archive (ingest + live capture) | **done** | no |
| 1 | Archive Q&A commands (`~said`, `~quote`, stats) | **done** | no |
| 2 | Persona cards (accurate + hyperbolic, offline batch) | **done (v1)** | yes, offline |
| 3 | Runtime roleplay (`~persona`/`~hyper`, random reactions) | **done (v1)** | yes, local |
| 3.5 | Fine-tuned voice model (one model, all personas) | a weekend + ~$5–20/run | training |
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

## Phase 2 — Persona cards (exemplars first, description last)

**A persona is not a prose description.** A paragraph summary fed to a model
produces generic text wearing a costume — it cannot carry someone's emotes,
casing, rhythm, or vocabulary. The voice lives in their *actual messages*, so
the card's real payload is a curated **exemplar bank**, and the model's job at
runtime is continuation, not acting on a character sheet.

An offline batch job (`scripts/build_persona.py <user>`) produces, per user:

1. **Exemplar bank — the heart of it.** 200–400 of their real messages,
   selected three ways: the most-characteristic (containing their signature
   n-grams — high-frequency phrases rare in everyone else's chat, TF-IDF
   against the rest of the archive); a random spread across time so the bank
   isn't hostage to one loud week; and topic-tagged clusters so runtime can
   *retrieve* their real messages about whatever chat is currently discussing.
   Stored as message ids — the bank re-materializes from the archive, always
   verbatim. **Each exemplar carries its lead-in:** the message *plus the 2–3
   lines it was replying to* (via `chat_archive.context_before`), because a
   line like "exactly" or "I am black" is meaningless — or misleading — without
   what it answered. The persona learns *what they say in what situation*, not
   just a bag of their words.
2. **Stats block — pure counting, no AI.** Top emotes with frequencies,
   message-length distribution, capitalization/punctuation habits, busiest
   hours, favorite @targets. Used to *validate* output (does the generated
   line use emotes they actually use?) and to weight exemplar selection.
3. **Voice spec — small, LLM-written, scaffolding only.** The few hard rules
   examples can't show by themselves: "never capitalizes, no punctuation,
   1–6 words, types X when excited, never asks questions." Plus the
   *hyperbolic* variant: the same rules with their three most recognizable
   tendencies amplified to absurdity (the comedy card). This is ~5% of the
   runtime prompt — the exemplars are the persona.

Cards are cached at `data/unsynced/personas/<user>.json` (gitignored — this is
personal data about real people and never ships), with source-message count
and build date inside so staleness is visible. New chat accumulates in the
archive automatically; rebuilding the card re-selects exemplars from the
up-to-date corpus — that's the whole "top-up" step.

## Phase 3 — Runtime roleplay

Two entry points, one engine:

- **Command:** `~as <user> [topic]` — the bot answers in persona.
- **Random reaction:** every eligible message rolls `reaction_chance`
  (default **1/1000**); on a hit, the bot picks a persona (the message's
  author, or random among built cards) and reacts *to the recent
  conversation*, not just the trigger line.

Engine — **many-shot voice cloning**, not description-driven roleplay. The
prompt is dominated by the person's verbatim messages:

```
voice spec (~5%)                      "never capitalizes, 1–6 words, ..."
exemplar messages (~80%)              200+ real lines: their signature
                                      phrases + lines retrieved from the
                                      archive about the CURRENT topic
live conversation (~15%)              the last ~15 messages in the channel
task                                  "write the next message <user> would
                                      type here"
```

With hundreds of their real lines literally in front of it, a strong model
reproduces the emotes, casing, and catchphrases because they're *in the
context*, not described. Modern context windows make this trivial — 300 chat
messages ≈ 4–6k tokens. This is the standard way to get voice mimicry without
training a model, and it should be built first because Phase 3.5 reuses every
piece of it.

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

## Phase 3.5 — Fine-tuning: the "train once, top up later" upgrade

If many-shot isn't uncanny enough, the next rung is a model whose *weights*
have absorbed the chat — trained on `(recent conversation → the user's actual
next message)` pairs straight from the archive.

- **One model serves every persona — not one per person.** Each training row
  is prefixed with the speaker (`<persona=user>`), and generation picks the
  persona by prompting that prefix. Twenty personas is still one model and one
  training run; adding a person is adding rows, not models.
- **Two routes:**
  - *Hosted fine-tune* (e.g. OpenAI's mini models — self-serve): upload the
    pairs, train for a few dollars at friend-group data sizes, call it like
    any API. Easiest. (Anthropic has no self-serve fine-tuning.)
  - *LoRA on an open 7–9B model* in a rented cloud GPU (a few hours,
    ~$5–15/run), then quantize to GGUF and run **inference locally for free**
    — an 8 GB consumer card can run what it cannot train. Most private; most
    tinkering.
- **Top-ups** are exactly what they sound like: re-run the job every few
  months with the new rows the archive accumulated (`sent_at` makes selecting
  "everything since last run" trivial).
- **Expectation-setting:** fine-tuned small models are the *uncanny* option —
  eerily voice-accurate, noticeably dumber, occasionally incoherent (the
  famous trained-on-our-group-chat effect — which is often the funnier
  failure mode). Many-shot on a frontier model is more coherent, slightly
  less uncanny. Build 3 first, A/B them, keep whichever makes the group laugh
  harder.

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
- **Local (the tinkering path, and the answer for *edgy* content):** hosted
  Claude/OpenAI **refuse** to generate slurs or speak in the voice of someone
  who uses them, and OpenAI fine-tuning **rejects** such training data at two
  gates (training-file moderation + a post-training safety eval). So for
  uncensored persona content, a local open model is the route — no refusals,
  no provider account risk, no data leaving the machine, free per call.
  - On this machine's **AMD RX 5700 XT (8 GB, RDNA1)**: ROCm does **not**
    support this card on Windows — use the **Vulkan** backend (standard
    Adrenalin driver). Shortest path is **LM Studio** (pick the Vulkan
    runtime); raw `llama.cpp` Vulkan build is the fast-but-manual alt; Ollama
    works only via its newer Vulkan path on RDNA1 (set `OLLAMA_VULKAN=1`, or
    use the `ollama-for-amd` fork). **Leave Flash Attention OFF on RDNA1** (it
    ~3× slows generation here).
  - Models that fit 8 GB at Q4_K_M and won't refuse (abliterated / dolphin):
    **Llama-3.1-8B-Instruct-abliterated** (~4.9 GB, strong default),
    **dolphin-2.9.4-llama3.1-8b**, **Mistral-7B-Instruct-v0.3-abliterated**
    (smallest/fastest), **Qwen2.5-7B-Instruct-abliterated**. Expect ~25–45
    tok/s on Vulkan — comfortably past reading speed. Keep context 4–8k so the
    model + KV cache fit in 8 GB.
  - **Less-restricted hosted** middle ground (no local setup): OpenRouter hosts
    uncensored models (e.g. Dolphin-Mistral-24B "Venice", a free rate-limited
    tier exists; paid uncensored ~$0.25–$1 / M tokens). Generates freely — but
    see the Twitch note below; generating is not the hard part, posting is.
  - Verdict: local is fine for experiments + total privacy + edgy content,
    a bit weaker at mimicry than frontier-cheap hosted. Don't buy hardware for
    this; the 5700 XT already works via Vulkan.
- **No-LLM fallback that's still funny — BUILT:** `utils/persona_markov.py`
  (order-N word chains from a user's archived messages) +
  `scripts/persona_preview.py <user>` (terminal-only preview, posts nothing).
  Zero-cost, fully local, no provider content policy in play. Surreal but
  unmistakably them. It does *not* use conversation context — that's the LLM
  version's job — but it's the always-available, TOS-free warm-up and fallback.

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
- **Twitch output is the real constraint, separate from any model's policy.**
  Posting slurs/hate to Twitch is zero-tolerance and bans the bot *and* the
  operator's main account — there's no compliant channel (chat, whisper, or
  private) for it. The durable design is **generate → filter → maybe-send**: a
  denylist/classifier gates every message before it's posted (built today:
  `utils/output_filter.py`, gitignored blocklist, applied by `~mimic`). "Edgy"
  survives the filter; slurs don't. Cleanest of all is to redact slurs from the
  source text *before* it reaches the model, so personas carry the cadence and
  humor without the bannable words.

## Open questions (decide when building)

- Reaction persona choice: always the triggering author, or weighted random
  among all cards? (Author-reactive is funnier; random is more chaotic.)
- Should live capture store *all* joined channels or only allowlisted ones?
  (Storage is cheap; consent is the real question.)
- Persona refresh cadence — manual rebuild vs auto-rebuild every N new
  messages.
- Whether Phase 4 lexical metrics deserve their own `~chatstats` expansion
  even without the LLM layer.
