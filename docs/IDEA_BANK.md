# Idea bank

Running list of fun-feature ideas - researched enough to act on later, parked
so they do not get lost. Current priorities live in [ROADMAP.md](ROADMAP.md);
older persona roadmap history is archived at
[archive/PERSONA_BOT_ROADMAP.md](archive/PERSONA_BOT_ROADMAP.md).

## Letterboxd integration — parked, viable via RSS

Wanted: `~lb <user>` showing recent films/ratings, film-club features, etc.

Research findings (verified June 2026):

- **The official API is effectively unavailable** for hobby projects: still
  request-only beta (email `api@letterboxd.com`), no self-serve signup, hobby
  requests reportedly sit unanswered for months. Don't wait on it.
- **Scraping the site is explicitly against ToS** ("any robot, spider,
  scraper… to access, acquire, copy, or monitor any portion of the Service" —
  letterboxd.com/legal/terms-of-use). Don't.
- **The sanctioned path exists: per-member RSS feeds.**
  `https://letterboxd.com/<username>/rss/` is documented in Letterboxd's own
  FAQ and verified live: returns the ~50 most recent diary entries with film
  title + year, member rating, watched date, rewatch/like flags, poster, the
  review text, and a **TMDB movie id**. Two caveats: it only exposes *recent*
  activity (no full history, no watchlist), and the endpoint 403s generic
  bot User-Agents — send a browser-like UA.
- The included TMDB id links cleanly into **TMDB's free API** for posters,
  cast, and metadata — which also powers film trivia on its own.

Sketch when picked up: `services/letterboxd.py` (RSS fetch + parse, cached
~15 min), `~lb <user>` command, optional "new diary entry" announcements for
registered users. LLM not required.

## "Real or AI?" — Turing-test game (wanted, high priority)

A guessing game, and the chat literally invented it live (earnest: *"ok turing
test, was that a message i wrote or ai generated"*). Command e.g.
`~realorai [user]`:

1. Pick a chatter (named, or random recent regular).
2. Coin-flip: either **pull a real line** from their archive, or **generate** a
   persona line (`persona_llm.generate`).
3. Post it as a quoted line; chat guesses "real" or "AI"; first/most correct
   wins; then reveal.

The whole game is in the **heuristics that pick the line** — a boring quote
ruins it:
- **Skip** single emotes, 1-word lines, pure links/mentions, and ultra-short
  fragments (no fun to guess).
- **Skip** very long / hyper-specific lines (too identifiable, or obviously
  generated).
- **Prefer** lines that "make a statement" — declarative, has a verb, a real
  opinion or claim (`re` heuristic on the real pull; instruct the generator the
  same way). "Legavish is stating a fact"-energy, not a stray reaction.
- Keep a scoreboard table (SQLite) for points.

Both halves reuse what's built: real pull = `chat_archive` query with the
length/shape filter; AI half = the persona engine. Needs only the picker
heuristics + a scoreboard.

## Organic reply-frequency (conversation/reaction refinement — wanted)

Right now reactions fire on a flat per-message probability and `~persona`
replies whenever invoked. A real person in a 5-way conversation doesn't reply to
every line — they answer what's **directed at them** or what continues a thread
they're already in. Make "how often it responds" a real parameter and bias it:

- Detect **directed-at-persona**: the message @-mentions the persona, says their
  name, or is a direct reply in a thread the persona was just in.
- Response = `directed?` (high chance) vs `ambient?` (low chance) + a cooldown,
  instead of uniform random.
- Optional: a lightweight "is this a conversational opening" check so it jumps in
  on questions/hot-takes more than on one-word spam.

This generalizes the current `reaction_chance`; the autonomous "bot hangs out in
chat as personas" mode becomes far more natural.

## Embedding-based voice/topic space (wanted — the next big direction)

Token statistics (log-odds markers, TF-IDF stylometry) top out at "which exact
words does this person overuse" — they can never see that two people talk
about the same THINGS in different words, and language itself (a bilingual
chatter) reads as one giant marker. Embeddings fix both, and the
infrastructure is already running: LM Studio serves a local embedding model
(`/v1/embeddings`, e.g. nomic-embed-text) next to the chat models.

Plan sketch:
1. Embed each chatter's distinctive exemplar lines (the RAG signature sample
   is already computed) -> mean-pool into one vector per person.
2. `~like` v3 = cosine in that space (semantic similarity), shown alongside
   the current shared-marker evidence (lexical similarity). The disagreement
   between the two is itself interesting (same topics, different voice = a
   lurker-twin; same words, different topics = a copycat).
3. Cluster the vectors -> the personality-map idea below (groups emerge from
   geometry instead of hand-built trait lists).
4. Stretch: project onto trait axes (define each end of an axis with example
   sentences, embed them, measure where each chatter falls) — "second-order"
   personality traits separated from first-order topic/word overlap.

## Chat personality maps / psychometrics research (wanted, bigger idea)

Full working note: [CHAT_PERSONALITY_RESEARCH.md](CHAT_PERSONALITY_RESEARCH.md).

Use the archive as a high-volume behavioral corpus to map chatters, cliques,
and latent "personality" dimensions from language. This is partly a fun feature
and partly a possible research/art project: Twitch chat logs as a messy but rich
source for measuring stable social/personality signals over time.

Sketch:
- Build per-user embeddings from message samples, topic distributions, emote
  usage, reply targets, timing, message length, punctuation/caps habits, and
  lexical markers.
- Cluster users into "chat gangs" / social neighborhoods: who talks to whom,
  who shares emotes/phrases, who appears in the same conversational contexts.
- Derive interpretable axes instead of only Big Five imitation: irony density,
  aggression/playfulness, lore-reference rate, emote reliance, question-vs-claim
  style, night-owl/activity rhythms, social centrality, topic breadth, etc.
- Compare stability across time: do dimensions survive across months/years,
  channels, and changing chat metas? This is the interesting "does it replicate"
  question.
- If this ever becomes public/research-facing, get consent from included users
  before publishing logs, examples, or identifiable scores. Use aggregate or
  anonymized outputs by default.

Deliverables could start small: `~psyche <user>` for private/fun metrics,
offline notebooks that render 2D/3D maps of chatters, and a generated report
for each cluster with evidence snippets.

## Trivia & games

- **Who-said-it** — post a real archive quote, chat guesses the author,
  scoreboard in SQLite. The single highest fun-per-effort feature once the
  archive exists. (Roadmap Phase 5.)
- **Emote/lore glossary RAG** — build a private glossary from 7TV/BTTV/FFZ
  emote names plus archive contexts, so archive-Q&A and personas can understand
  local references like emotes, recurring bits, shortened game names (`wow` ->
  World of Warcraft), and phrases that only make sense inside the chat. Start
  with retrieval/evidence, not fine-tuning.
- **Film trivia** — TMDB free API: "name the film from cast + year", poster
  blur-up reveals, etc. Pairs with the Letterboxd item.
- **Chat Wrapped** — yearly per-user recap from the archive: messages sent,
  top emotes, most-pinged friend, busiest day, signature phrase. Spotify-
  Wrapped energy, pure SQL.
- **First-seen anniversaries** — bot quietly notes "5,000 days… ok, 2 years
  since <user>'s first recorded message" style milestones.

## Personas — graduated

Accurate + hyperbolic per-user personas, rare random in-character reactions
(config-driven, with conversation context), playful trait experiments, and the
chat-archive Q&A layer have moved from idea to active system/roadmap. Current
state lives in [ROADMAP.md](ROADMAP.md), with the data layer in
[CHAT_ARCHIVE.md](CHAT_ARCHIVE.md).

- **Markov-chain mini-personas** — order-2 word chains per user; zero-cost,
  fully local, surreally funny. Good warm-up before the LLM version and a
  fallback when the API is down.

## Smaller / someday

- **`~remindme` / timers** — classic utility, trivially stored in SQLite.
- **Emote-usage leaderboards** — per-channel top emotes by week, from the
  archive; zero new infrastructure.
- **Log rotation for `data/bot.log`** — not fun, but the file currently grows
  unbounded; a dated-file handler is ~10 lines. Do during any Phase-0 work.
