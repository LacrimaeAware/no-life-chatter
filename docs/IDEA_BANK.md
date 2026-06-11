# Idea bank

Running list of fun-feature ideas — researched enough to act on later, parked
so they don't get lost. Big items graduate to their own doc (see
[PERSONA_BOT_ROADMAP.md](PERSONA_BOT_ROADMAP.md)).

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

## Trivia & games

- **Who-said-it** — post a real archive quote, chat guesses the author,
  scoreboard in SQLite. The single highest fun-per-effort feature once the
  archive exists. (Roadmap Phase 5.)
- **Film trivia** — TMDB free API: "name the film from cast + year", poster
  blur-up reveals, etc. Pairs with the Letterboxd item.
- **Chat Wrapped** — yearly per-user recap from the archive: messages sent,
  top emotes, most-pinged friend, busiest day, signature phrase. Spotify-
  Wrapped energy, pure SQL.
- **First-seen anniversaries** — bot quietly notes "5,000 days… ok, 2 years
  since <user>'s first recorded message" style milestones.

## Personas — graduated

Accurate + hyperbolic per-user personas, rare random in-character reactions
(1/1000 roll, with conversation context), playful Big-5-flavored
psychometrics, and the chat-archive Q&A layer ("did X ever say Y?") all live
in [PERSONA_BOT_ROADMAP.md](PERSONA_BOT_ROADMAP.md), with the data layer in
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
