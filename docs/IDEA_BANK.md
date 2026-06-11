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
