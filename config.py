"""Central configuration for NoLifeChatter.

All user-specific settings (which channels to join, who the admins are, where
data lives) come from ``config.toml``.  All secrets (API keys, client secret)
come from the environment / ``.env``.  Nothing personal is hard-coded anywhere
else in the project, so the bot is safe to publish and easy for someone else to
run against their own Twitch account.

Copy ``config.example.toml`` -> ``config.toml`` and ``.env.example`` -> ``.env``
and fill them in before running the bot.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load secrets from .env (if present) so os.getenv works everywhere.
load_dotenv(BASE_DIR / ".env")

CONFIG_PATH = Path(os.getenv("NLC_CONFIG", BASE_DIR / "config.toml"))


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found at '{CONFIG_PATH}'.\n"
            "Copy 'config.example.toml' to 'config.toml' and edit it before "
            "running the bot."
        )
    with open(CONFIG_PATH, "rb") as fh:
        return tomllib.load(fh)


_cfg = _load_config()


def _resolve(path_str: str) -> str:
    """Resolve a path from config relative to the project root."""
    path = Path(path_str)
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


# ----------------------------- bot identity -----------------------------
_bot = _cfg.get("bot", {})
PREFIX: str = _bot.get("prefix", "~")
CHANNELS: list[str] = [c.lower() for c in _bot.get("channels", [])]
# Optional message posted on connect ("" disables it).
READY_MESSAGE: str = (_bot.get("ready_message") or "").strip()
# If set, only post the ready message to this one channel; "" = all channels.
READY_CHANNEL: str = (_bot.get("ready_channel") or "").strip().lower()

if not CHANNELS:
    raise ValueError(
        "No channels configured. Set bot.channels in config.toml to at least "
        "one Twitch channel for the bot to join."
    )

# ------------------------------- authorization ---------------------------
_auth = _cfg.get("auth", {})
# Admins: may toggle their own translation, language and output mode.
ADMINS: list[str] = [u.lower() for u in _auth.get("admins", [])]
# Super admins: may also flip global / per-channel translation switches.
SUPER_ADMINS: list[str] = [u.lower() for u in _auth.get("super_admins", [])]

# ---------------------------------- paths --------------------------------
_paths = _cfg.get("paths", {})
DB_PATH: str = _resolve(_paths.get("database", "data/synced/bot_settings.db"))
TOKEN_FILE: str = _resolve(_paths.get("token_file", "data/unsynced/token_data.json"))
# GOOGLE_APPLICATION_CREDENTIALS env var wins if set; otherwise use config.
GOOGLE_CREDENTIALS: str = _resolve(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    or _paths.get("google_credentials", "data/unsynced/google-service-account.json")
)

# ------------------------------- translation -----------------------------
_tr = _cfg.get("translation", {})
DEFAULT_TARGET: str = _tr.get("default_target", "EN").upper()
# lingua confidence scale (0..1). Used by practice mode's single-guess gate.
# (Channel auto-translate uses the share-based gate below instead, because
# lingua's absolute scores are miscalibrated across languages.)
MIN_CONFIDENCE: float = float(_tr.get("min_confidence", 0.4))
# Minimum word count before channel auto-translate considers a message (short
# messages misdetect). Non-Latin-script messages, and known speakers, bypass it.
MIN_WORDS: int = int(_tr.get("min_words", 4))
# Channel auto-translate gate. The decision is about the *distribution*, not an
# absolute score: a message is "confidently not the target language" when the
# best foreign guess wins the head-to-head share against the target by at least
# MIN_FOREIGN_SHARE. (best / (best + target) >= share.) This translates clear
# Spanish/etc. that never reaches a high absolute score, while still skipping
# English/junk where the foreign guess only edges English out. 0.63 cleanly
# separates the two on a labelled test set. MIN_FOREIGN_SIGNAL is a tiny floor
# so near-uniform noise (everything ~0.05, share meaningless) is still skipped.
MIN_FOREIGN_SHARE: float = float(_tr.get("min_foreign_share", 0.63))
MIN_FOREIGN_SIGNAL: float = float(_tr.get("min_foreign_signal", 0.10))
# Short Latin-script messages (< MIN_WORDS words) are unreliable to detect AND
# to translate — fragments and leftover emote names get "translated" into
# nonsense. So a short message is only translated when a single language is
# detected with at least this absolute confidence ("avoid short phrases unless
# very sure"). Real short foreign like "danke schön" (0.68) or "buongiorno"
# (0.95) clears it; "nah fam"/"ge"/emote junk (~0.25) does not.
MIN_SHORT_CONFIDENCE: float = float(_tr.get("min_short_confidence", 0.55))

# --------------------------------- comedy --------------------------------
# Default chats for the ~funny comedy-influence ranking. Should be the
# conversational chats (not big stream chats where laughter reacts to the
# stream). Empty falls back to all joined channels.
_co = _cfg.get("comedy", {})
COMEDY_CHANNELS: list[str] = [c.lower() for c in _co.get("channels", [])]

# ------------------------------ speaker profiles -------------------------
# The bot learns which languages each user writes in, to translate them more
# reliably (and to avoid translating users who only ever write English).
_sp = _cfg.get("speaker", {})
# A user becomes a "known speaker" of a language only after this many LONG
# (>= long_words) confidently-foreign sentences in it — a deliberately HIGH bar
# so garbage/short misdetections can't flag someone (they used to at count=3).
SPEAKER_MIN_COUNT: int = int(_sp.get("min_count", 20))
# A "massive" speaker (many long sentences) — the only tier that unlocks
# single-word auto-translation.
SPEAKER_MASSIVE_COUNT: int = int(_sp.get("massive_count", 50))
# Only sentences with at least this many words count toward speaker flagging.
SPEAKER_LONG_WORDS: int = int(_sp.get("long_words", 10))
SUPPORTED_LANGS: set[str] = {
    lang.upper()
    for lang in _tr.get(
        "supported",
        [
            "EL", "ZH-CN", "ES", "FR", "AR", "RU", "HU", "TR", "KO", "JA",
            "VI", "DE", "IT", "PT", "PL", "CS", "SK", "UK", "LA", "EN",
        ],
    )
}

# ------------------------------- chat archive ----------------------------
# Searchable archive of chat messages (docs/CHAT_ARCHIVE.md): historical
# Chatterino logs ingested by scripts/ingest_chatterino.py plus, when
# live_capture is on, every message the bot sees. Lives in a gitignored dir.
_ar = _cfg.get("archive", {})
ARCHIVE_DB: str = _resolve(_ar.get("database", "data/unsynced/chat_archive.db"))
ARCHIVE_LIVE: bool = bool(_ar.get("live_capture", True))
# Default Chatterino logs root for the ingest script (so you don't retype it).
ARCHIVE_CHATTERINO_LOGS: str = _ar.get("chatterino_logs", "")
# Spelling variants that should count as the same channel/user, e.g.
# {"typo_name" = "real_name"} merges a typo-named channel into the real one.
ARCHIVE_ALIASES: dict = dict(_ar.get("aliases", {}))
# More precise aliases. These are layered on top of archive.aliases, but only
# affect the relevant side so alt accounts do not accidentally rename channels.
ARCHIVE_USER_ALIASES: dict = dict(_ar.get("user_aliases", {}))
ARCHIVE_CHANNEL_ALIASES: dict = dict(_ar.get("channel_aliases", {}))

# ------------------------------- personas --------------------------------
# Persona features (docs/PERSONA_BOT_ROADMAP.md). The ~markov/~mimic commands
# post a Markov-generated line in a chatter's style; output is run through the
# blocklist below first so the bot never posts a bannable line to Twitch.
_pe = _cfg.get("persona", {})
MIMIC_ENABLED: bool = bool(_pe.get("mimic_enabled", True))
# Chance (0..1) that any given chat message triggers a random persona reaction.
# Ambient reactions use the LLM persona engine; Markov is explicit-command only.
# e.g. 0.002 ~= 1 in 500 messages. A cooldown stops it bunching up.
REACTION_CHANCE: float = float(_pe.get("reaction_chance", 0.0))
REACTION_DIRECTED_CHANCE: float = float(_pe.get("reaction_directed_chance", 0.0))
REACTION_COOLDOWN: float = float(_pe.get("reaction_cooldown", 90))
REACTION_CONTINUE_CHANCE: float = float(_pe.get("reaction_continue_chance", 0.0))
REACTION_MAX_CONTINUATIONS: int = int(_pe.get("reaction_max_continuations", 1))
REACTION_CONTINUE_DELAY: float = float(_pe.get("reaction_continue_delay", 1.5))
PERSONA_COMMAND_CONTINUE_CHANCE: float = float(_pe.get("command_continue_chance", 0.0))
PERSONA_COMMAND_MAX_CONTINUATIONS: int = int(_pe.get("command_max_continuations", 1))
PERSONA_COMMAND_CONTINUE_DELAY: float = float(_pe.get("command_continue_delay", 1.5))
# Usernames random reactions should never mimic (command/stats bots produce
# junk). Explicit ~markov/~mimic still works on them if you really ask.
EXCLUDE_USERS: set[str] = {
    u.lower() for u in _pe.get("exclude_users", [
        "streamelements", "nightbot", "fossabot", "moobot", "wizebot",
        "soundalerts", "streamlabs", "pokemoncommunitygame", "potatbotat",
        "buttsbot", "supibot", "kunszg",
    ])
}
_llm = _cfg.get("llm", {})
# Denylist of terms the bot must never post (one per line, '#' comments).
# Kept OUT of the repo — lives in a gitignored file. Empty/missing = no filter.
BLOCKLIST_FILE: str = _resolve(
    _pe.get("blocklist_file", _llm.get("blocklist_file", "data/unsynced/blocklist.txt"))
)

# LLM persona engine (~persona / ~hyper, and optionally the random reaction).
# Points at any OpenAI-compatible chat endpoint — LM Studio's local server by
# default (free, local, private). Leave it; just run LM Studio with a model.
LLM_ENDPOINT: str = _llm.get("endpoint", "http://127.0.0.1:1234/v1/chat/completions")
LLM_MODEL: str = _llm.get("model", "local")  # LM Studio uses whatever's loaded
# Live A/B: when non-empty, every persona generation randomly rolls one of
# these model ids and the posted line is tagged (#llama / #lora) so chat can
# judge them side by side. Empty list = always LLM_MODEL.
LLM_AB_MODELS: list = list(_llm.get("ab_models", []))
LLM_EMBED_MODEL: str = str(_llm.get("embed_model", "")).strip()
# Use the per-message embedding index (scripts/build_message_index.py) to add
# meaning-based retrieval to persona evidence. Needs the index built.
LLM_SEMANTIC_RETRIEVAL: bool = bool(_llm.get("semantic_retrieval", False))
# Semantic retrieval is powerful but can hallucinate "nearby" chat concepts.
# Anchored hits share at least one cleaned query term; unanchored hits must be
# much more confident before they enter a persona prompt.
LLM_SEMANTIC_MIN_SCORE: float = float(_llm.get("semantic_min_score", 0.50))
LLM_SEMANTIC_UNANCHORED_MIN_SCORE: float = float(
    _llm.get("semantic_unanchored_min_score", 0.62)
)
# Short names usable as ~persona's model= override (model=lora -> real id).
LLM_MODEL_SHORTCUTS: dict = {
    str(k).lower(): str(v) for k, v in _llm.get("model_shortcuts", {}).items()
}
LLM_TIMEOUT: float = float(_llm.get("timeout", 90))
LLM_EXEMPLARS: int = int(_llm.get("exemplars", 150))   # real lines put in the prompt
LLM_RELEVANT_EXEMPLARS: int = int(
    _llm.get("relevant_exemplars", min(90, max(0, int(LLM_EXEMPLARS * 0.6))))
)
LLM_CONTEXT: int = int(_llm.get("context_messages", 25))  # recent chat lines for context
LLM_RETRY_EXEMPLARS: int = int(_llm.get("retry_exemplars", min(60, LLM_EXEMPLARS)))
LLM_RETRY_CONTEXT: int = int(_llm.get("retry_context_messages", min(12, LLM_CONTEXT)))
# Top retrieval hits expanded into ±2-line "chat moment" snippets — evidence of
# how the author RESPONDS, not just their vocabulary. Each costs ~5 lines of
# the relevant budget. 0 disables snippets.
LLM_SNIPPET_HITS: int = int(_llm.get("snippet_hits", 8))
# Samples per reply; the best valid one is posted (copies/URLs/echoes of other
# chatters' lines are rejected). The big prompt is server-cached, so extra
# samples are much cheaper than the first.
LLM_CANDIDATES: int = int(_llm.get("candidates", 2))
# Private JSONL log of every persona generation: evidence fed, all candidates
# with rejection reasons, final output, timing. Lives in a gitignored dir.
PERSONA_LOG: bool = bool(_pe.get("log_enabled", True))
PERSONA_LOG_FILE: str = _resolve(_pe.get("log_file", "data/unsynced/persona_logs.jsonl"))
# Authorship classifier (powers ~whosaid + persona eval). Trained offline by
# scripts/train_classifier.py; model file is gitignored.
CLASSIFIER_FILE: str = _resolve(_pe.get("classifier_file", "data/unsynced/persona_classifier.pkl"))
# Ambient random reactions are LLM-only. Markov stays behind ~mimic/~markov.
REACTION_USE_LLM: bool = bool(_pe.get("reaction_use_llm", True))

# --------------------------------- secrets --------------------------------
# Read straight from the environment so they never live in tracked files.
TWITCH_CLIENT_ID: str | None = os.getenv("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET: str | None = os.getenv("TWITCH_CLIENT_SECRET")
DEEPL_API_KEY: str | None = os.getenv("DEEPL_API_KEY")
