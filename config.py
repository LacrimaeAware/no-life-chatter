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
# lingua confidence scale (0..1). ~0.40 separates real foreign sentences from
# short English/junk. (This is a different scale than the old langdetect 0.7.)
MIN_CONFIDENCE: float = float(_tr.get("min_confidence", 0.4))
# Minimum word count before channel auto-translate considers a message (short
# messages misdetect). Non-Latin-script messages bypass this.
MIN_WORDS: int = int(_tr.get("min_words", 4))
# How far the best foreign-language score must beat the target language's own
# score for a message to count as "confidently not the target language". Guards
# against ambiguous messages where several languages (incl. English) are close.
MIN_MARGIN: float = float(_tr.get("min_margin", 0.15))

# ------------------------------ speaker profiles -------------------------
# The bot learns which languages each user writes in, to translate them more
# reliably (and to avoid translating users who only ever write English).
_sp = _cfg.get("speaker", {})
# Confident messages in a language before a user is a "known speaker" of it.
# A simple count threshold — once it's reached, the language flips on for them.
SPEAKER_MIN_COUNT: int = int(_sp.get("min_count", 3))
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

# --------------------------------- secrets --------------------------------
# Read straight from the environment so they never live in tracked files.
TWITCH_CLIENT_ID: str | None = os.getenv("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET: str | None = os.getenv("TWITCH_CLIENT_SECRET")
DEEPL_API_KEY: str | None = os.getenv("DEEPL_API_KEY")
