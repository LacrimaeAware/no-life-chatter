"""Per-user command bans: banned chatters are ignored by the dispatcher.

Super-admin only (~banuser / ~unbanuser). Stored in the settings DB so bans
survive restarts; cached in-process so the per-message check is free. The ban
silences COMMANDS only — translation and archiving of their messages are
unaffected.
"""

import sqlite3

import config

_cache = None


def _conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS banned_users ("
                 " user_name TEXT PRIMARY KEY, banned_at TEXT)")
    return conn


def _load():
    global _cache
    if _cache is None:
        with _conn() as conn:
            _cache = {r[0] for r in conn.execute("SELECT user_name FROM banned_users")}
    return _cache


def is_banned(user: str) -> bool:
    return bool(user) and user.lower().lstrip("@") in _load()


def ban(user: str) -> None:
    user = user.lower().lstrip("@")
    with _conn() as conn:
        conn.execute("INSERT OR REPLACE INTO banned_users VALUES (?, datetime('now'))",
                     (user,))
    _load().add(user)


def unban(user: str) -> bool:
    user = user.lower().lstrip("@")
    with _conn() as conn:
        cur = conn.execute("DELETE FROM banned_users WHERE user_name = ?", (user,))
    _load().discard(user)
    return cur.rowcount > 0


def banned_list() -> list:
    return sorted(_load())
