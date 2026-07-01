"""Users whose messages are never auto-translated (super-admin managed).

Some chatters write English slang/abbreviations that the language detector
misreads as foreign ("kk", "lmaoq", "oh in hs bg"). A super admin can opt those
users out of auto-translation with ~notranslate; this stores the list.
"""
import sqlite3

import config

_cache = None


def _conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS autotranslate_optout "
                 "(user_name TEXT PRIMARY KEY)")
    return conn


def _load():
    global _cache
    with _conn() as conn:
        _cache = {r[0] for r in conn.execute("SELECT user_name FROM autotranslate_optout")}
    return _cache


def is_opted_out(name: str) -> bool:
    if _cache is None:
        _load()
    return (name or "").lower() in _cache


def set_opt_out(name: str, off: bool = True) -> bool:
    """off=True stops auto-translating them; off=False re-enables. Returns off."""
    name = (name or "").lower()
    with _conn() as conn:
        if off:
            conn.execute("INSERT OR IGNORE INTO autotranslate_optout VALUES (?)", (name,))
        else:
            conn.execute("DELETE FROM autotranslate_optout WHERE user_name=?", (name,))
    _load()
    return off


def list_opted_out() -> list[str]:
    if _cache is None:
        _load()
    return sorted(_cache)
