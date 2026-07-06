"""Anti-spam command cooldowns — punishes sustained stacking, warns first.

Design:
- The malicious pattern is sending MORE commands while your previous one is
  still processing. Using commands a lot but waiting for each reply is fine.
- First time you stack past the limit you get a WARNING, not a ban — so a
  mistaken double-call or a syntax check never costs you a cooldown. Only if you
  keep stacking (another breach within WARN_GRACE) does a cooldown apply.
- Cooldowns are gentle and CAPPED: BASE doubles per repeat offense but never
  exceeds MAX_COOLDOWN. Strikes are forgiven after a clean DECAY_SECONDS.
- Every real cooldown is recorded to the settings DB (command_warnings) for
  review (~warnings, super admin).
"""

import logging
import sqlite3
import time

import config

BASE_COOLDOWN = 3 * 60          # first real cooldown: 3 minutes
MAX_COOLDOWN = 15 * 60          # escalation is capped here (was unbounded doubling)
STACK_LIMIT = 3                 # this many commands stacked while one is pending = a breach
DECAY_SECONDS = 2 * 3600        # strikes forgiven after a clean 2h (was 24h)
WARN_GRACE = 15 * 60            # breach again within this of a warning -> cooldown

_pending = {}            # user -> count of in-flight commands
_stacked = {}            # user -> commands sent while pending
_cooldown_until = {}     # user -> epoch
_notified = set()        # users already told about their current cooldown
_warned_at = {}          # user -> time we last said "slow down" (no ban)


def _conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS command_warnings ("
                 " id INTEGER PRIMARY KEY, user_name TEXT, at TEXT,"
                 " reason TEXT, cooldown_minutes REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS command_strikes ("
                 " user_name TEXT PRIMARY KEY, strikes INTEGER, last_at REAL)")
    return conn


def _strikes(user):
    with _conn() as conn:
        row = conn.execute("SELECT strikes, last_at FROM command_strikes "
                           "WHERE user_name=?", (user,)).fetchone()
    if not row:
        return 0
    strikes, last_at = row
    if time.time() - (last_at or 0) > DECAY_SECONDS:
        return 0   # clean stretch -> forgiven
    return strikes


def _offense(user):
    strikes = _strikes(user) + 1
    cooldown = min(MAX_COOLDOWN, BASE_COOLDOWN * (2 ** (strikes - 1)))
    with _conn() as conn:
        conn.execute("INSERT OR REPLACE INTO command_strikes VALUES (?,?,?)",
                     (user, strikes, time.time()))
        conn.execute("INSERT INTO command_warnings (user_name, at, reason, cooldown_minutes)"
                     " VALUES (?, datetime('now'), ?, ?)",
                     (user, "kept stacking commands after a warning",
                      round(cooldown / 60, 1)))
    _cooldown_until[user] = time.time() + cooldown
    _notified.discard(user)
    logging.warning(f"command cooldown: {user} for {cooldown/60:.0f}m (strike {strikes})")
    return cooldown


def before(user: str):
    """Call before dispatching. Returns:
    ('ok', None)        — run the command (pending count incremented)
    ('drop', None)      — silently ignore (already notified about cooldown)
    ('warn', None)      — first breach: tell them to slow down, but NO cooldown
    ('cooldown', secs)  — cooldown active/just triggered: notify once
    """
    user = (user or "").lower()
    now = time.time()
    until = _cooldown_until.get(user, 0)
    if now < until:
        if user in _notified:
            return ("drop", None)
        _notified.add(user)
        return ("cooldown", int(until - now))
    if _pending.get(user, 0) > 0:
        _stacked[user] = _stacked.get(user, 0) + 1
        if _stacked[user] >= STACK_LIMIT:
            _stacked[user] = 0
            # First breach (or one long after the last warning) is a WARNING
            # only. A repeat breach within the grace window is a real cooldown.
            if now - _warned_at.get(user, 0) >= WARN_GRACE:
                _warned_at[user] = now
                return ("warn", None)
            secs = int(_offense(user))
            _notified.add(user)
            return ("cooldown", secs)
        # a couple of impatient re-sends are tolerated — they still run
    _pending[user] = _pending.get(user, 0) + 1
    return ("ok", None)


def after(user: str):
    """Call when a command finishes (success or error)."""
    user = (user or "").lower()
    _pending[user] = max(0, _pending.get(user, 0) - 1)
    if _pending[user] == 0:
        _stacked[user] = 0   # they waited; slate cleaned


def warnings_list(limit=10):
    with _conn() as conn:
        return conn.execute(
            "SELECT user_name, at, cooldown_minutes FROM command_warnings "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
