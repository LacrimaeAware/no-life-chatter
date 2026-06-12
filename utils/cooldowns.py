"""Anti-spam command cooldowns — punishes stacking, not heavy use.

Design (user-specified):
- The malicious pattern is sending MORE commands while your previous one is
  still processing. Using commands a lot but waiting for each reply is fine
  and never punished. Someone else prompting right after you is fine —
  everything here is strictly per-user.
- Trigger: 3+ commands stacked while one is still pending → automatic
  cooldown, 5 minutes, DOUBLING on every subsequent offense (5, 10, 20...).
  One in-chat notice; further commands during cooldown are silently dropped.
- Every offense is recorded to the settings DB (command_warnings) for later
  review (~warnings, super admin). Strike counts persist; they decay after a
  clean 24h.
"""

import logging
import sqlite3
import time

import config

BASE_COOLDOWN = 5 * 60
STACK_LIMIT = 3          # pending + this many more = offense
DECAY_SECONDS = 24 * 3600

_pending = {}            # user -> count of in-flight commands
_stacked = {}            # user -> commands sent while pending
_cooldown_until = {}     # user -> epoch
_notified = set()        # users already told about their current cooldown


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
        return 0   # clean day -> forgiven
    return strikes


def _offense(user):
    strikes = _strikes(user) + 1
    cooldown = BASE_COOLDOWN * (2 ** (strikes - 1))
    with _conn() as conn:
        conn.execute("INSERT OR REPLACE INTO command_strikes VALUES (?,?,?)",
                     (user, strikes, time.time()))
        conn.execute("INSERT INTO command_warnings (user_name, at, reason, cooldown_minutes)"
                     " VALUES (?, datetime('now'), ?, ?)",
                     (user, f"stacked {STACK_LIMIT}+ commands while one was pending",
                      round(cooldown / 60, 1)))
    _cooldown_until[user] = time.time() + cooldown
    _notified.discard(user)
    logging.warning(f"command cooldown: {user} for {cooldown/60:.0f}m (strike {strikes})")
    return cooldown


def before(user: str):
    """Call before dispatching. Returns:
    ('ok', None)        — run the command (pending count incremented)
    ('drop', None)      — silently ignore (already notified about cooldown)
    ('cooldown', secs)  — NEW cooldown triggered or first drop: notify once
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
            secs = int(_offense(user))
            _notified.add(user)   # the offense notice IS the one notification
            return ("cooldown", secs)
        # a single impatient re-send is tolerated — it still runs
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
