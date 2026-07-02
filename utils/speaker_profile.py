"""Per-user language profiles.

The bot records which languages each user is *confidently* detected writing in.
Once a user has written a language enough times (a simple count threshold — no
percentages), they're a "known speaker" of it, and the bot will translate even
their short messages in that language.

Profiles are keyed by Twitch user id (same as user_settings) and are global
across channels — a person's language doesn't change per channel.

Stored in the `user_languages` table (see scripts/init_db.py).
"""

import logging
import sqlite3

import config


def record_language(user_id, lang: str) -> None:
    """Count one confident observation of `user_id` writing in `lang`."""
    lang = (lang or "").upper()
    if not user_id or not lang:
        return
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO user_languages (user_id, lang, count) VALUES (?, ?, 1)
                ON CONFLICT(user_id, lang) DO UPDATE SET count = count + 1
                """,
                (str(user_id), lang),
            )
            conn.commit()
    except Exception as e:
        logging.warning(f"record_language failed: {e}")


def get_profile(user_id) -> dict:
    """Return {LANG: {'count': int, 'flagged': bool}} for a user."""
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            rows = conn.execute(
                "SELECT lang, count, flagged FROM user_languages WHERE user_id = ?",
                (str(user_id),),
            ).fetchall()
    except Exception as e:
        logging.warning(f"get_profile failed: {e}")
        return {}
    return {r[0]: {"count": r[1], "flagged": bool(r[2])} for r in rows}


def known_speaker(profile: dict, lang: str, min_count: int | None = None) -> bool:
    """True if `profile` is manually flagged for `lang`, or has written it enough.

    Takes a profile dict (from get_profile) so callers can evaluate it against
    the user's history *before* recording the current message.
    """
    min_count = config.SPEAKER_MIN_COUNT if min_count is None else min_count
    entry = profile.get((lang or "").upper())
    if not entry:
        return False
    return entry["flagged"] or entry["count"] >= min_count


def known_languages(user_id) -> set:
    """Languages this user is an established speaker of (flagged via ~speak, or
    written confidently >= SPEAKER_MIN_COUNT LONG sentences). Used by the
    translate gate's short-message concession."""
    profile = get_profile(user_id)
    return {lang for lang in profile if known_speaker(profile, lang)}


def is_massive_speaker(user_id, lang: str) -> bool:
    """A heavy, established speaker (>= SPEAKER_MASSIVE_COUNT long foreign
    sentences) — the only tier that unlocks SINGLE-WORD auto-translation."""
    entry = get_profile(user_id).get((lang or "").upper())
    if not entry:
        return False
    return entry["flagged"] or entry["count"] >= config.SPEAKER_MASSIVE_COUNT


def reset_counts() -> int:
    """Wipe auto-accumulated speaker counts (keep manual ~speak flags). Run when
    the flagging criteria change so polluted counts don't grandfather people in.
    Returns rows removed."""
    with sqlite3.connect(config.DB_PATH) as conn:
        cur = conn.execute("DELETE FROM user_languages WHERE flagged = 0")
        conn.commit()
        return cur.rowcount


def flag_speaker(user_id, lang: str, on: bool = True) -> None:
    """Manually mark (or unmark) a user as a speaker of `lang`."""
    lang = (lang or "").upper()
    if not user_id or not lang:
        return
    val = 1 if on else 0
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO user_languages (user_id, lang, count, flagged) VALUES (?, ?, 0, ?)
            ON CONFLICT(user_id, lang) DO UPDATE SET flagged = ?
            """,
            (str(user_id), lang, val, val),
        )
        conn.commit()
