"""Per-user language profiles.

The bot records which languages each user is *confidently* detected writing in.
That history is then used to:
  - translate borderline/short messages from a KNOWN speaker of a language, and
  - avoid translating users who only ever write English (a stray misdetection of
    their message is almost certainly wrong).

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


def english_only(profile: dict, min_total: int | None = None) -> bool:
    """True if `profile` shows a solid history that is *entirely* English."""
    min_total = config.SPEAKER_ENGLISH_ONLY_MIN if min_total is None else min_total
    if min_total <= 0 or not profile:
        return False
    # A manual non-English flag always overrides.
    if any(L != "EN" and v["flagged"] for L, v in profile.items()):
        return False
    total = sum(v["count"] for v in profile.values())
    english = profile.get("EN", {}).get("count", 0)
    return total >= min_total and english == total


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
