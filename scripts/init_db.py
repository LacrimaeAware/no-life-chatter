"""Create (or upgrade) the SQLite settings database.

Run this once before starting the bot for the first time:

    python scripts/init_db.py

It is safe to run repeatedly — it only creates tables/columns that are missing
and never deletes data.
"""

import logging
import os
import sqlite3
import sys

# Make the project root importable so we can read config.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _column_names(cursor, table):
    cursor.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def init_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()

        # Per-user settings (keyed by Twitch user id).
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT PRIMARY KEY,
                translation_enabled BOOLEAN NOT NULL DEFAULT 0,
                translation_language TEXT NOT NULL DEFAULT 'EN',
                output_mode TEXT DEFAULT 'default',
                output_channel TEXT DEFAULT NULL,
                practice_mode INTEGER DEFAULT 0,
                native_lang TEXT DEFAULT 'EN',
                learn_lang TEXT DEFAULT 'ES',
                romanize_enabled INTEGER DEFAULT 0
            )
            """
        )

        # Global on/off switches.
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS global_settings (
                setting_name TEXT PRIMARY KEY,
                setting_value BOOLEAN NOT NULL
            )
            """
        )

        # Per-channel translation toggle.
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS channel_settings (
                channel_name TEXT PRIMARY KEY,
                translation_enabled BOOLEAN NOT NULL DEFAULT 0,
                other_setting TEXT DEFAULT NULL
            )
            """
        )

        # Recently active chatters per channel (used to strip names before
        # translating). Contains usernames, so the DB is git-ignored.
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS channel_users (
                channel_name TEXT,
                user_name TEXT,
                last_active TIMESTAMP,
                PRIMARY KEY (channel_name, user_name)
            )
            """
        )

        # Per-user language profiles: how often each user is detected writing
        # in each language (+ a manual "flagged" override). Used to translate
        # known speakers reliably and skip English-only users.
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS user_languages (
                user_id TEXT,
                lang TEXT,
                count INTEGER DEFAULT 0,
                flagged INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, lang)
            )
            """
        )

        # Seed the global translation switch OFF so a fresh install never
        # translates anything until you explicitly enable it.
        c.execute(
            """
            INSERT INTO global_settings (setting_name, setting_value)
            VALUES ('global_translation', 0)
            ON CONFLICT(setting_name) DO NOTHING
            """
        )

        conn.commit()
    logging.info("Database initialized at %s", db_path)


if __name__ == "__main__":
    init_db(config.DB_PATH)
