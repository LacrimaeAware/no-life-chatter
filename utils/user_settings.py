import sqlite3
import logging

import config

DB = config.DB_PATH

def get_user_settings(user_id):
    """
    Retrieves all user settings as a dict for a given user_id.
    """
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM user_settings WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    if row is None:
        conn.close()
        logging.warning(f"No settings found for user {user_id}.")
        return None
    # get column names BEFORE closing
    columns = [d[0] for d in c.description]
    conn.close()
    settings = dict(zip(columns, row))
    logging.info(f"Retrieved settings for user {user_id}: {settings}")
    return settings

def update_user_setting(user_id, setting, value):
    """
    Upserts a specific setting for a user.
    """
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    query = f"""
    INSERT INTO user_settings (user_id, {setting})
    VALUES (?, ?)
    ON CONFLICT(user_id) DO UPDATE SET {setting} = excluded.{setting};
    """
    c.execute(query, (user_id, value))
    conn.commit()
    conn.close()
    logging.info(f"Updated setting '{setting}' for user {user_id} to {value}.")

def ensure_user_settings(user_id, default_settings=None):
    """
    Ensures there is a row for the user; creates with defaults if missing.
    """
    if default_settings is None:
        default_settings = {
            'translation_enabled': 0,
            'translation_language': 'EN',
            'output_mode': 'default',
            'output_channel': None,
            # NEW defaults for practice mode:
            'practice_mode': 0,
            'native_lang': 'EN',
            'learn_lang': 'ES',
            'romanize_enabled': 0
        }

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT user_id FROM user_settings WHERE user_id = ?", (user_id,))
    exists = c.fetchone() is not None
    if not exists:
        logging.info(f"No settings found for user {user_id}, initializing defaults.")
        cols = ', '.join(default_settings.keys())
        placeholders = ', '.join(['?'] * len(default_settings))
        vals = tuple(default_settings.values())
        c.execute(f"INSERT INTO user_settings (user_id, {cols}) VALUES (?, {placeholders})", (user_id,) + vals)
        conn.commit()
        logging.info(f"Default settings initialized for user {user_id}.")
    else:
        logging.info(f"Settings already exist for user {user_id}.")
    conn.close()

# OPTIONAL: quick helper to dump settings textually
def format_user_settings(user_id):
    s = get_user_settings(user_id)
    if not s:
        return "No settings."
    keys = ['translation_enabled','translation_language','output_mode','output_channel',
            'practice_mode','native_lang','learn_lang']
    return ', '.join(f"{k}={s.get(k)!r}" for k in keys if k in s)
