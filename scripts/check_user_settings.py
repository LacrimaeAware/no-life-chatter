"""Print the stored settings for a given Twitch user id.

    python scripts/check_user_settings.py <user_id>
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def get_user_settings(user_id):
    conn = sqlite3.connect(config.DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM user_settings WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    columns = [d[0] for d in c.description]
    conn.close()
    if result:
        return dict(zip(columns, result))
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_user_settings.py <user_id>")
        sys.exit(1)

    user_id = sys.argv[1]
    settings = get_user_settings(user_id)
    if settings:
        print(f"Settings for user {user_id}: {settings}")
    else:
        print(f"No settings found for user {user_id}.")
