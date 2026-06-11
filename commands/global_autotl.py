import sqlite3
import logging
from auth import is_super_admin
import config

# Configure logging
description = (
    'Global auto-translate switch (super admins).\n'
    '  ~global_autotl on|off'
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_global_setting(setting_value):
    db_path = config.DB_PATH
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO global_settings (setting_name, setting_value)
                VALUES ('global_translation', ?)
                ON CONFLICT(setting_name) DO UPDATE SET
                setting_value=excluded.setting_value;
            """, (setting_value,))
            conn.commit()
            logging.info(f"Global translation setting updated to {setting_value}")
    except Exception as e:
        logging.error(f"Error updating global setting: {e}")

async def handle_global_autotl(bot, message, params):
    if not is_super_admin(message.author.name):
        await message.channel.send("You do not have permission to manage global translation settings.")
        logging.warning(f"Unauthorized access attempt by {message.author.name}")
        return

    if not params or params[0] not in ['on', 'off']:
        await message.channel.send("Invalid command usage. Use 'global_autotl on' or 'global_autotl off'.")
        return

    new_status = 1 if params[0] == 'on' else 0
    update_global_setting(new_status)
    await message.channel.send(f"Global automatic translation {'enabled' if new_status else 'disabled'}.")
    logging.info(f"Global automatic translation toggled to {'on' if new_status else 'off'} by {message.author.name}")
