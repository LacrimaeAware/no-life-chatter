import sqlite3
import logging
from auth import is_super_admin
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_channel_setting(channel_name, setting_value):
    db_path = config.DB_PATH
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO channel_settings (channel_name, translation_enabled)
                VALUES (?, ?)
                ON CONFLICT(channel_name) DO UPDATE SET
                translation_enabled=excluded.translation_enabled;
            """, (channel_name, setting_value))
            conn.commit()
            logging.info(f"Translation setting for channel {channel_name} updated to {setting_value}")
    except Exception as e:
        logging.error(f"Error updating channel setting for {channel_name}: {e}")

async def handle_chan_autotl(bot, message, params):
    if not is_super_admin(message.author.name):
        await message.channel.send("You do not have permission to manage channel translation settings.")
        logging.warning(f"Unauthorized access attempt by {message.author.name}")
        return

    if not params or len(params) < 2 or params[1] not in ['on', 'off']:
        await message.channel.send("Invalid command usage. Use 'chan_autotl [channel_name] on' or 'chan_autotl [channel_name] off'.")
        return

    channel_name = params[0]
    new_status = 1 if params[1] == 'on' else 0
    update_channel_setting(channel_name, new_status)
    await message.channel.send(f"Automatic translation for channel '{channel_name}' {'enabled' if new_status else 'disabled'}.")
    logging.info(f"Automatic translation for channel {channel_name} toggled to {'on' if new_status else 'off'} by {message.author.name}")
