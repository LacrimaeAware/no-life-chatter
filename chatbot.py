import json
import logging
import sqlite3
import threading
import time

from twitchio.ext import commands

import config
from handlers import MessageHandler
from utils.token_manager import manage_token_lifecycle

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_token():
    """Safely read the bot's access token from the token file."""
    try:
        with open(config.TOKEN_FILE, "r") as file:
            data = json.load(file)
        return data["access_token"]
    except FileNotFoundError:
        logging.error(
            "Token file not found at %s. Run scripts/get_initial_token.py first.",
            config.TOKEN_FILE,
        )
        return None


def start_token_management():
    """Start the background thread that keeps the OAuth token fresh."""
    token_thread = threading.Thread(target=manage_token_lifecycle, daemon=True)
    token_thread.start()


def initialize_channel_settings(channels, default_translation_enabled=False):
    """Make sure every configured channel has a settings row."""
    with sqlite3.connect(config.DB_PATH) as conn:
        cursor = conn.cursor()
        for channel in channels:
            cursor.execute(
                "SELECT channel_name FROM channel_settings WHERE channel_name = ?",
                (channel,),
            )
            if not cursor.fetchone():
                cursor.execute(
                    """
                    INSERT INTO channel_settings (channel_name, translation_enabled)
                    VALUES (?, ?)
                    """,
                    (channel, int(default_translation_enabled)),
                )
                logging.info(f"Default settings initialized for channel {channel}")
        conn.commit()


class Bot(commands.Bot):
    def __init__(self):
        self.token = get_token()
        self.client_id = config.TWITCH_CLIENT_ID
        self.prefix = config.PREFIX

        super().__init__(
            token=self.token,
            client_id=self.client_id,
            prefix=self.prefix,
            initial_channels=config.CHANNELS,
        )

        self.handler = MessageHandler(self)
        initialize_channel_settings(config.CHANNELS)

    async def event_ready(self):
        logging.info(f"Ready | Logged in as {self.nick}")
        if config.READY_MESSAGE:
            for channel_name in config.CHANNELS:
                channel = self.get_channel(channel_name)
                if channel:
                    await channel.send(config.READY_MESSAGE)

    async def event_message(self, message):
        await self.handler.process_message(message)


def main():
    logging.info("Starting token management thread...")
    start_token_management()
    time.sleep(2)  # give the token manager a moment to refresh if needed

    bot = Bot()
    bot.run()


if __name__ == "__main__":
    main()
