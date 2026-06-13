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
        logging.info(f"Ready | Logged in as {self.nick} (id {self.user_id})")
        logging.info(f"Joined channels: {[c.name for c in self.connected_channels]}")
        self.handler.message_service.start_resident_idle_loop()
        # Sending the ready message doubles as a self-test: if Twitch is dropping
        # our messages (unverified phone, suspended, shadowban), we'll either get
        # a NOTICE (logged by event_raw_data below) or a send exception here.
        if config.READY_MESSAGE:
            targets = [config.READY_CHANNEL] if config.READY_CHANNEL else config.CHANNELS
            for channel_name in targets:
                channel = self.get_channel(channel_name)
                if not channel:
                    logging.warning(f"Ready msg: not joined to '{channel_name}' — cannot send.")
                    continue
                try:
                    await channel.send(config.READY_MESSAGE)
                    logging.info(f"Ready msg sent to #{channel_name} (if it didn't appear, watch for a NOTICE).")
                except Exception as e:
                    logging.error(f"Ready msg send FAILED for #{channel_name}: {e!r}")

    async def event_message(self, message):
        await self.handler.process_message(message)

    async def event_raw_data(self, data):
        # Surface Twitch's own NOTICE messages — this is how Twitch tells a bot
        # *why* a message was dropped (e.g. msg-id=msg_requires_verified_phone_number,
        # msg_banned, msg_channel_suspended, msg_duplicate, msg_rejected). Without
        # this they're invisible and the bot just looks silently broken.
        for line in data.split("\r\n"):
            if " NOTICE " in line and "tmi.twitch.tv" in line:
                logging.warning(f"Twitch NOTICE: {line.strip()}")

    async def event_error(self, error, data=None):
        logging.error(f"twitchio event error: {error!r} | data={data!r}")


def acquire_single_instance_lock():
    """Bind a localhost port as a machine-wide mutex. A second chatbot.py —
    from the keep-alive loop, the login autostart, or a manual launch — fails
    the bind and exits instead of double-connecting to Twitch (which makes the
    bot answer every command twice). The OS releases the port on ANY exit, so
    a crash can never wedge the lock."""
    import socket
    import sys
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", 48917))
    except OSError:
        logging.error("Another NoLifeChatter instance is already running — exiting.")
        sys.exit(0)
    lock.listen(1)
    return lock  # keep a reference so the socket lives for the process lifetime


def main():
    instance_lock = acquire_single_instance_lock()  # noqa: F841
    logging.info("Starting token management thread...")
    start_token_management()
    time.sleep(2)  # give the token manager a moment to refresh if needed

    bot = Bot()
    bot.run()


if __name__ == "__main__":
    main()
