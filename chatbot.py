import json
import logging
import sqlite3
import threading

from twitchio.ext import commands

import config
from handlers import MessageHandler
from utils import token_manager

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


def start_token_management(on_refresh=None):
    """Start the background thread that keeps the OAuth token fresh. on_refresh
    is called after each successful refresh so the live bot can adopt it."""
    token_thread = threading.Thread(
        target=token_manager.manage_token_lifecycle,
        kwargs={"on_refresh": on_refresh},
        daemon=True,
    )
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
        self._auth_failures = 0

        super().__init__(
            token=self.token,
            client_id=self.client_id,
            prefix=self.prefix,
            initial_channels=config.CHANNELS,
        )

        self.handler = MessageHandler(self)
        initialize_channel_settings(config.CHANNELS)

    def apply_current_token(self):
        """Push the latest token from disk into the LIVE twitchio session.

        The background refresher writes new tokens to the file, but twitchio
        (2.10) strips 'oauth:' at init and caches the token on both the websocket
        (`_connection._token`) and the HTTP client (`_http.token`); on reconnect
        it re-sends that cached token. So after a refresh the live socket keeps
        presenting the OLD token and Twitch answers 'Login authentication failed'
        forever until a restart. Re-applying the current file token here makes the
        next (re)connect and Helix call use the fresh one. Returns True if the
        live token changed. (Touches twitchio internals — pinned to 2.10.0.)"""
        raw = get_token()
        if not raw:
            return False
        bare = raw.replace("oauth:", "")
        changed = False
        conn = getattr(self, "_connection", None)
        if conn is not None and getattr(conn, "_token", None) != bare:
            conn._token = bare
            changed = True
        http = getattr(self, "_http", None)
        if http is not None and getattr(http, "token", None) != bare:
            http.token = bare
            changed = True
        if changed:
            logging.info("Applied refreshed token to the live connection.")
        return changed

    def _handle_auth_failure(self):
        """Self-heal the recurring stale-token wedge: when Twitch rejects login,
        re-read the (refreshed) file token and apply it so the next reconnect
        succeeds. If applying changes nothing AND failures persist, the refresh
        token itself is dead — say so loudly instead of looping silently."""
        self._auth_failures += 1
        if self.apply_current_token():
            logging.warning(
                "Login auth failed — re-applied the current disk token; the next "
                "reconnect will use it.")
        elif self._auth_failures >= 6:
            logging.critical(
                "Login auth still failing after %d tries and the disk token is "
                "unchanged — the refresh token is likely revoked. Re-authorize "
                "with scripts/get_initial_token.py.", self._auth_failures)

    async def event_ready(self):
        self._auth_failures = 0
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
                if "Login authentication failed" in line:
                    self._handle_auth_failure()

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

    # Refresh synchronously BEFORE connecting so the bot reads a fresh file token
    # (no startup race), then start the background loop that adopts later
    # refreshes into the live session.
    logging.info("Checking token freshness before connect...")
    try:
        token_manager.check_and_refresh_token()
    except Exception as e:
        logging.error(f"Startup token check failed: {e}")

    bot = Bot()
    start_token_management(on_refresh=bot.apply_current_token)
    bot.run()


if __name__ == "__main__":
    main()
