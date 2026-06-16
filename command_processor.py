import asyncio
import logging
import urllib.error
from collections import defaultdict

from command_registry import command_handlers, load_command_handlers  # noqa: F401
from utils.command_bans import is_banned
from utils import cooldowns

# Slow LLM generation commands. These are serialized PER CHANNEL so their
# replies arrive one at a time instead of as a simultaneous batch (which also
# trips Twitch's anti-spam), and an identical request already in flight is
# dropped with a brief one-time notice instead of generated twice.
SERIAL_COMMANDS = {"persona", "hyper", "generate"}

_OFFLINE_SIGNS = ("10061", "actively refused", "connection refused", "max retries",
                  "connection aborted", "failed to establish", "[errno 111]")


def _backend_offline(exc):
    """True if exc looks like the local model server (LM Studio) being down —
    so embedding/LLM commands can say so instead of a generic error."""
    if isinstance(exc, (urllib.error.URLError, ConnectionError)):
        return True
    return any(sign in str(exc).lower() for sign in _OFFLINE_SIGNS)


class CommandProcessor:
    def __init__(self, bot):
        self.bot = bot
        self._gen_locks = defaultdict(asyncio.Lock)   # channel -> serialization lock
        self._inflight = defaultdict(set)             # channel -> signatures running/queued
        self._warned = defaultdict(set)               # channel -> sigs already warned about

    async def process_command(self, message):
        # command-banned users are silently ignored (~banuser/~unbanuser)
        if message.author and is_banned(message.author.name):
            logging.info(f"Ignored command from banned user {message.author.name}")
            return
        parts = message.content.lstrip(self.bot.prefix).split()
        if not parts:  # bare '~' / '~~' — not a command, don't crash
            return
        command = parts[0].lower()
        params = parts[1:]

        handler = command_handlers.get(command)
        if not handler:
            logging.warning(f"Command not recognized: {command}")
            await message.channel.send(f"Command not recognized. Try {self.bot.prefix}help for command list.")
            return

        # Anti-spam: stacking commands while your previous one is still
        # processing is the troll pattern; heavy-but-patient use never
        # triggers (utils/cooldowns.py — escalating, per-user, reviewable
        # via ~warnings).
        user = message.author.name if message.author else ""
        verdict, secs = cooldowns.before(user)
        if verdict == "drop":
            return
        if verdict == "cooldown":
            await message.channel.send(
                f"@{user} slow down — command cooldown {secs // 60}m {secs % 60}s "
                "(stacking commands before the bot answers).")
            return

        if command in SERIAL_COMMANDS and message.channel:
            await self._run_serial(message, command, params, handler, user)
        else:
            await self._run(message, command, params, handler, user)

    async def _report_error(self, message, command, exc):
        if _backend_offline(exc):
            await message.channel.send(
                "🔌 local model server is offline — this command needs it (start LM Studio).")
        else:
            logging.error(f"Error handling command {command}: {exc}")
            await message.channel.send("An error occurred while processing your command.")

    async def _run(self, message, command, params, handler, user):
        try:
            await handler(self.bot, message, params)
        except Exception as e:
            await self._report_error(message, command, e)
        finally:
            cooldowns.after(user)

    async def _run_serial(self, message, command, params, handler, user):
        """Generation commands: one at a time per channel, dropping an exact
        duplicate that is already running/queued so chat can't batch-spam the
        bot (and risk an auto-timeout)."""
        channel = message.channel.name
        sig = command + " " + " ".join(params).strip().lower()

        if sig in self._inflight[channel]:
            cooldowns.after(user)   # we are NOT running it — release the pending slot
            if sig not in self._warned[channel]:
                self._warned[channel].add(sig)
                await message.channel.send(f"@{user} already on that one ⏳")
            return

        self._inflight[channel].add(sig)
        try:
            async with self._gen_locks[channel]:
                await handler(self.bot, message, params)
        except Exception as e:
            await self._report_error(message, command, e)
        finally:
            self._inflight[channel].discard(sig)
            self._warned[channel].discard(sig)
            cooldowns.after(user)
