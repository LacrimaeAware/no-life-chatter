import asyncio
import logging
import urllib.error
from collections import defaultdict, deque

from command_registry import command_handlers, load_command_handlers  # noqa: F401
from utils.command_bans import is_banned
from utils import cooldowns

# GPU-heavy LLM generation commands. ONLY these (and only with args) are
# cooldown-tracked — spamming them hammers the GPU. They are also serialized per
# channel so their replies don't arrive as a batch. Every fast command
# (~help, ~why, ~said, a bare ~persona usage error, ...) runs freely and can
# NEVER cost a cooldown — that was the over-eager behavior chat complained about.
SERIAL_COMMANDS = {"persona", "hyper", "generate"}

# Commands that may build dynamic axes. They hit LM Studio chat + embeddings
# from blocking helper code, so they need one global lane across all channels.
MODEL_QUEUE_COMMANDS = {"top", "bottom", "axis"}

_OFFLINE_SIGNS = ("10061", "actively refused", "connection refused", "max retries",
                  "connection aborted", "failed to establish", "[errno 111]")


def _backend_offline(exc):
    """True if exc looks like the local model server (LM Studio) being down —
    so embedding/LLM commands can say so instead of a generic error."""
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, (urllib.error.URLError, ConnectionError)):
        return True
    return any(sign in str(exc).lower() for sign in _OFFLINE_SIGNS)


def _backend_rejected(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return True
    text = str(exc).lower()
    return (
        "http error 400" in text
        or "http error 500" in text
        or "failed to load model" in text
        or "has not started loading" in text
        or "operation canceled" in text
    )


class CommandProcessor:
    def __init__(self, bot):
        self.bot = bot
        self._gen_locks = defaultdict(asyncio.Lock)   # channel -> serialization lock
        self._inflight = defaultdict(set)             # channel -> signatures running/queued
        self._warned = defaultdict(set)               # channel -> sigs already warned about
        self._model_lock = asyncio.Lock()
        self._model_queue = deque()
        self._model_inflight = set()

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

        user = message.author.name if message.author else ""

        # Only GPU-heavy generation commands WITH args go through the cooldown +
        # serialization path. Everything else (incl. a bare '~persona' that just
        # prints usage) runs freely — no cooldown, no lock.
        if command in MODEL_QUEUE_COMMANDS and params and message.channel:
            await self._run_model_queued(message, command, params, handler, user)
        elif command in SERIAL_COMMANDS and params and message.channel:
            verdict, secs = cooldowns.before(user)
            if verdict == "drop":
                return
            if verdict == "warn":
                await message.channel.send(
                    f"@{user} slow down — wait for the bot to reply before re-sending "
                    "(keep stacking heavy commands and it's a short cooldown).")
                return
            if verdict == "cooldown":
                await message.channel.send(
                    f"@{user} command cooldown {secs // 60}m {secs % 60}s — you kept "
                    "stacking heavy commands after the warning.")
                return
            await self._run_serial(message, command, params, handler, user)
        else:
            await self._run(message, command, params, handler)

    async def _report_error(self, message, command, exc):
        if _backend_offline(exc):
            await message.channel.send(
                "🔌 local model server is offline — this command needs it (start LM Studio).")
        elif _backend_rejected(exc):
            logging.warning(f"Local model rejected command {command}: {exc}")
            await message.channel.send(
                "Local model is up, but rejected/overloaded that request. "
                "Try again after the model queue clears.")
        else:
            logging.error(f"Error handling command {command}: {exc}")
            await message.channel.send("An error occurred while processing your command.")

    async def _run(self, message, command, params, handler):
        """Fast commands: run immediately, no cooldown tracking."""
        try:
            await handler(self.bot, message, params)
        except Exception as e:
            await self._report_error(message, command, e)

    async def _run_model_queued(self, message, command, params, handler, user):
        """One global lane for dynamic-axis commands.

        LM Studio can keep chat completions alive while the embedding model is
        loading/unloading. Without this queue, several ~top calls can hit the
        embedding endpoint at once and come back late/out of order.
        """
        sig = command + " " + " ".join(params).strip().lower()
        if sig in self._model_inflight:
            await message.channel.send(f"@{user} already queued/running that model command ⏳")
            return

        position = len(self._model_queue) + (1 if self._model_lock.locked() else 0)
        self._model_inflight.add(sig)
        self._model_queue.append(sig)
        if position:
            await message.channel.send(f"@{user} queued for model work (#{position})")

        try:
            async with self._model_lock:
                try:
                    self._model_queue.remove(sig)
                except ValueError:
                    pass
                await handler(self.bot, message, params)
        except Exception as e:
            await self._report_error(message, command, e)
        finally:
            self._model_inflight.discard(sig)
            try:
                self._model_queue.remove(sig)
            except ValueError:
                pass

    async def _run_serial(self, message, command, params, handler, user):
        """GPU-heavy generation commands: one at a time per channel, with an
        immediate 'processing' ack (so you know it started), and exact duplicates
        already in flight dropped with a brief notice."""
        channel = message.channel.name
        sig = command + " " + " ".join(params).strip().lower()

        if sig in self._inflight[channel]:
            cooldowns.after(user)   # we are NOT running it — release the pending slot
            if sig not in self._warned[channel]:
                self._warned[channel].add(sig)
                await message.channel.send(f"@{user} already on that one ⏳")
            return

        self._inflight[channel].add(sig)
        # immediate, minimal feedback that the heavy command started
        # (Duardo: "no feedback anything started"; keep it short, don't echo args).
        await message.channel.send(f"@{user} Processing...")
        try:
            async with self._gen_locks[channel]:
                await handler(self.bot, message, params)
        except Exception as e:
            await self._report_error(message, command, e)
        finally:
            self._inflight[channel].discard(sig)
            self._warned[channel].discard(sig)
            cooldowns.after(user)
