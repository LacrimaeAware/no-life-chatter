import logging
import urllib.error

from command_registry import command_handlers, load_command_handlers  # noqa: F401
from services import model_queue
from utils.command_bans import is_banned

# Commands that hit LM Studio hard enough to share one global lane across all
# channels. Required commands need the model; optional commands can fall back to
# receipts/deterministic output when the server is down.
MODEL_REQUIRED_COMMANDS = {
    "persona", "hyper", "generate",
    "top", "bottom", "axis", "why", "irony", "traits", "distinct",
}
MODEL_OPTIONAL_COMMANDS = {"askchat", "emote", "explain"}
MODEL_QUEUE_COMMANDS = MODEL_REQUIRED_COMMANDS | MODEL_OPTIONAL_COMMANDS
GENERATE_LOCAL_SUBCOMMANDS = {"save", "list", "del", "delete"}
RAW_RECEIPT_FLAGS = {"raw", "scores", "vector", "evidence", "receipts", "noai"}

_OFFLINE_SIGNS = (
    "10061", "actively refused", "connection refused", "max retries",
    "connection aborted", "failed to establish", "[errno 111]",
)
_REJECTED_SIGNS = (
    "http error 400", "http error 500", "failed to load model",
    "has not started loading", "operation canceled", "timed out", "timeout",
    "server disconnected",
)


def _backend_offline(exc):
    """True only when the local model server looks unreachable."""
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, urllib.error.URLError):
        return any(sign in str(getattr(exc, "reason", exc)).lower() for sign in _OFFLINE_SIGNS)
    if isinstance(exc, ConnectionError):
        return True
    return any(sign in str(exc).lower() for sign in _OFFLINE_SIGNS)


def _backend_rejected(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return True
    text = str(exc).lower()
    return any(sign in text for sign in _REJECTED_SIGNS)


def _has_raw_flag(params, start=0) -> bool:
    return any((p or "").lower() in RAW_RECEIPT_FLAGS for p in params[start:])


def _axis_term(params) -> str:
    """The trait phrase for ~top/~bottom — every bare word joined (traits can be
    multi-word, e.g. 'anal lover'), skipping flags (burst, user=<name>, @user)
    and a numeric n. Matches how the command parses the trait, so the cache
    check (fast-path vs model queue) is made against the real axis name."""
    words = []
    for p in params:
        low = (p or "").lower()
        if (low == "burst" or low.startswith("user=")
                or (p or "").startswith("@") or low.isdigit()):
            continue
        words.append(low)
    return " ".join(words).strip()


def _model_command_kind(command, params) -> str | None:
    """Return 'required', 'optional', or None for a command invocation."""
    if not params:
        return None
    command = (command or "").lower()
    first = params[0].lower()
    if command == "generate" and first in GENERATE_LOCAL_SUBCOMMANDS:
        return None
    if command == "generate" and any((p or "").lower() == "engine=markov" for p in params):
        return None
    if command == "askchat":
        return None if first in RAW_RECEIPT_FLAGS else "optional"
    if command == "emote":
        return None if _has_raw_flag(params, 1) else "optional"
    if command == "explain":
        if first not in {"emote", "emotes"} or len(params) < 2:
            return None
        return None if _has_raw_flag(params, 2) else "optional"
    if command == "why" and first in {"emote", "emotes"}:
        if len(params) < 2:
            return None
        return None if _has_raw_flag(params, 2) else "optional"
    if command in {"top", "bottom"}:
        # Only a genuinely NEW axis is model work. Built-in poles and
        # already-built dynamic axes (incl. their opposite pole / aliases)
        # resolve with no LLM call — run them fast so they don't stall behind
        # a live axis build in the global queue.
        term = _axis_term(params)
        if term:
            from utils import persona_axes
            if persona_axes.axis_cached(term):
                return None
        return "required"
    if command in MODEL_REQUIRED_COMMANDS:
        return "required"
    if command in MODEL_OPTIONAL_COMMANDS:
        return "optional"
    return None


class CommandProcessor:
    def __init__(self, bot):
        self.bot = bot

    async def process_command(self, message):
        # command-banned users are silently ignored (~banuser/~unbanuser)
        if message.author and is_banned(message.author.name):
            logging.info(f"Ignored command from banned user {message.author.name}")
            return
        parts = message.content.lstrip(self.bot.prefix).split()
        if not parts:
            return
        command = parts[0].lower()
        params = parts[1:]

        handler = command_handlers.get(command)
        if not handler:
            logging.warning(f"Command not recognized: {command}")
            await message.channel.send(f"Command not recognized. Try {self.bot.prefix}help for command list.")
            return

        user = message.author.name if message.author else ""
        model_kind = _model_command_kind(command, params)
        if model_kind and message.channel:
            await self._run_model_queued(message, command, params, handler, user, model_kind)
        else:
            await self._run(message, command, params, handler)

    async def _report_error(self, message, command, exc):
        if _backend_offline(exc):
            await message.channel.send(model_queue.OFFLINE_MESSAGE)
        elif _backend_rejected(exc):
            logging.warning(f"Local model rejected command {command}: {exc}")
            await message.channel.send(
                "Local model is up but busy/rejected that request. "
                "Try again after the model queue clears.")
        else:
            logging.error(f"Error handling command {command}: {exc}")
            await message.channel.send("An error occurred while processing your command.")

    async def _run(self, message, command, params, handler):
        """Fast commands: run immediately."""
        try:
            await handler(self.bot, message, params)
        except Exception as e:
            await self._report_error(message, command, e)

    def model_queue_status(self) -> str:
        return model_queue.status()

    def clear_model_queue(self) -> int:
        return model_queue.clear_pending()

    async def _run_model_queued(self, message, command, params, handler, user, model_kind):
        """One shared lane for command invocations that use LM Studio."""
        sig = command + " " + " ".join(params).strip().lower()
        label = command + (" " + " ".join(params[:2]) if params else "")

        async def work():
            await handler(self.bot, message, params)

        try:
            await model_queue.submit(
                label=label,
                user=user,
                user_key=user,
                sig=sig,
                model_kind=model_kind,
                send=message.channel.send,
                work=work,
            )
        except Exception as e:
            await self._report_error(message, command, e)
