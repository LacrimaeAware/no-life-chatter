"""Shared gate for LM Studio work.

Commands, resident personas, and ambient persona reactions all compete for the
same local GPU/model server. This module gives them one process-wide lane so
they do not stampede LM Studio and then report misleading "offline" failures.
"""

import asyncio
import logging
from collections import deque

import aiohttp

import config

OFFLINE_MESSAGE = "local model server is offline - this command needs it (start LM Studio)."

_lock = asyncio.Lock()
_queue = deque()
_entries = {}
_user_active = {}
_sig_active = {}
_cancelled = set()
_running = None
_ticket = 0


def _base_url() -> str:
    return config.LLM_ENDPOINT.split("/v1/")[0]


async def server_state() -> str:
    """Return up, busy, or offline.

    "offline" is reserved for connection-refused / host-unreachable style
    failures. Timeouts and odd HTTP states usually mean LM Studio is busy,
    loading, or wedged, not literally off.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=2)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(_base_url() + "/v1/models") as resp:
                return "up" if resp.status == 200 else "busy"
    except asyncio.TimeoutError:
        return "busy"
    except aiohttp.ClientConnectorError:
        return "offline"
    except OSError as exc:
        text = str(exc).lower()
        if "10061" in text or "connection refused" in text or "actively refused" in text:
            return "offline"
        return "busy"
    except Exception as exc:
        logging.debug("model server probe was inconclusive: %r", exc)
        return "busy"


def is_busy() -> bool:
    return _lock.locked() or bool(_queue)


def status() -> str:
    running = _entries.get(_running)
    pending = [_entries[t] for t in list(_queue) if t in _entries]
    if not running and not pending:
        return "model queue idle"
    bits = []
    if running:
        bits.append(f"running {running['label']}")
    if pending:
        listed = " ".join(f"#{i + 1}:{entry['label']}" for i, entry in enumerate(pending[:5]))
        bits.append("queued " + listed)
        if len(pending) > 5:
            bits.append(f"+{len(pending) - 5} more")
    return "model queue: " + " | ".join(bits)


def clear_pending() -> int:
    pending = list(_queue)
    _cancelled.update(pending)
    for ticket in pending:
        entry = _entries.pop(ticket, None)
        if entry:
            if entry.get("user_key") and _user_active.get(entry["user_key"]) == ticket:
                _user_active.pop(entry["user_key"], None)
            if entry.get("sig") and _sig_active.get(entry["sig"]) == ticket:
                _sig_active.pop(entry["sig"], None)
    _queue.clear()
    return len(pending)


def running_ticket():
    return _running


def _short_label(label: str) -> str:
    label = " ".join((label or "model").split())
    return label[:48]


async def _notify(send, text: str):
    if send:
        await send(text)


async def submit(
    *,
    label: str,
    work,
    model_kind: str = "required",
    send=None,
    user: str = "",
    user_key: str | None = None,
    sig: str | None = None,
    queue_if_busy: bool = True,
    announce: bool = True,
):
    """Run one model job through the shared queue.

    model_kind is "required" for commands that cannot answer without LM Studio
    and "optional" for commands that can produce deterministic receipts when it
    is unavailable. work must be an async callable.
    """
    global _ticket, _running

    user_key = (user_key or "").lower() or None
    sig = (sig or "").strip().lower() or None

    if not is_busy():
        state = await server_state()
        if state == "offline":
            if model_kind == "optional":
                return await work()
            await _notify(send, OFFLINE_MESSAGE)
            return None

    if not queue_if_busy and is_busy():
        return None

    if user_key and user_key in _user_active:
        await _notify(send, f"@{user} wait for your current model command to finish.")
        return None
    if sig and sig in _sig_active:
        await _notify(send, f"@{user} that model command is already queued/running.")
        return None

    _ticket += 1
    ticket = _ticket
    queued = _lock.locked() or bool(_queue)
    entry = {
        "ticket": ticket,
        "label": _short_label(label),
        "user": user,
        "user_key": user_key,
        "sig": sig,
        "queued": queued,
    }
    _entries[ticket] = entry
    if user_key:
        _user_active[user_key] = ticket
    if sig:
        _sig_active[sig] = ticket

    if queued:
        _queue.append(ticket)
        if announce:
            await _notify(send, f"@{user} queued for model work (#{len(_queue)})")
    elif announce:
        await _notify(send, f"@{user} Processing...")

    try:
        async with _lock:
            if ticket in _cancelled:
                return None
            try:
                _queue.remove(ticket)
            except ValueError:
                pass
            _running = ticket
            if entry.get("queued") and announce:
                await _notify(send, f"@{user} Processing...")
            state = await server_state()
            if state == "offline":
                if model_kind == "optional":
                    return await work()
                await _notify(send, OFFLINE_MESSAGE)
                return None
            return await work()
    finally:
        if _running == ticket:
            _running = None
        _cancelled.discard(ticket)
        _entries.pop(ticket, None)
        if user_key and _user_active.get(user_key) == ticket:
            _user_active.pop(user_key, None)
        if sig and _sig_active.get(sig) == ticket:
            _sig_active.pop(sig, None)
        try:
            _queue.remove(ticket)
        except ValueError:
            pass
