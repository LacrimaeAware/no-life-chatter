"""Runtime state for channel-scoped resident personas.

The state file is intentionally under data/unsynced: this is live operating
state, not public configuration. Commands can update it while the bot runs, and
the message service reads it on each chat message.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from utils import chat_archive

STATE_FILE = Path("data/unsynced/resident_personas.json")
MODES = {"regular", "response", "random", "silent"}

DEFAULTS = {
    "mode": "regular",
    "chance": 0.02,
    "topic_chance": 0.16,
    "topic_curve": 2.0,
    "directed_chance": 0.65,
    "directed_cooldown": 0.0,
    "greeting_chance": 0.75,
    "cooldown": 20.0,
    "idle_chance": 0.025,
    "idle_after": 180.0,
    "idle_interval": 75.0,
    "idle_cooldown": 240.0,
    "max_bot_streak": 3,
    "reply_to_trigger": True,
    "prefix": "",
    "context": "",
    "until": 0.0,
}


def _prob(value: float) -> float:
    return max(0.0, min(1.0, value))


def now() -> float:
    return time.time()


def load_all() -> dict:
    try:
        with STATE_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_all(data: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)


def _normalize_state(channel: str, state: dict) -> dict:
    out = dict(DEFAULTS)
    out.update(state or {})
    out["channel"] = chat_archive.normalize_channel(channel)
    out["persona"] = chat_archive.normalize_author(out.get("persona") or "")
    out["mode"] = str(out.get("mode") or "regular").lower()
    if out["mode"] not in MODES:
        out["mode"] = "regular"
    for key in (
        "chance", "topic_chance", "directed_chance", "greeting_chance",
        "cooldown", "directed_cooldown", "idle_chance", "idle_after",
        "idle_interval", "idle_cooldown", "topic_curve", "until",
    ):
        try:
            out[key] = float(out.get(key) or 0)
        except Exception:
            out[key] = float(DEFAULTS.get(key, 0))
    for key in ("chance", "topic_chance", "directed_chance", "greeting_chance", "idle_chance"):
        out[key] = _prob(out[key])
    for key in ("cooldown", "directed_cooldown", "idle_after", "idle_interval", "idle_cooldown"):
        out[key] = max(0.0, out[key])
    out["topic_curve"] = max(0.25, out["topic_curve"])
    out["until"] = max(0.0, out["until"])
    try:
        out["max_bot_streak"] = int(out.get("max_bot_streak") or DEFAULTS["max_bot_streak"])
    except Exception:
        out["max_bot_streak"] = DEFAULTS["max_bot_streak"]
    out["prefix"] = str(out.get("prefix") or "")
    out["context"] = str(out.get("context") or "")
    raw_reply = out.get("reply_to_trigger")
    if isinstance(raw_reply, str):
        out["reply_to_trigger"] = raw_reply.strip().lower() not in {"0", "false", "no", "off"}
    else:
        out["reply_to_trigger"] = bool(raw_reply)
    return out


def get(channel: str) -> dict | None:
    channel = chat_archive.normalize_channel(channel)
    data = load_all()
    raw = data.get(channel)
    if not raw:
        return None
    state = _normalize_state(channel, raw)
    if not state.get("persona"):
        return None
    if state["until"] and state["until"] <= now():
        data.pop(channel, None)
        save_all(data)
        return None
    return state


def set_state(channel: str, **updates) -> dict:
    channel = chat_archive.normalize_channel(channel)
    data = load_all()
    state = _normalize_state(channel, data.get(channel, {}))
    for key, value in updates.items():
        if value is not None:
            state[key] = value
    state = _normalize_state(channel, state)
    state["updated_at"] = now()
    data[channel] = state
    save_all(data)
    return state


def clear(channel: str) -> bool:
    channel = chat_archive.normalize_channel(channel)
    data = load_all()
    existed = channel in data
    data.pop(channel, None)
    save_all(data)
    return existed


def active_channels() -> list[dict]:
    out = []
    for channel in sorted(load_all()):
        state = get(channel)
        if state:
            out.append(state)
    return out


def format_status(state: dict) -> str:
    if not state:
        return "resident persona: off"
    until = float(state.get("until") or 0)
    if until:
        remaining = max(0, int(until - now()))
        tail = f", {remaining // 3600}h{(remaining % 3600) // 60:02d}m left"
    else:
        tail = ", no expiry"
    return (
        f"#{state['channel']}: {state['persona']} mode={state['mode']} "
        f"chance={state['chance']:.3g} topic={state['topic_chance']:.3g} "
        f"curve={state['topic_curve']:.2g} directed={state['directed_chance']:.3g} "
        f"cooldown={int(state['cooldown'])}s "
        f"idle={state['idle_chance']:.3g}/{int(state['idle_interval'])}s{tail}"
    )


def format_line(state: dict, line: str) -> str:
    line = re.sub(r"\s+", " ", line or "").strip()
    persona = re.escape(str(state.get("persona") or ""))
    if persona:
        line = re.sub(rf"^{persona}\s*[:>-]\s*", "", line, flags=re.I).strip()
    prefix = str(state.get("prefix") or "").strip()
    if prefix:
        prefix_words = re.escape(prefix.replace("\U0001f4e3", "").strip())
        if prefix_words:
            line = re.sub(rf"^{prefix_words}\s*(?:\U0001f4e3)?\s*", "", line, flags=re.I)
        line = line.strip()
        if not line:
            return ""
        line = f"{prefix} {line}".strip()
    return line
