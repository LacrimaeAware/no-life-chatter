"""Implicit funniness labels: score chat's reaction to the bot's own lines.

The user's insight: we don't need to ask anyone whether a persona line landed —
chat tells us. After the bot posts a persona/generate line, watch the next
~60s of that channel: laugh emotes (KEKW/LMAO/OMEGALUL...), replies that
mention the bot, and general burst are summed into a reaction score that gets
appended to the persona log (type=reaction_feedback). Over weeks this becomes
a labeled funniness dataset for free — the training signal for the judge
problem (which generations actually land), no oracle time spent.

Never raises into the message path; windows are tiny and self-expiring.
"""

import logging
import re
import time

from utils.persona_llm import log_event

WINDOW_SECONDS = 60
MAX_MESSAGES = 14

_LAUGH_RE = re.compile(
    r"(kekw|kek\b|lulw?\b|omegalul|lmf?ao+|icant\b|xdd+|looo+l|hahah+|"
    r"\U0001F602|\U0001F923|pfff+|lmaoo+)", re.IGNORECASE)

_windows = []  # [{channel, t0, text, meta, msgs, laughs, responders, mentions}]


def watch(channel: str, text: str, meta: dict) -> None:
    """Open a reaction window for a line the bot just posted."""
    try:
        _windows.append({
            "channel": (channel or "").lower(), "t0": time.time(),
            "text": text, "meta": dict(meta or {}),
            "msgs": 0, "laughs": 0, "responders": set(), "mentions": 0,
        })
        if len(_windows) > 12:   # safety: never accumulate
            _finalize(_windows.pop(0))
    except Exception as e:
        logging.debug(f"reaction watch failed: {e}")


def observe(channel: str, author: str, content: str, bot_nick: str = "") -> None:
    """Feed every incoming chat message; closes/accumulates open windows."""
    try:
        now = time.time()
        chan = (channel or "").lower()
        author = (author or "").lower()
        for w in list(_windows):
            if now - w["t0"] > WINDOW_SECONDS or w["msgs"] >= MAX_MESSAGES:
                _windows.remove(w)
                _finalize(w)
                continue
            if w["channel"] != chan or not author or author == (bot_nick or "").lower():
                continue
            w["msgs"] += 1
            laughs = len(_LAUGH_RE.findall(content or ""))
            if laughs:
                w["laughs"] += min(laughs, 3)
                w["responders"].add(author)
            if bot_nick and bot_nick.lower() in (content or "").lower():
                w["mentions"] += 1
                w["responders"].add(author)
    except Exception as e:
        logging.debug(f"reaction observe failed: {e}")


def _finalize(w: dict) -> None:
    try:
        score = w["laughs"] + 2 * w["mentions"]
        log_event({
            "type": "reaction_feedback",
            "channel": w["channel"],
            "bot_line": w["text"],
            "meta": w["meta"],
            "window_msgs": w["msgs"],
            "laughs": w["laughs"],
            "bot_mentions": w["mentions"],
            "unique_responders": len(w["responders"]),
            "reaction_score": score,
        })
    except Exception as e:
        logging.debug(f"reaction finalize failed: {e}")
