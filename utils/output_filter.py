"""Output guard: block the bot from POSTING bannable content.

Whatever the bot sends to Twitch is on the bot account, regardless of how the
text was produced (Markov, an LLM, anything). Generated personas recombine real
chat, which can contain slurs/hate terms that get a bot banned. This module
checks generated text against a denylist before it's ever sent.

The denylist itself is NOT in this repo — it's loaded from a gitignored file
(config.BLOCKLIST_FILE, default data/unsynced/blocklist.txt: one term per line,
'#' comments). So the public showcase contains the mechanism, not a slur
catalog. With no file present, is_clean() returns True for everything (the
filter is opt-in via the blocklist).
"""

import logging
import re

import config

_terms = None


def _collapse(s: str) -> str:
    """Lowercase and drop everything but [a-z0-9] so spacing/punctuation/leet
    evasion ('n i g g a', 'f@g') collapses to the bare term for matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _load():
    global _terms
    if _terms is None:
        terms = set()
        try:
            # utf-8-sig: a BOM would otherwise glue to the first '#' and turn that
            # comment line into a live blocked term
            with open(config.BLOCKLIST_FILE, encoding="utf-8-sig") as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        c = _collapse(line)
                        if c:
                            terms.add(c)
        except FileNotFoundError:
            logging.info("output_filter: no blocklist file; nothing is filtered.")
        _terms = terms
    return _terms


def is_clean(text: str) -> bool:
    """True if text contains no denylisted term (evasion-collapsed substring)."""
    collapsed = _collapse(text)
    if not collapsed:
        return True
    return not any(term in collapsed for term in _load())


def active() -> bool:
    """Whether a non-empty denylist is loaded. Posting features that recombine
    real chat should refuse to send when this is False (fail closed) — Twitch
    bans the bot *and* the operator for posted slurs, with no safe channel."""
    return len(_load()) > 0
