"""Zero-cost persona generator: per-user Markov word chains.

Builds an order-N word-level Markov model from everything a user has said in
the archive, then samples new messages from it. No LLM, no API, no network —
just statistical recombination of their *own* words, so it reproduces their
vocabulary, emotes and cadence for free and with no provider content policy in
play. (It does NOT understand conversation context — that's what the LLM
persona, Phase 2/3 in docs/PERSONA_BOT_ROADMAP.md, is for. This is the warm-up
toy and the always-available fallback.)

Output is recombined real chat, so it can reproduce whatever the source said,
slurs included — see the persona docs on Twitch-side output filtering before
any of this is ever posted to chat.
"""

import random
import re
from collections import defaultdict

from utils import chat_archive

_BOUNDARY = ("\x02",)  # sentinel marking start/end of a message


def build_from_messages(msgs, order: int = 2, label: str = "fusion"):
    """Build a Markov model from an explicit message list (e.g. a multi-person
    fusion for ~generate). Returns None if there's nothing usable."""
    chain = defaultdict(list)
    starts = []
    used = 0
    for m in msgs:
        toks = m.split()
        if not toks:
            continue
        used += 1
        padded = list(_BOUNDARY) * order + toks + list(_BOUNDARY)
        starts.append(tuple(padded[:order]))
        for i in range(len(padded) - order):
            chain[tuple(padded[i:i + order])].append(padded[i + order])

    if not starts:
        return None
    return {
        "author": label,
        "order": order,
        "chain": dict(chain),
        "starts": starts,
        "source_messages": used,
    }


def build(author: str, order: int = 2, min_messages: int = 40, channel: str = None):
    """Build a Markov model for author, or None if they have too little text.
    With `channel`, prefers their messages in that chat (their voice THERE),
    falling back to full history when they barely chat in it."""
    msgs = []
    if channel:
        msgs = chat_archive.messages_for(author, channel=channel)
    if len(msgs) < max(min_messages, 300):
        msgs = chat_archive.messages_for(author)
    if len(msgs) < min_messages:
        return None
    return build_from_messages(msgs, order=order,
                               label=chat_archive.normalize_author(author))


_model_cache = {}


def get_model(author: str, order: int = 2, min_messages: int = 40, channel: str = None):
    """Cached build() — so the ~mimic command doesn't re-scan a user's whole
    history on every call. Cache is per process; clears on bot restart."""
    channel_key = chat_archive.normalize_channel(channel) if channel else None
    key = (chat_archive.normalize_author(author), order, channel_key)
    if key not in _model_cache:
        _model_cache[key] = build(author, order=order, min_messages=min_messages,
                                  channel=channel_key)
    return _model_cache[key]


def generate(model, max_words: int = 40, rng: random.Random = None):
    """Sample one message from a model. Returns a string (may be empty-ish)."""
    if not model or not model["starts"]:
        return None
    rng = rng or random
    order = model["order"]
    state = list(rng.choice(model["starts"]))
    out = []
    for _ in range(max_words):
        nxts = model["chain"].get(tuple(state[-order:]))
        if not nxts:
            break
        nxt = rng.choice(nxts)
        if nxt in _BOUNDARY:
            break
        out.append(nxt)
        state.append(nxt)
    return " ".join(out).strip()


def sample(author: str, n: int = 5, order: int = 2, max_words: int = 40, seed=None):
    """Build + generate n messages for author. Returns (model_info, [lines])."""
    model = build(author, order=order)
    if not model:
        return None, []
    rng = random.Random(seed)
    lines = []
    for _ in range(n * 4):  # oversample, drop empties / verbatim echoes, dedupe
        line = generate(model, max_words=max_words, rng=rng)
        if line and len(line.split()) >= order and line not in lines:
            lines.append(line)
        if len(lines) >= n:
            break
    return {"source_messages": model["source_messages"]}, lines
