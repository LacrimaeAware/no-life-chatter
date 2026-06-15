"""Comedy Influence — does a chatter make OTHER people laugh?

The defining metric (kept deliberately simple and re-checkable):

  For each "setup" message m posted by author A, compare how many DISTINCT
  OTHER users laugh in the 30s AFTER m versus the 30s BEFORE m:

      lift(m) = after_laughers - before_laughers      (each capped at CAP)

  influence(A) = mean lift over all of A's eligible setups.

Why the before/after DELTA (this is the whole point):
  A streamer's chat reacts to the STREAM, not to each other — a funny moment
  on stream makes everyone post KEKW at once. That ambient laughter sits in the
  window BEFORE a message just as much as AFTER it, so it cancels in the delta.
  A chatter is only credited for laughter that STARTS after they speak. A
  synchronized external wave raises before≈after for everyone → ~0 credit. So
  the metric measures conversational comedic influence, not "was posting while a
  stream was funny."

Constraints baked in (all requested):
  - laughers are OTHER users only — laughing at your own joke never counts.
  - a setup counts only if another user is actually present to react
    (`others_near`), so monologues into dead air don't score.
  - a message that is ITSELF just a laugh is not a setup (no laugh-chain credit).
  - bots / command lines are excluded as setups and as reactions.
  - distinct laughers are capped per window (CAP) so one big coincidental wave
    can't dominate, and BREADTH (how many different people you've made laugh)
    is tracked separately as an anti-"one friend always laughs" signal.

Scoring: per chat, influence is turned into an IQ-style index
(100 + 15·z over rankable authors). Each chat has its own baseline/dynamics, so
a user's overall score is the setup-weighted combine of their per-chat z across
the configured conversational chats (config.COMEDY_CHANNELS).

This is a fun mirror over the archive, not a measurement of who is objectively
funny. Limitation: a chatter who reliably posts right before genuine external
laughs (and rarely before the ambient rate) can still pick up a little credit;
the delta + roster baseline shrink it but do not erase it.
"""

import re
from collections import defaultdict, deque
from datetime import date

import config
from utils import chat_archive

# Conversational chats to rank by default (set in config.toml, gitignored).
# Falls back to all joined channels if unset.
DEFAULT_CHANNELS = tuple(config.COMEDY_CHANNELS) or tuple(config.CHANNELS)
WINDOW_SECS = 30        # look this many seconds before / after a setup
FWD_MSG_CAP = 80        # safety bound on the forward scan inside a burst
CAP = 4                 # max distinct laughers counted per window
MIN_SETUPS = 60         # eligible setups needed to enter a chat's baseline pool
MIN_BREADTH = 4         # must have made >= this many DIFFERENT people laugh
SHRINK_K = 200          # confidence shrinkage: pull small samples toward 100

# Laughter / amusement reactions. Extends utils.reaction_tracker._LAUGH_RE with
# plain 'lol', rofl, hehe and the 💀 ("I'm dead") skull.
_LAUGH_RE = re.compile(
    r"(kekw?|kekl|lulw?|omegalul|megalul|lmf?ao+|\blo+l+\b|icant|"
    r"xd+|hahah+|ahah+|hehe+|pff+|rofl|\U0001F602|\U0001F923|\U0001F480)",
    re.IGNORECASE,
)

_cache = {}  # channel -> result dict


def _secs(s, _ord={}):
    """'YYYY-MM-DD HH:MM:SS' -> monotonic seconds. Caches the date->ordinal map
    (only a few hundred distinct dates) so this is cheap over ~1M rows."""
    d = s[:10]
    o = _ord.get(d)
    if o is None:
        o = date(int(d[:4]), int(d[5:7]), int(d[8:10])).toordinal()
        _ord[d] = o
    return o * 86400 + int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def _is_laugh(text):
    return bool(_LAUGH_RE.search(text or ""))


def _is_command(text):
    return (text or "").lstrip()[:1] in ("~", "!", "$", "<", "#", "/")


def compute_channel(channel):
    """Single forward sweep of one chat's timeline. Returns
    {'authors': {a: {influence, setups, breadth, best_after, best_text}},
     'mean': float, 'sd': float, 'n': int}. Cached per channel for the process."""
    channel = chat_archive.normalize_channel(channel)
    if channel in _cache:
        return _cache[channel]

    conn = chat_archive.connect()
    rows = conn.execute(
        "SELECT sent_at, author, content FROM messages WHERE channel=? ORDER BY sent_at, id",
        (channel,),
    ).fetchall()

    n = len(rows)
    times = [0] * n
    auth = [""] * n
    laugh = [False] * n
    _norm = {}
    _noise = {}
    for i, (sent_at, a, content) in enumerate(rows):
        an = _norm.get(a)
        if an is None:
            an = _norm[a] = chat_archive.normalize_author(a)
        times[i] = _secs(sent_at)
        auth[i] = an
        laugh[i] = _is_laugh(content)

    setups = defaultdict(int)
    lift_sum = defaultdict(float)
    breadth = defaultdict(set)
    best_after = defaultdict(lambda: -1)
    best_text = {}

    dq = deque()  # (t, author) of laughs within the trailing WINDOW_SECS
    for i in range(n):
        ti = times[i]
        # evict laughs older than the before-window
        floor = ti - WINDOW_SECS
        while dq and dq[0][0] < floor:
            dq.popleft()

        ai = auth[i]
        content_i = rows[i][2]
        is_noise = _noise.get(ai)
        if is_noise is None:
            is_noise = _noise[ai] = chat_archive._is_noise_author(ai)
        eligible = not laugh[i] and not _is_command(content_i) and not is_noise
        if eligible:
            # forward window: distinct OTHER laughers + is anyone else present
            after_laughers = set()
            others_present = False
            cnt = 0
            j = i + 1
            while j < n and times[j] <= ti + WINDOW_SECS and cnt < FWD_MSG_CAP:
                aj = auth[j]
                if aj != ai:
                    others_present = True
                    if laugh[j]:
                        after_laughers.add(aj)
                        if len(after_laughers) >= CAP:
                            break
                j += 1
                cnt += 1

            if others_present:
                before_laughers = {a for (_t, a) in dq if a != ai}
                after = min(len(after_laughers), CAP)
                before = min(len(before_laughers), CAP)
                setups[ai] += 1
                lift_sum[ai] += after - before
                breadth[ai] |= after_laughers
                if after > before and after > best_after[ai]:
                    best_after[ai] = after
                    best_text[ai] = content_i

        if laugh[i]:
            dq.append((ti, ai))

    authors = {}
    for a, ns in setups.items():
        authors[a] = {
            "influence": lift_sum[a] / ns,
            "setups": ns,
            "breadth": len(breadth[a]),
            "best_after": best_after[a],
            "best_text": best_text.get(a, ""),
        }

    rankable = [s["influence"] for s in authors.values() if s["setups"] >= MIN_SETUPS]
    if rankable:
        mean = sum(rankable) / len(rankable)
        var = sum((x - mean) ** 2 for x in rankable) / len(rankable)
        sd = var ** 0.5
    else:
        mean, sd = 0.0, 0.0

    result = {"authors": authors, "mean": mean, "sd": sd, "n": n}
    _cache[channel] = result
    return result


def _z(stats, res):
    if not res["sd"]:
        return 0.0
    return (stats["influence"] - res["mean"]) / res["sd"]


def _index(zc, total_setups):
    """IQ-style index with confidence shrinkage: a thin sample is pulled toward
    100 (the roster average) so a couple of lucky windows can't crown someone."""
    shrunk = zc * total_setups / (total_setups + SHRINK_K)
    return round(100 + 15 * shrunk)


def _combine(canon, channels):
    """Setup-weighted combine of one author's per-chat z. Returns the aggregate
    dict (pre-gate) or None if they appear in no qualifying chat."""
    zw = w = infl_w = breadth = 0.0
    best_after, best_text = -1, ""
    chats_used = []
    for ch in channels:
        res = compute_channel(ch)
        s = res["authors"].get(canon)
        if not s or s["setups"] < MIN_SETUPS:
            continue
        zw += _z(s, res) * s["setups"]
        infl_w += s["influence"] * s["setups"]
        w += s["setups"]
        breadth += s["breadth"]
        chats_used.append(chat_archive.normalize_channel(ch))
        if s["best_after"] > best_after:
            best_after, best_text = s["best_after"], s["best_text"]
    if not w:
        return None
    zc = zw / w
    return {
        "index": _index(zc, w),
        "z": zc,
        "lift": infl_w / w,
        "setups": int(w),
        "breadth": int(breadth),
        "best_after": best_after,
        "best_text": best_text,
        "chats": chats_used,
    }


def user_score(user, channels=DEFAULT_CHANNELS):
    """A user's comedy index across `channels`, or None if they lack the setups
    or breadth to be scored reliably."""
    agg = _combine(chat_archive.normalize_author(user), channels)
    if not agg or agg["breadth"] < MIN_BREADTH:
        return None
    return agg


def leaderboard(channels=DEFAULT_CHANNELS, n=5, bottom=False):
    """Ranked comedic influence across `channels`. Authors need MIN_SETUPS in at
    least one chat and MIN_BREADTH distinct laughers overall."""
    authors = set()
    for ch in channels:
        res = compute_channel(ch)
        authors.update(a for a, s in res["authors"].items() if s["setups"] >= MIN_SETUPS)
    out = []
    for a in authors:
        agg = _combine(a, channels)
        if agg and agg["breadth"] >= MIN_BREADTH:
            out.append((a, agg))
    out.sort(key=lambda kv: kv[1]["index"], reverse=not bottom)
    return out[:n]
