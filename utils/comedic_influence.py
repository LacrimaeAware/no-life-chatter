"""Comedy Influence — does a chatter make OTHER people laugh, broadly?

The defining metric (re-checkable, three ideas stacked):

1. SPARK, not coincidence. For each "setup" message m by author A, look at the
   distinct OTHER users who laugh in the 30s AFTER m but were NOT already laughing
   in the 30s BEFORE m. Those are the people A *sparked* (started laughing). The
   "minus before" is the stream-confound fix: a chat reacting to the STREAM is
   already laughing before m too, so it cancels — A is only credited for laughter
   that begins when they speak, not for posting during a chat-wide KEKW wave.

2. BROAD beats CLIQUE. Count how many times each person B sparked-laughed at A
   across all of A's setups: spark[A][B]. A's score sums √(spark[A][B]) over B —
   so B's 1st laugh ≈ 1 but B's 100th adds almost nothing. One buddy who laughs at
   everything you say can't run up your score; many DIFFERENT people each laughing
   a little does. The "effective number of laughers" N_eff = (Σ√spark)² / Σspark
   is ~1–2 for a clique and ≈ breadth for broad appeal — used to gate cliques out.

3. REGULARS ONLY. influence(A) = Σ√spark / setups (clique-robust spark rate) →
   an IQ-style index (100 + 15·z over the roster) with shrinkage toward 100. Only
   real regulars are ranked (MIN_REGULAR combined setups): in a chat with ~1M
   lines, someone with a few dozen is a visitor, not a regular, and their rate is
   noise. This is a PER-LINE rate ("when you talk, do people laugh?"), so the
   highest-volume chatters who mostly say mundane things rank low, not high — that
   is the point (reach/total-reactions just measures who talks most).

Baked-in constraints (all requested): self-laughs never count; a setup needs
another user actually present to react; a message that is itself just a laugh is
not a setup; bots / command lines are excluded as setups and reactions. Utility
bots are dropped two ways — the manual block list (config.EXCLUDE_USERS) AND a
heuristic: an account whose lines are mostly templated bot output (_BOT_OUTPUT)
is skipped automatically. Each chat has its own baseline; a user's overall score
is the setup-weighted combine across the configured conversational chats
(config.COMEDY_CHANNELS).

A fun mirror over the archive, not a measure of who is objectively funny —
"funny" is subjective and this only captures one slice: who reliably gets a laugh
out of OTHER people when they talk.
"""

import re
import time
from collections import defaultdict, deque
from datetime import date

import config
from utils import chat_archive

# Conversational chats to rank by default (set in config.toml, gitignored).
# Falls back to all joined channels if unset.
DEFAULT_CHANNELS = tuple(config.COMEDY_CHANNELS) or tuple(config.CHANNELS)
WINDOW_SECS = 30        # look this many seconds before / after a setup
BURST_SECS = 12         # same-author fragments this close are one setup
FWD_MSG_CAP = 80        # safety bound on the forward scan inside a burst
AFTER_CAP = 8           # max distinct laughers collected per window (cost bound)
CACHE_TTL_SECS = 300    # also invalidated immediately by a newer live message
MIN_SETUPS = 200        # per-chat floor to enter a chat's baseline pool
MIN_REGULAR = 1000      # combined setups to be RANKED — a real regular, not a visitor
MIN_EFF_LAUGHERS = 3.0  # effective (clique-robust) distinct laughers to be ranked
SHRINK_K = 250          # confidence shrinkage: pull thin samples toward 100
BOT_OUTPUT_RATIO = 0.55 # exclude an account if >= this share of its lines are bot output
BOT_MIN_MSGS = 40       # ... and it has at least this many lines (skip tiny samples)

# Laughter / amusement reactions. Extends utils.reaction_tracker._LAUGH_RE with
# plain 'lol', rofl, hehe and the 💀 ("I'm dead") skull.
_LAUGH_RE = re.compile(
    r"(kekw?|kekl|lulw?|omegalul|megalul|lmf?ao+|\blo+l+\b|icant|"
    r"xd+|hahah+|ahah+|hehe+|pff+|rofl|\U0001F602|\U0001F923|\U0001F480)",
    re.IGNORECASE,
)

# Templated bot OUTPUT (ban/timeout logs, pong/uptime, deactivated lookups,
# now-live/now-playing). An account whose lines are mostly this is a utility bot,
# not a chatter — excluded automatically so the bot list isn't pure whack-a-mole.
_BOT_OUTPUT = re.compile(
    r"has been (banned|timed|unbanned)|\bpong!|latency:|uptime:|\(deactivated\)|"
    r"tos_indefinite|no logs are collected|not-whitelisted|is playing .+? (since|for)|"
    r"⛔|is now live|\bfollowage\b|\baccount age\b",
    re.IGNORECASE,
)

_cache = {}  # channel -> {cached_at, watermark, result}


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


def _stats_from_spark(sp, setups):
    """Turn one author's per-laugher spark counts into score fields."""
    raw = sum(sp.values())               # total spark events
    eff = sum(c ** 0.5 for c in sp.values())  # clique-robust (√ diminishing)
    return {
        "influence": eff / setups if setups else 0.0,
        "setups": setups,
        "breadth": len(sp),               # distinct people sparked
        "neff": (eff * eff / raw) if raw else 0.0,  # effective # of laughers
    }


def compute_channel(channel):
    """Single forward sweep of one chat's timeline. Returns
    {'authors': {a: {influence, setups, breadth, neff, best_net, best_text}},
     'mean': float, 'sd': float, 'n': int}. Cached per channel for the process."""
    channel = chat_archive.normalize_channel(channel)
    conn = chat_archive.connect()
    newest = conn.execute(
        "SELECT sent_at, id FROM messages WHERE channel=? "
        "ORDER BY sent_at DESC, id DESC LIMIT 1",
        (channel,),
    ).fetchone()
    watermark = tuple(newest) if newest else None
    now = time.monotonic()
    cached = _cache.get(channel)
    if (cached and cached["watermark"] == watermark
            and now - cached["cached_at"] < CACHE_TTL_SECS):
        return cached["result"]

    rows = conn.execute(
        "SELECT sent_at, author, content FROM messages WHERE channel=? ORDER BY sent_at, id",
        (channel,),
    ).fetchall()

    n = len(rows)
    times = [0] * n
    auth = [""] * n
    laugh = [False] * n
    _norm = {}
    bot_hits = defaultdict(int)
    msg_count = defaultdict(int)
    for i, (sent_at, a, content) in enumerate(rows):
        an = _norm.get(a)
        if an is None:
            an = _norm[a] = chat_archive.normalize_author(a)
        times[i] = _secs(sent_at)
        auth[i] = an
        laugh[i] = _is_laugh(content)
        msg_count[an] += 1
        if _BOT_OUTPUT.search(content or ""):
            bot_hits[an] += 1

    # accounts whose output is mostly templated bot text are utility bots, not
    # chatters — drop them up front (a heuristic on top of the manual block list).
    botlike = {a for a, c in msg_count.items()
               if c >= BOT_MIN_MSGS and bot_hits[a] / c >= BOT_OUTPUT_RATIO}
    human = {
        a for a in msg_count
        if a not in botlike and not chat_archive._is_noise_author(a)
    }

    # Twitch users commonly split one thought across several rapid messages. Keep
    # only the final eligible fragment as the setup so one reaction cannot reward
    # every line in the burst.
    next_setup = [-1] * n
    latest_setup = {}
    for i in range(n - 1, -1, -1):
        ai = auth[i]
        content_i = rows[i][2]
        eligible = ai in human and not laugh[i] and not _is_command(content_i)
        if eligible:
            next_setup[i] = latest_setup.get(ai, -1)
            latest_setup[ai] = i

    spark = defaultdict(lambda: defaultdict(int))  # A -> {B: # setups B sparked}
    setups = defaultdict(int)
    best_net = defaultdict(int)
    best_text = {}

    dq = deque()  # (t, author) of laughs within the trailing WINDOW_SECS
    for i in range(n):
        ti = times[i]
        floor = ti - WINDOW_SECS
        while dq and dq[0][0] < floor:
            dq.popleft()

        ai = auth[i]
        content_i = rows[i][2]
        later = next_setup[i]
        superseded = later >= 0 and times[later] - ti <= BURST_SECS
        if ai in human and not laugh[i] and not _is_command(content_i) and not superseded:
            after = set()
            others_present = False
            cnt = 0
            j = i + 1
            while j < n and times[j] <= ti + WINDOW_SECS and cnt < FWD_MSG_CAP:
                aj = auth[j]
                if aj != ai and aj in human:
                    others_present = True
                    if laugh[j]:
                        after.add(aj)
                        if len(after) >= AFTER_CAP:
                            break
                j += 1
                cnt += 1

            if others_present:
                before = {a for (_t, a) in dq if a != ai}
                net = after - before          # people A actually sparked
                setups[ai] += 1
                sp = spark[ai]
                for b in net:
                    sp[b] += 1
                if len(net) > best_net[ai]:
                    best_net[ai] = len(net)
                    best_text[ai] = content_i

        if laugh[i] and ai in human:
            dq.append((ti, ai))

    authors = {}
    for a, ns in setups.items():
        s = _stats_from_spark(spark.get(a, {}), ns)
        s["best_net"] = best_net.get(a, 0)
        s["best_text"] = best_text.get(a, "")
        s["spark"] = dict(spark.get(a, {}))
        authors[a] = s

    rankable = [s["influence"] for s in authors.values() if s["setups"] >= MIN_SETUPS]
    if rankable:
        mean = sum(rankable) / len(rankable)
        sd = (sum((x - mean) ** 2 for x in rankable) / len(rankable)) ** 0.5
    else:
        mean, sd = 0.0, 0.0

    result = {"authors": authors, "mean": mean, "sd": sd, "n": n}
    _cache[channel] = {
        "cached_at": now,
        "watermark": watermark,
        "result": result,
    }
    return result


def _z(stats, res):
    if not res["sd"]:
        return 0.0
    return (stats["influence"] - res["mean"]) / res["sd"]


def _index(zc, total_setups):
    """IQ-style index with confidence shrinkage: a thin sample is pulled toward
    100 (the roster average) so a few lucky windows can't crown someone."""
    shrunk = zc * total_setups / (total_setups + SHRINK_K)
    return round(100 + 15 * shrunk)


def _combine(canon, channels):
    """Setup-weighted combine of one author's per-chat z. Returns the aggregate
    dict (pre-gate) or None if they appear in no qualifying chat."""
    zw = w = 0.0
    combined_spark = defaultdict(int)
    best_net, best_text = -1, ""
    chats_used = []
    for ch in channels:
        res = compute_channel(ch)
        s = res["authors"].get(canon)
        if not s or s["setups"] < MIN_SETUPS:
            continue
        zw += _z(s, res) * s["setups"]
        w += s["setups"]
        for laugher, count in s.get("spark", {}).items():
            combined_spark[laugher] += count
        chats_used.append(chat_archive.normalize_channel(ch))
        if s["best_net"] > best_net:
            best_net, best_text = s["best_net"], s["best_text"]
    if not w:
        return None
    zc = zw / w
    raw = sum(combined_spark.values())
    eff = sum(count ** 0.5 for count in combined_spark.values())
    return {
        "index": _index(zc, w),
        "z": zc,
        "setups": int(w),
        "breadth": len(combined_spark),
        "neff": (eff * eff / raw) if raw else 0.0,
        "best_net": best_net,
        "best_text": best_text,
        "chats": chats_used,
    }


def _rankable(agg):
    """A real regular with broad enough appeal to be scored."""
    return bool(agg) and agg["setups"] >= MIN_REGULAR and agg["neff"] >= MIN_EFF_LAUGHERS


def user_score(user, channels=DEFAULT_CHANNELS):
    """A user's comedy index across `channels`, or None unless they're a real
    regular (MIN_REGULAR combined setups) with clique-robust breadth."""
    agg = _combine(chat_archive.normalize_author(user), channels)
    return agg if _rankable(agg) else None


def leaderboard(channels=DEFAULT_CHANNELS, n=5, bottom=False):
    """Ranked comedic influence across `channels`, restricted to real regulars
    (MIN_REGULAR combined setups, MIN_EFF_LAUGHERS effective distinct laughers)."""
    authors = set()
    for ch in channels:
        res = compute_channel(ch)
        authors.update(a for a, s in res["authors"].items() if s["setups"] >= MIN_SETUPS)
    out = [(a, agg) for a in authors
           for agg in [_combine(a, channels)] if _rankable(agg)]
    out.sort(key=lambda kv: kv[1]["index"], reverse=not bottom)
    return out[:n]


def clear_cache():
    """Drop process-local results after maintenance or test fixture changes."""
    _cache.clear()
