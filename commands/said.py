import time

from utils.chat_archive import (
    nearest_author_lines,
    normalize_author,
    normalize_channel,
    said,
    search_all,
    search_all_count,
)

description = (
    "Search the chat archive: did a user ever say something? Exact first, "
    "then a normalized close-match fallback. Use 'anyone' to search everyone. "
    "Supports chat=<channel>; use ~saidnext for more matches.\n"
    "  ~said <user> [chat=<channel>] <phrase>   |   ~said anyone [chat=<channel>] <phrase>"
)

PAGE_SIZE = 3
SESSION_TTL = 60
_SESSIONS = {}


def _clip(text, n=180):
    return text if len(text) <= n else text[: n - 1] + "..."


def _parse_chat_scope(params):
    rest, channel = [], None
    for p in params:
        low = p.lower()
        if low.startswith("chat="):
            value = low.split("=", 1)[1].strip().lstrip("#")
            channel = None if value in ("", "*", "all") else value
        else:
            rest.append(p)
    return rest, channel


def _scope_label(channel):
    return f" in #{normalize_channel(channel)}" if channel else ""


def _session_key(message):
    author = message.author.name.lower() if message.author else ""
    channel = message.channel.name.lower() if message.channel else ""
    return author, channel


def _save_session(message, payload):
    payload["expires"] = time.time() + SESSION_TTL
    _SESSIONS[_session_key(message)] = payload


def _format_rows(rows, everyone: bool):
    parts = []
    for sent_at, channel, *rest in rows:
        if everyone:
            author, content = rest
            author = normalize_author(author)
            parts.append(f"{author}#{channel} {sent_at[:10]}: \"{_clip(content, 80)}\"")
        else:
            content = rest[0]
            parts.append(f"#{channel} {sent_at[:10]}: \"{_clip(content, 90)}\"")
    return " | ".join(parts)


async def handle_said(bot, message, params):
    params, channel = _parse_chat_scope(params)
    if len(params) < 2:
        await message.channel.send("Usage: ~said <user|anyone> [chat=<channel>] <phrase>")
        return
    user, phrase = params[0], " ".join(params[1:])
    display_user = normalize_author(user)

    if user.lower() in ("anyone", "*", "everyone"):
        total = search_all_count(phrase, channel=channel)
        rows = search_all(phrase, limit=PAGE_SIZE, channel=channel)
        if not rows:
            await message.channel.send(
                f"Nobody on record saying \"{_clip(phrase, 80)}\"{_scope_label(channel)}."
            )
            return
        if total > len(rows):
            _save_session(message, {
                "kind": "anyone",
                "phrase": phrase,
                "channel": channel,
                "offset": len(rows),
                "total": total,
            })
        suffix = f" ({len(rows)}/{total}; ~saidnext)" if total > len(rows) else f" ({total})"
        await message.channel.send(
            f"Search \"{_clip(phrase, 50)}\"{_scope_label(channel)}{suffix}: "
            + _format_rows(rows, everyone=True)
        )
        return

    total, rows = said(user, phrase, limit=PAGE_SIZE, channel=channel)
    if total == 0:
        near = nearest_author_lines(user, phrase, limit=1, channel=channel)
        if near:
            score, sent_at, _channel, content = near[0]
            await message.channel.send(
                f"No exact record for {display_user}{_scope_label(channel)}. "
                f"Closest ({score:.0%}) on {sent_at[:10]}: "
                f"\"{_clip(content)}\""
            )
            return
        await message.channel.send(
            f"No record of {display_user} saying \"{_clip(phrase, 80)}\"{_scope_label(channel)}."
        )
        return
    times = "once" if total == 1 else f"{total} times"
    if total > len(rows):
        _save_session(message, {
            "kind": "author",
            "user": user,
            "phrase": phrase,
            "channel": channel,
            "offset": len(rows),
            "total": total,
        })
    next_hint = " Use ~saidnext for more." if total > len(rows) else ""
    await message.channel.send(
        f"{display_user} said that {times}{_scope_label(channel)}: "
        + _format_rows(rows, everyone=False)
        + next_hint
    )


async def saidnext(bot, message):
    key = _session_key(message)
    sess = _SESSIONS.get(key)
    if not sess or sess.get("expires", 0) < time.time():
        _SESSIONS.pop(key, None)
        await message.channel.send("No recent ~said search to continue.")
        return
    channel = sess.get("channel")
    offset = sess.get("offset", 0)
    total = sess.get("total", 0)
    if sess["kind"] == "anyone":
        rows = search_all(sess["phrase"], limit=PAGE_SIZE, offset=offset, channel=channel)
        everyone = True
    else:
        _total, rows = said(
            sess["user"], sess["phrase"], limit=PAGE_SIZE, offset=offset, channel=channel
        )
        everyone = False
    if not rows:
        _SESSIONS.pop(key, None)
        await message.channel.send("No more matches.")
        return
    sess["offset"] = offset + len(rows)
    sess["expires"] = time.time() + SESSION_TTL
    if sess["offset"] >= total:
        _SESSIONS.pop(key, None)
        suffix = f" ({sess['offset']}/{total}; end)"
    else:
        suffix = f" ({sess['offset']}/{total}; ~saidnext)"
    await message.channel.send("Next matches" + suffix + ": " + _format_rows(rows, everyone))
