"""Resolve archive authors to Twitch user IDs (identity should be id-dominant).

Names change; ids don't. This fills the author_ids table by resolving every
current archive author through the log aggregator's API (which returns
{login, id} for any ACTIVE login). Dead/renamed old logins can't be resolved
retroactively (Twitch frees them) — those remain the rename-oracle's job —
but from now on identity merging can be id-first, and the live bot records
ids as it sees chatters (handlers.py).

After resolving, any two archive authors sharing an id are THE SAME PERSON
as a fact — reported at the end for alias application.

    python scripts/resolve_twitch_ids.py [--min-messages 100]
"""

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import chat_archive  # noqa: E402

API = "https://logs.zonian.dev/api/{channel}/{user}?jsonBasic=1"


def ensure_table(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS author_ids ("
        " author TEXT PRIMARY KEY, twitch_id TEXT, checked_at TEXT, "
        " display TEXT, last_seen_live TEXT)")
    for column in ("display", "last_seen_live"):
        try:
            conn.execute(f"ALTER TABLE author_ids ADD COLUMN {column} TEXT")
        except Exception:
            pass
    conn.commit()


def resolve(channel, user):
    url = API.format(channel=urllib.request.quote(channel),
                     user=urllib.request.quote(user))
    req = urllib.request.Request(url, headers={"User-Agent": "NoLifeChatter-ids"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=15))
        u = (d.get("request") or {}).get("user") or {}
        if u.get("id"):
            return str(u["id"])
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-messages", type=int, default=100)
    args = ap.parse_args()
    conn = chat_archive.connect()
    ensure_table(conn)
    rows = conn.execute(
        "SELECT author, COUNT(*) c FROM messages GROUP BY author "
        "HAVING c >= ? ORDER BY c DESC", (args.min_messages,)).fetchall()
    have = {a for a, in conn.execute(
        "SELECT author FROM author_ids WHERE twitch_id IS NOT NULL")}
    todo = [a for a, _ in rows if a not in have and "bot" not in a]
    print(f"{len(todo)} authors to resolve...")
    for i, a in enumerate(todo, 1):
        ch = conn.execute(
            "SELECT channel FROM messages WHERE author=? GROUP BY channel "
            "ORDER BY COUNT(*) DESC LIMIT 1", (a,)).fetchone()[0]
        tid = resolve(ch, a)
        conn.execute(
            "INSERT INTO author_ids (author, twitch_id, checked_at) VALUES (?,?,?) "
            "ON CONFLICT(author) DO UPDATE SET "
            "twitch_id=excluded.twitch_id, checked_at=excluded.checked_at",
            (a, tid, time.strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        print(f"  ({i}/{len(todo)}) {a}: {tid or 'UNRESOLVED (dead/renamed login)'}",
              flush=True)
        time.sleep(0.35)

    print("\n=== authors sharing a twitch id (same person, FACT) ===")
    dupes = conn.execute(
        "SELECT twitch_id, GROUP_CONCAT(author) FROM author_ids "
        "WHERE twitch_id IS NOT NULL GROUP BY twitch_id "
        "HAVING COUNT(*) > 1").fetchall()
    if not dupes:
        print("  none among resolved (current) logins")
    for tid, names in dupes:
        print(f"  id {tid}: {names}")
    n_dead = conn.execute(
        "SELECT COUNT(*) FROM author_ids WHERE twitch_id IS NULL").fetchone()[0]
    print(f"\nunresolved (dead old names — rename-oracle territory): {n_dead}")


if __name__ == "__main__":
    main()
