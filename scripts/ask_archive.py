"""Query the chat archive from the command line (full, unabridged answers).

    python scripts/ask_archive.py said <user> <phrase...>   # did X ever say Y
    python scripts/ask_archive.py near <user> <phrase...>   # closest lines by X
    python scripts/ask_archive.py quote <user>              # random quote by X
    python scripts/ask_archive.py stats <user>              # summary numbers
    python scripts/ask_archive.py search <phrase...>        # all authors

The in-chat commands (~said, ~quote, ...) answer the same questions but keep
replies Twitch-short; this CLI shows everything.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import chat_archive  # noqa: E402


def main():
    # Twitch content is full of emoji; without this, piping/redirecting output
    # on Windows (cp1252) crashes with UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = args[0].lower()

    if cmd == "said" and len(args) >= 3:
        user, phrase = args[1], " ".join(args[2:])
        total, rows = chat_archive.said(user, phrase, limit=10)
        print(f"{user} said \"{phrase}\": {total} time(s)")
        for sent_at, channel, content in rows:
            print(f"  [{sent_at}] #{channel}: {content}")
        if total == 0:
            near = chat_archive.nearest_author_lines(user, phrase, limit=5)
            if near:
                print("Closest normalized matches:")
                for score, sent_at, channel, content in near:
                    print(f"  {score:.1%} [{sent_at}] #{channel}: {content}")
    elif cmd == "near" and len(args) >= 3:
        user, phrase = args[1], " ".join(args[2:])
        near = chat_archive.nearest_author_lines(user, phrase, limit=10, min_score=0.65)
        print(f"Closest lines by {user} to \"{phrase}\":")
        if not near:
            print("  no close matches")
        for score, sent_at, channel, content in near:
            print(f"  {score:.1%} [{sent_at}] #{channel}: {content}")
    elif cmd == "quote":
        row = chat_archive.random_quote(args[1])
        if row:
            print(f"[{row[0]}] #{row[1]}: {row[2]}")
        else:
            print(f"no messages archived for {args[1]}")
    elif cmd == "stats":
        s = chat_archive.stats(args[1])
        if not s:
            print(f"no messages archived for {args[1]}")
        else:
            print(f"{args[1]}: {s['messages']} messages, first {s['first']}, last {s['last']}, "
                  f"avg {s['avg_chars']} chars, busiest hour {s['busiest_hour']}:00")
    elif cmd == "search":
        phrase = " ".join(args[1:])
        for sent_at, channel, author, content in chat_archive.search_all(phrase, limit=25):
            print(f"[{sent_at}] #{channel} {author}: {content}")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
