"""Ingest Chatterino 2 chat logs into the searchable archive.

    python scripts/ingest_chatterino.py [logs-root] [--channels a,b,c] [--since YYYY-MM-DD]

logs-root is Chatterino's per-channel log directory, e.g.
%AppData%/Chatterino2/Logs/Twitch/Channels (one subdirectory per channel,
files named <channel>-YYYY-MM-DD.log). It can also be set once as
archive.chatterino_logs in config.toml and omitted here.

Safe to re-run any time: files already ingested (same mtime) are skipped, and
a file that grew since last run (today's live log) has its day re-imported
cleanly. See docs/CHAT_ARCHIVE.md for the format spec this parser implements.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils.chat_archive import FILE_DATE_RE, connect, ingest_file, normalize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs_root", nargs="?", default=config.ARCHIVE_CHATTERINO_LOGS,
                    help="Chatterino Channels directory (default: archive.chatterino_logs from config.toml)")
    ap.add_argument("--channels", default="",
                    help="comma-separated channels to ingest (default: every subdirectory)")
    ap.add_argument("--since", default="",
                    help="only ingest files dated YYYY-MM-DD or later")
    args = ap.parse_args()

    if not args.logs_root:
        ap.error("no logs-root given and archive.chatterino_logs is not set in config.toml")
    root = os.path.abspath(args.logs_root)
    if not os.path.isdir(root):
        ap.error(f"not a directory: {root}")

    wanted = {normalize(c) for c in args.channels.split(",") if c.strip()}

    totals = {"files": 0, "skipped": 0, "chat": 0, "system": 0, "header": 0,
              "empty": 0, "modaction": 0}
    for entry in sorted(os.listdir(root)):
        chan_dir = os.path.join(root, entry)
        if not os.path.isdir(chan_dir):
            continue
        channel = normalize(entry)
        if wanted and channel not in wanted:
            continue
        chan_rows = 0
        for fname in sorted(os.listdir(chan_dir)):
            m = FILE_DATE_RE.search(fname)
            if not m:
                continue
            date = m.group(1)
            if args.since and date < args.since:
                continue
            counts = ingest_file(os.path.join(chan_dir, fname), channel, date)
            totals["files"] += 1
            if counts.get("skipped"):
                totals["skipped"] += 1
                continue
            for k in ("chat", "system", "header", "empty", "modaction"):
                totals[k] += counts[k]
            chan_rows += counts["chat"]
        if chan_rows:
            logging.info(f"{entry}: +{chan_rows} messages (-> channel '{channel}')")

    logging.info(
        f"Done. files={totals['files']} (skipped {totals['skipped']} unchanged) | "
        f"messages={totals['chat']} system={totals['system']} headers={totals['header']} "
        f"empty={totals['empty']} modactions={totals['modaction']}"
    )
    conn = connect()
    n, authors, channels = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT author), COUNT(DISTINCT channel) FROM messages"
    ).fetchone()
    logging.info(f"Archive now: {n} messages, {authors} authors, {channels} channels "
                 f"({config.ARCHIVE_DB})")


if __name__ == "__main__":
    main()
