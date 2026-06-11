"""Download Twitch user logs via logs.zonian.dev and optionally import them.

This is for private archive enrichment. It asks the Zonian "best logs" mirror
which instances have logs for a channel/user, downloads monthly raw text logs
from the first working instance, stores the raw files under data/unsynced/, and
can insert non-duplicate rows into the local chat archive.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import chat_archive  # noqa: E402
import config  # noqa: E402


API_BASE = "https://logs.zonian.dev"
USER_AGENT = "NoLifeChatter private archive importer/1.0"
RAW_LINE_RE = re.compile(
    r"^\[(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2})\]\s+"
    r"#?(?P<channel>\S+)\s+(?P<author>[^:]+):\s?(?P<content>.*)$"
)
# Globally-common Twitch bots, safe to ship publicly. Channel-specific bots and
# your own bot account belong in config's EXCLUDE_USERS (gitignored), which is
# unioned in at runtime — that keeps personal account names out of this repo.
DEFAULT_NOISE_USERS = {
    "automod",
    "nightbot",
    "streamelements",
    "streamlabs",
    "supibot",
    "moobot",
    "fossabot",
}
try:  # extend with the user's private exclude list, if configured
    import config as _cfg
    DEFAULT_NOISE_USERS = DEFAULT_NOISE_USERS | {u.lower() for u in getattr(_cfg, "EXCLUDE_USERS", set())}
except Exception:
    pass


@dataclass(frozen=True)
class ParsedLine:
    channel: str
    author: str
    sent_at: str
    content: str


def _split_users(value: str) -> list[str]:
    return [
        part.strip().lstrip("@")
        for part in re.split(r"[\s,;]+", value or "")
        if part.strip().lstrip("@")
    ]


def _request_text(url: str, timeout: int = 45) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


def _request_json(url: str) -> dict:
    return json.loads(_request_text(url))


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower())
    return value or "unknown"


def _month_key(day: dict) -> tuple[int, int]:
    return int(day["year"]), int(day["month"])


def _parse_raw_month(text: str, local_tz: ZoneInfo) -> list[ParsedLine]:
    rows = []
    for raw in text.splitlines():
        match = RAW_LINE_RE.match(raw)
        if not match:
            continue
        content = match.group("content").strip()
        if not content:
            continue
        utc_dt = datetime.strptime(
            f"{match.group('date')} {match.group('time')}",
            "%Y-%m-%d %H:%M:%S",
        ).replace(tzinfo=timezone.utc)
        local_dt = utc_dt.astimezone(local_tz)
        rows.append(
            ParsedLine(
                channel=chat_archive.normalize_channel(match.group("channel")),
                author=chat_archive.normalize_author(match.group("author")),
                sent_at=local_dt.strftime("%Y-%m-%d %H:%M:%S"),
                content=content,
            )
        )
    return rows


def _instance_month_urls(instances: list[str], channel: str, user: str,
                         year: int, month: int) -> list[str]:
    chan = urllib.parse.quote(channel.lower())
    usr = urllib.parse.quote(user.lower())
    return [
        f"{base.rstrip('/')}/channel/{chan}/user/{usr}/{year}/{month}"
        for base in instances
    ]


def _download_month(instances: list[str], channel: str, user: str,
                    year: int, month: int) -> tuple[str, str] | str | None:
    errors = []
    for url in _instance_month_urls(instances, channel, user, year, month):
        try:
            text = _request_text(url)
        except urllib.error.HTTPError as exc:
            errors.append((url, f"HTTP {exc.code}: {exc.reason}", exc.code))
            continue
        except Exception as exc:
            errors.append((url, str(exc), None))
            continue
        if text.strip().startswith("<!doctype html") or text.strip().startswith("<!DOCTYPE html"):
            errors.append((url, "got HTML instead of raw logs", None))
            continue
        return url, text
    if errors and all(code == 404 for _url, _error, code in errors):
        return "missing"
    if errors:
        print("    all instances failed for month:")
        for url, error, _code in errors[:4]:
            print(f"      {url}: {error}")
    return None


def _save_raw(out_root: Path, channel: str, user: str, year: int, month: int,
              text: str, source_url: str) -> Path:
    raw_dir = out_root / "raw" / _safe_name(channel) / _safe_name(user)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{year:04d}-{month:02d}.log"
    header = (
        f"# source: {source_url}\n"
        f"# downloaded_at_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    path.write_text(header + text, encoding="utf-8", newline="\n")
    return path


def _sent_at(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _substantial_match_key(content: str, min_chars: int) -> str:
    key = chat_archive.line_match_key(content)
    if len(key) < min_chars:
        return ""
    if len(key.split()) < 3:
        return ""
    return key


def _duplicate_reason(conn, row: ParsedLine, window_hours: float, min_chars: int) -> str | None:
    exact = conn.execute(
        "SELECT 1 FROM messages "
        "WHERE channel = ? AND author = ? AND sent_at = ? AND content = ? "
        "LIMIT 1",
        (row.channel, row.author, row.sent_at, row.content),
    ).fetchone()
    if exact:
        return "exact"

    if window_hours <= 0:
        return None
    key = _substantial_match_key(row.content, min_chars)
    if not key:
        return None
    dt = _sent_at(row.sent_at)
    if not dt:
        return None

    start = (dt - timedelta(hours=window_hours)).strftime("%Y-%m-%d %H:%M:%S")
    end = (dt + timedelta(hours=window_hours)).strftime("%Y-%m-%d %H:%M:%S")
    candidates = conn.execute(
        "SELECT content FROM messages "
        "WHERE channel = ? AND author = ? AND sent_at BETWEEN ? AND ?",
        (row.channel, row.author, start, end),
    ).fetchall()
    for (content,) in candidates:
        if _substantial_match_key(content, min_chars) == key:
            return "near_time"
    return None


def _import_rows(rows: list[ParsedLine], src_path: Path,
                 dedupe_window_hours: float, dedupe_min_chars: int) -> Counter:
    counts = Counter()
    if not rows:
        return counts
    conn = chat_archive.connect()
    with conn:
        for row in rows:
            duplicate = _duplicate_reason(conn, row, dedupe_window_hours, dedupe_min_chars)
            if duplicate:
                counts[f"skipped_{duplicate}"] += 1
                continue
            conn.execute(
                "INSERT INTO messages (channel, author, sent_at, content, source, src_path) "
                "VALUES (?, ?, ?, ?, 'zonian', ?)",
                (row.channel, row.author, row.sent_at, row.content, str(src_path)),
            )
            counts["inserted"] += 1
    return counts


def _summary_path(out_root: Path, channel: str) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root / f"{_safe_name(channel)}_download_summary.json"


def _is_noise_user(author: str, canonical: str, excluded: set[str]) -> bool:
    return (
        author in excluded
        or canonical in excluded
        or author in DEFAULT_NOISE_USERS
        or canonical in DEFAULT_NOISE_USERS
        or author.endswith("bot")
        or canonical.endswith("bot")
    )


def archive_users(channel: str, min_messages: int, include_excluded: bool) -> tuple[list[str], list[dict]]:
    """Return locally-known authors for a channel, ordered by archive count."""
    conn = chat_archive.connect()
    normalized_channel = chat_archive.normalize_channel(channel)
    rows = conn.execute(
        """
        SELECT author, COUNT(*) AS n, MIN(sent_at), MAX(sent_at)
        FROM messages
        WHERE channel = ?
        GROUP BY author
        HAVING n >= ?
        ORDER BY n DESC, author
        """,
        (normalized_channel, min_messages),
    ).fetchall()

    users = []
    skipped = []
    excluded = set(getattr(config, "EXCLUDE_USERS", set()))
    for author, count, _first_seen, _last_seen in rows:
        author = chat_archive.normalize(author)
        canonical = chat_archive.normalize_author(author)
        if not include_excluded and _is_noise_user(author, canonical, excluded):
            skipped.append(
                {
                    "user": author,
                    "count": count,
                    "reason": "bot/noise account",
                }
            )
            continue
        users.append(author)

    users = list(dict.fromkeys(users))
    print(
        f"Loaded {len(users)} users from local #{normalized_channel} archive "
        f"(min {min_messages:,} messages"
        f"{', including excluded' if include_excluded else ', excluded bot/noise accounts skipped'})."
    )
    if users:
        preview = ", ".join(users[:20])
        suffix = " ..." if len(users) > 20 else ""
        print(f"Archive users: {preview}{suffix}")
    if skipped:
        preview = ", ".join(f"{item['user']} ({item['count']:,})" for item in skipped[:12])
        suffix = " ..." if len(skipped) > 12 else ""
        print(f"Skipped excluded: {preview}{suffix}")
    return users, skipped


def download_user(channel: str, user: str, out_root: Path, local_tz: ZoneInfo,
                  import_archive: bool, limit_months: int | None = None,
                  sleep_s: float = 0.25, dedupe_window_hours: float = 12.0,
                  dedupe_min_chars: int = 16) -> dict:
    api_url = f"{API_BASE}/api/{urllib.parse.quote(channel)}/{urllib.parse.quote(user)}"
    print(f"\n== {channel}/{user} ==")
    print(f"API: {api_url}")
    try:
        info = _request_json(api_url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print("  no logs reported by API")
            return {
                "user": user,
                "error": "api_not_found",
                "months_reported": 0,
                "months_downloaded": 0,
                "months_missing": 0,
                "rows_parsed": 0,
                "rows_inserted": 0,
                "failures": [],
            }
        raise
    if info.get("error"):
        print(f"  API error: {info.get('error')}")
        return {"user": user, "error": info.get("error")}

    req = info.get("request", {})
    resolved_channel = (req.get("channel") or {}).get("login") or channel
    resolved_user = (req.get("user") or {}).get("login") or user
    logged = info.get("loggedData") or {}
    months = sorted({_month_key(day) for day in logged.get("list", [])})
    if limit_months:
        months = months[-limit_months:]
    instances = (info.get("userLogs") or {}).get("instances") or []
    if not months:
        print("  no logged days reported")
        return {"user": user, "resolved_user": resolved_user, "months": 0, "downloaded": 0}
    if not instances:
        print("  no instances reported")
        return {"user": user, "resolved_user": resolved_user, "months": len(months), "downloaded": 0}

    print(
        f"  resolved: #{resolved_channel}/{resolved_user}; "
        f"{len(months)} months from {months[0][0]}-{months[0][1]:02d} "
        f"to {months[-1][0]}-{months[-1][1]:02d}; {len(instances)} instances"
    )

    total_rows = 0
    total_inserted = 0
    total_skipped_exact = 0
    total_skipped_near = 0
    downloaded = 0
    missing = 0
    failures = []
    for i, (year, month) in enumerate(months, 1):
        print(f"  [{i}/{len(months)}] {year:04d}-{month:02d}", end="", flush=True)
        result = _download_month(instances, resolved_channel, resolved_user, year, month)
        if result == "missing":
            missing += 1
            print(" no logs")
            continue
        if not result:
            print(" failed")
            failures.append(f"{year:04d}-{month:02d}")
            continue
        source_url, text = result
        raw_path = _save_raw(out_root, resolved_channel, resolved_user, year, month, text, source_url)
        rows = _parse_raw_month(text, local_tz)
        import_counts = (
            _import_rows(rows, raw_path, dedupe_window_hours, dedupe_min_chars)
            if import_archive else Counter()
        )
        inserted = import_counts["inserted"]
        total_rows += len(rows)
        total_inserted += inserted
        total_skipped_exact += import_counts["skipped_exact"]
        total_skipped_near += import_counts["skipped_near_time"]
        downloaded += 1
        if import_archive:
            skipped = import_counts["skipped_exact"] + import_counts["skipped_near_time"]
            print(f" {len(rows):,} rows, +{inserted:,} new, {skipped:,} duplicate")
        else:
            print(f" {len(rows):,} rows saved")
        if sleep_s > 0:
            time.sleep(sleep_s)

    return {
        "user": user,
        "resolved_user": resolved_user,
        "resolved_channel": resolved_channel,
        "months_reported": len(months),
        "months_downloaded": downloaded,
        "months_missing": missing,
        "rows_parsed": total_rows,
        "rows_inserted": total_inserted,
        "rows_skipped_exact": total_skipped_exact,
        "rows_skipped_near_time": total_skipped_near,
        "failures": failures,
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channel", required=True, help="Twitch channel login to pull logs from")
    ap.add_argument("--users", default="", help="comma/space-separated usernames")
    ap.add_argument("--users-file", default="", help="optional text file, one user per line")
    ap.add_argument("--from-archive", action="store_true",
                    help="add locally-known users from chat_archive.db")
    ap.add_argument("--users-from-channel", default="",
                    help="archive channel to select users from; defaults to --channel")
    ap.add_argument("--min-archive-messages", type=int, default=25,
                    help="minimum local messages required with --from-archive")
    ap.add_argument("--include-excluded", action="store_true",
                    help="include users from config.persona.exclude_users when using --from-archive")
    ap.add_argument("--list-users-only", action="store_true",
                    help="print users selected from the local archive and exit")
    ap.add_argument("--out-root", default="data/unsynced/external_logs/zonian")
    ap.add_argument("--import-archive", action="store_true",
                    help="insert non-duplicate rows into data/unsynced/chat_archive.db")
    ap.add_argument("--dedupe-window-hours", type=float, default=12.0,
                    help="when importing, skip substantial same-author same-channel "
                         "lines with matching normalized text within +/- this many hours")
    ap.add_argument("--dedupe-min-chars", type=int, default=16,
                    help="minimum normalized text length for timezone-tolerant dedupe")
    ap.add_argument("--local-tz", default="America/New_York")
    ap.add_argument("--limit-months", type=int, default=0,
                    help="debug/test mode: only newest N months")
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    users = _split_users(args.users)
    archive_source = {"enabled": False}
    if args.from_archive:
        source_channel = args.users_from_channel or args.channel
        archive_selected, archive_skipped = archive_users(
            source_channel,
            max(1, args.min_archive_messages),
            include_excluded=args.include_excluded,
        )
        users.extend(archive_selected)
        archive_source = {
            "enabled": True,
            "channel": source_channel,
            "min_messages": max(1, args.min_archive_messages),
            "include_excluded": args.include_excluded,
            "selected": archive_selected,
            "skipped": archive_skipped,
        }
    if args.users_file:
        path = Path(args.users_file)
        if path.exists():
            users.extend(
                line.strip().lstrip("@")
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
    users = list(dict.fromkeys(user.lower() for user in users if user))
    if args.list_users_only:
        if not users:
            raise SystemExit("No users selected.")
        for user in users:
            print(user)
        return
    if not users:
        raise SystemExit("No users given. Pass --users, --users-file, or --from-archive.")

    out_root = Path(args.out_root)
    local_tz = ZoneInfo(args.local_tz)
    summaries = []
    totals = Counter()
    for user in users:
        try:
            summary = download_user(
                args.channel,
                user,
                out_root,
                local_tz,
                import_archive=args.import_archive,
                limit_months=args.limit_months or None,
                sleep_s=args.sleep,
                dedupe_window_hours=max(0.0, args.dedupe_window_hours),
                dedupe_min_chars=max(1, args.dedupe_min_chars),
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"\n!! {args.channel}/{user} failed: {exc}")
            summary = {"user": user, "error": str(exc)}
        summaries.append(summary)
        for key in (
            "months_downloaded", "months_missing", "rows_parsed", "rows_inserted",
            "rows_skipped_exact", "rows_skipped_near_time",
        ):
            totals[key] += int(summary.get(key) or 0)

    summary_doc = {
        "downloaded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "channel": args.channel,
        "import_archive": args.import_archive,
        "out_root": str(out_root),
        "archive_source": archive_source,
        "dedupe": {
            "window_hours": max(0.0, args.dedupe_window_hours),
            "min_chars": max(1, args.dedupe_min_chars),
            "near_time_rule": "same channel + same author + same normalized substantial text within window",
        },
        "totals": dict(totals),
        "users": summaries,
    }
    summary_path = _summary_path(out_root, args.channel)
    summary_path.write_text(json.dumps(summary_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDone.")
    print(f"Summary: {summary_path}")
    print(
        f"Months downloaded: {totals['months_downloaded']:,}; "
        f"months with no logs: {totals['months_missing']:,}; "
        f"rows parsed: {totals['rows_parsed']:,}; "
        f"new archive rows: {totals['rows_inserted']:,}; "
        f"duplicates skipped: "
        f"{(totals['rows_skipped_exact'] + totals['rows_skipped_near_time']):,}"
    )


if __name__ == "__main__":
    main()
