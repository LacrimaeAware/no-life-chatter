"""Export persona supervised fine-tuning examples from the chat archive.

The output is private training data and defaults to data/unsynced/fine_tune/,
which is gitignored. Each JSONL row uses OpenAI-style chat messages:

    system: stable task instruction
    user: <persona=name> plus recent chat context
    assistant: the real next message that persona wrote

This script does not train a model. It prepares the dataset for a rented-GPU
LoRA/QLoRA run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils import chat_archive  # noqa: E402

SYSTEM_PROMPT = (
    "You are a local Twitch chat persona model. The active persona is written "
    "as <persona=name>. Given recent chat, write the next single Twitch chat "
    "message from that persona. Match their style, vocabulary, casing, emotes, "
    "punctuation, humor, and usual length. Do not explain. Output one chat "
    "message only."
)

URL_RE = re.compile(r"\b(?:https?://)?\S+\.\S+\b", re.IGNORECASE)


def _dt(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _split_csv(value: str) -> list[str]:
    return [part.strip().lower().lstrip("#@") for part in (value or "").split(",") if part.strip()]


def _parse_aliases(value: str) -> dict[str, str]:
    aliases = {}
    for part in (value or "").split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise SystemExit(f"Invalid alias {part!r}; use alias=canonical")
        alias, canonical = part.split("=", 1)
        alias = chat_archive.normalize_author(alias)
        canonical = chat_archive.normalize_author(canonical)
        if alias and canonical:
            aliases[alias] = canonical
    return aliases


def _apply_export_aliases(value: str) -> None:
    for alias, canonical in _parse_aliases(value).items():
        chat_archive.USER_ALIASES[alias] = canonical


def _clean_line(value: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    if len(value) > max_chars:
        value = value[:max_chars].rstrip()
    return value


def _usable_output(content: str, min_words: int, max_chars: int, allow_urls: bool) -> bool:
    content = (content or "").strip()
    if not content or content.startswith(config.PREFIX):
        return False
    if len(content) > max_chars:
        return False
    if len(content.split()) < min_words:
        return False
    if not allow_urls and URL_RE.search(content):
        return False
    return True


def _format_context(rows, max_chars: int) -> str:
    lines = []
    for row in rows:
        content = _clean_line(row["content"], max_chars)
        if content and not content.startswith(config.PREFIX):
            lines.append(f'{row["author"]}: {content}')
    return "\n".join(lines)


def _author_totals(conn) -> Counter:
    totals = Counter()
    for author, n in conn.execute("SELECT author, COUNT(*) FROM messages GROUP BY author"):
        totals[chat_archive.normalize_author(author)] += n
    return totals


def _candidate_authors(conn, requested: list[str], min_author_messages: int,
                       extra_exclude: list[str]) -> set[str]:
    excluded = {
        chat_archive.normalize_author(user)
        for user in (set(config.EXCLUDE_USERS) | set(extra_exclude))
    }
    if requested:
        return {chat_archive.normalize_author(author) for author in requested} - excluded
    totals = _author_totals(conn)
    return {
        author for author, n in totals.items()
        if n >= min_author_messages and author not in excluded
    }


def _channels(conn, requested: list[str]) -> list[str]:
    if requested:
        return [chat_archive.normalize_channel(channel) for channel in requested]
    configured = [chat_archive.normalize_channel(channel) for channel in config.CHANNELS]
    if configured:
        return configured
    return [row[0] for row in conn.execute("SELECT DISTINCT channel FROM messages")]


def _make_example(row, context_rows, max_context_chars: int) -> dict | None:
    context = _format_context(context_rows, max_context_chars)
    if not context:
        return None
    persona = chat_archive.normalize_author(row["author"])
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"<persona={persona}>\n"
                    f"Recent chat in #{row['channel']}:\n{context}\n\n"
                    f"Write {persona}'s next chat message."
                ),
            },
            {"role": "assistant", "content": row["content"]},
        ],
        "metadata": {
            "persona": persona,
            "channel": row["channel"],
            "sent_at": row["sent_at"],
            "message_id": row["id"],
        },
    }


def _reservoir_add(reservoir: dict, seen: Counter, example: dict,
                   max_examples: int, rng: random.Random) -> None:
    persona = example["metadata"]["persona"]
    seen[persona] += 1
    bucket = reservoir[persona]
    if len(bucket) < max_examples:
        bucket.append(example)
        return
    j = rng.randrange(seen[persona])
    if j < max_examples:
        bucket[j] = example


def export(args) -> dict:
    rng = random.Random(args.seed)
    conn = chat_archive.connect()
    conn.row_factory = None

    authors = _candidate_authors(
        conn, _split_csv(args.authors), args.min_author_messages, _split_csv(args.exclude_users)
    )
    channels = _channels(conn, _split_csv(args.channels))
    if not authors:
        raise SystemExit("No eligible authors. Lower --min-author-messages or pass --authors.")
    if not channels:
        raise SystemExit("No channels selected.")

    channel_placeholders = ",".join("?" for _ in channels)
    rows = conn.execute(
        "SELECT id, channel, author, sent_at, content FROM messages "
        f"WHERE channel IN ({channel_placeholders}) ORDER BY channel, id",
        channels,
    )

    reservoir = defaultdict(list)
    seen = Counter()
    skipped = Counter()
    context_by_channel = defaultdict(lambda: deque(maxlen=args.context))

    for msg_id, channel, author, sent_at, content in rows:
        row = {
            "id": msg_id,
            "channel": channel,
            "author": author,
            "sent_at": sent_at,
            "content": _clean_line(content, args.max_output_chars),
        }
        persona = chat_archive.normalize_author(author)
        current_dt = _dt(sent_at)
        raw_context = list(context_by_channel[channel])
        context = []
        for ctx in raw_context:
            if current_dt:
                ctx_dt = _dt(ctx["sent_at"])
                if ctx_dt and (current_dt - ctx_dt).total_seconds() > args.within_minutes * 60:
                    continue
            context.append(ctx)

        if persona not in authors:
            skipped["author"] += 1
        elif len(context) < args.min_context:
            skipped["context"] += 1
        elif not _usable_output(row["content"], args.min_words, args.max_output_chars, args.allow_urls):
            skipped["output"] += 1
        else:
            example = _make_example(row, context[-args.context:], args.max_context_chars)
            if example:
                _reservoir_add(reservoir, seen, example, args.max_examples_per_author, rng)

        if content and not content.lstrip().startswith(config.PREFIX):
            context_by_channel[channel].append(row)

    examples = [example for bucket in reservoir.values() for example in bucket]
    rng.shuffle(examples)

    train, val = [], []
    for example in examples:
        (val if rng.random() < args.val_ratio else train).append(example)

    out = Path(args.out)
    val_out = Path(args.val_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for example in train:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
    with val_out.open("w", encoding="utf-8", newline="\n") as fh:
        for example in val:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    per_author = Counter(example["metadata"]["persona"] for example in examples)
    return {
        "train": len(train),
        "validation": len(val),
        "total": len(examples),
        "authors": len(per_author),
        "channels": channels,
        "top_authors": per_author.most_common(20),
        "seen_candidates": dict(seen.most_common(20)),
        "skipped": dict(skipped),
        "out": str(out),
        "val_out": str(val_out),
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/unsynced/fine_tune/persona_train.jsonl")
    ap.add_argument("--val-out", default="data/unsynced/fine_tune/persona_val.jsonl")
    ap.add_argument("--authors", default="",
                    help="comma-separated personas to export; default = all eligible authors")
    ap.add_argument("--channels", default="",
                    help="comma-separated channels; default = bot.channels from config.toml")
    ap.add_argument("--exclude-users", default="", help="extra comma-separated users to skip")
    ap.add_argument("--user-aliases", default="",
                    help="comma-separated alias=canonical author merges for this export")
    ap.add_argument("--min-author-messages", type=int, default=500)
    ap.add_argument("--max-examples-per-author", type=int, default=8000)
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--min-context", type=int, default=3)
    ap.add_argument("--within-minutes", type=int, default=20)
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--max-output-chars", type=int, default=240)
    ap.add_argument("--max-context-chars", type=int, default=240)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--allow-urls", action="store_true")
    args = ap.parse_args()
    _apply_export_aliases(args.user_aliases)

    summary = export(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
