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
import math
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


TIMESTAMP_IN_TEXT_RE = re.compile(r"\[\d{2}:\d{2}:\d{2}\]")


def _usable_output(content: str, min_words: int, max_chars: int, allow_urls: bool) -> bool:
    content = (content or "").strip()
    if not content or content.startswith(config.PREFIX):
        return False
    # Bot-command lines ($gpt/$remind/!so) are real chat habits but terrible
    # training targets: a persona that learned them spams commands instead of
    # talking. (v2 lesson — addressing people is "@name ...", not a command.)
    if re.match(r"^[~$!][A-Za-z]{2,}", content):
        return False
    if len(content) > max_chars:
        return False
    if len(content.split()) < min_words:
        return False
    if not allow_urls and URL_RE.search(content):
        return False
    # A bracketed timestamp inside the text is a log-parse artifact (a merged
    # or malformed line) — never a real chat message. Don't train on it.
    if TIMESTAMP_IN_TEXT_RE.search(content):
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


def _example_from_candidate(cand: dict) -> dict:
    persona = cand["persona"]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"<persona={persona}>\n"
                    f"Recent chat in #{cand['channel']}:\n{cand['context']}\n\n"
                    f"Write {persona}'s next chat message."
                ),
            },
            {"role": "assistant", "content": cand["content"]},
        ],
        "metadata": {
            "persona": persona,
            "channel": cand["channel"],
            "sent_at": cand["sent_at"],
            "message_id": cand["id"],
        },
    }


def _reservoir_add(bucket: list, count: int, item: dict, cap: int,
                   rng: random.Random) -> None:
    """Unbiased reservoir sampling of `item` into `bucket` (<= cap entries)."""
    if len(bucket) < cap:
        bucket.append(item)
    else:
        j = rng.randrange(count)
        if j < cap:
            bucket[j] = item


_TOKEN_RE = re.compile(r"[a-z0-9']{2,}")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _distinctiveness_models(candidates: dict, min_z: float) -> dict:
    """Per-author {token: z} signature maps via the log-odds-ratio with an
    informative prior (Monroe, Colaresi & Quinn 2008, "Fightin' Words"):
    each author vs the rest of the selected group, add-0.5 smoothed, z-scored.

    z>0 = the author says the word more than the group average; high z is their
    signature vocabulary/emotes/topics. A message's distinctiveness is the sum
    of its signature tokens' z — the tail of "characteristically them" that the
    uniform sample drowned in filler. This is the fix to "what are we
    minimizing": train on what makes them *them*, not their average "lol".
    """
    global_counts = Counter()
    author_counts = {}
    for persona, items in candidates.items():
        c = Counter()
        for it in items:
            c.update(_tokens(it["content"]))
        author_counts[persona] = c
        global_counts.update(c)
    total = sum(global_counts.values())
    models = {}
    for persona, ac in author_counts.items():
        n_a = sum(ac.values())
        n_b = total - n_a
        zmap = {}
        if n_a > 0 and n_b > 0:
            for w, y_aw in ac.items():
                y_bw = global_counts[w] - y_aw
                num_a, den_a = y_aw + 0.5, n_a - y_aw + 0.5
                num_b, den_b = y_bw + 0.5, n_b - y_bw + 0.5
                if den_a <= 0 or den_b <= 0:
                    continue
                delta = math.log(num_a / den_a) - math.log(num_b / den_b)
                z = delta / math.sqrt(1.0 / num_a + 1.0 / num_b)
                if z >= min_z:
                    zmap[w] = z
        models[persona] = zmap
    return models


def _distinctiveness(content: str, zmap: dict) -> float:
    return sum(zmap.get(t, 0.0) for t in set(_tokens(content)))


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

    # Collect candidate examples per author (reservoir-capped so a huge channel
    # can't blow up memory); distinctiveness selection happens after.
    collect_cap = max(args.collect_cap_per_author, args.max_examples_per_author)
    candidates = defaultdict(list)
    collected = Counter()
    skipped = Counter()
    context_by_channel = defaultdict(lambda: deque(maxlen=args.context))

    for msg_id, channel, author, sent_at, content in rows:
        cleaned = _clean_line(content, args.max_output_chars)
        persona = chat_archive.normalize_author(author)
        current_dt = _dt(sent_at)
        context = []
        for ctx in context_by_channel[channel]:
            if current_dt:
                ctx_dt = _dt(ctx["sent_at"])
                if ctx_dt and (current_dt - ctx_dt).total_seconds() > args.within_minutes * 60:
                    continue
            context.append(ctx)

        if persona not in authors:
            skipped["author"] += 1
        elif len(context) < args.min_context:
            skipped["context"] += 1
        elif not _usable_output(cleaned, args.min_words, args.max_output_chars, args.allow_urls):
            skipped["output"] += 1
        else:
            context_str = _format_context(context[-args.context:], args.max_context_chars)
            if context_str:
                collected[persona] += 1
                _reservoir_add(
                    candidates[persona], collected[persona],
                    {"persona": persona, "channel": channel, "sent_at": sent_at,
                     "id": msg_id, "content": cleaned, "context": context_str},
                    collect_cap, rng,
                )

        if content and not content.lstrip().startswith(config.PREFIX):
            context_by_channel[channel].append(
                {"author": author, "sent_at": sent_at, "content": cleaned}
            )

    # Select per author: most-distinctive first, with a random tail for coverage.
    models = (
        _distinctiveness_models(candidates, args.distinctiveness_min_z)
        if args.distinctiveness_ratio > 0 else {}
    )
    keep = args.max_examples_per_author
    signature_kept = Counter()
    chosen = []
    for persona, items in candidates.items():
        if models.get(persona):
            scored = sorted(
                items, key=lambda it: _distinctiveness(it["content"], models[persona]),
                reverse=True,
            )
            n_sig = min(len(scored), int(keep * args.distinctiveness_ratio))
            signature_kept[persona] = n_sig
            picked = scored[:n_sig]
            rest = scored[n_sig:]
            rng.shuffle(rest)
            picked += rest[: max(0, keep - len(picked))]
        else:
            picked = list(items)
            rng.shuffle(picked)
            picked = picked[:keep]
        chosen.extend(picked)

    examples = [_example_from_candidate(c) for c in chosen]
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
    top_signatures = {
        persona: [w for w, _ in sorted(zmap.items(), key=lambda kv: -kv[1])[:12]]
        for persona, zmap in models.items() if zmap
    }
    return {
        "train": len(train),
        "validation": len(val),
        "total": len(examples),
        "authors": len(per_author),
        "channels": channels,
        "distinctiveness_ratio": args.distinctiveness_ratio,
        "signature_kept": dict(signature_kept.most_common(20)),
        "top_signatures": top_signatures,
        "top_authors": per_author.most_common(20),
        "collected_candidates": dict(collected.most_common(20)),
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
    ap.add_argument("--distinctiveness-ratio", type=float, default=0.6,
                    help="fraction of each author's examples picked by signature "
                         "distinctiveness (rest random for coverage); 0 = uniform")
    ap.add_argument("--distinctiveness-min-z", type=float, default=1.5,
                    help="min log-odds z-score for a token to count as signature")
    ap.add_argument("--collect-cap-per-author", type=int, default=40000,
                    help="max candidates held per author before selection (memory bound)")
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
