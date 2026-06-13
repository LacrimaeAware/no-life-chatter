"""Held-out reply eval for the persona engine.

This is the benchmark the persona docs kept asking for: find real moments
where a chatter spoke, hide their actual message, feed only the preceding chat
context to the normal persona generator, then score the generated line against
the real held-out reply.

Default mode only samples cases and writes private JSONL, so it is safe to run
without LM Studio:

    python scripts/eval_heldout_replies.py --sample-only

Run the actual local-model eval when LM Studio is up:

    python scripts/eval_heldout_replies.py --generate --authors user1,user2

Outputs default to data/unsynced/persona_heldout_eval.* and must stay private.
"""

from __future__ import annotations

import argparse
import asyncio
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
from services import llm  # noqa: E402
from utils import chat_archive, message_quality, persona_llm  # noqa: E402


DEFAULT_CASES = Path("data/unsynced/persona_heldout_eval_cases.jsonl")
DEFAULT_RESULTS = Path("data/unsynced/persona_heldout_eval_results.jsonl")
DEFAULT_REPORT = Path("data/unsynced/persona_heldout_eval_report.md")


def _split_csv(value: str) -> list[str]:
    return [part.strip().lower().lstrip("#@") for part in (value or "").split(",") if part.strip()]


def _dt(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _clean_line(value: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value[:max_chars].rstrip() if len(value) > max_chars else value


def _bot_command_like(value: str) -> bool:
    return bool(re.match(r"^[~$!<][A-Za-z]{2,}", (value or "").lstrip()))


def _context_rows_for_prompt(rows: list[dict]) -> list[tuple[str, str, str]]:
    return [(row["sent_at"], row["author"], row["content"]) for row in rows]


def _reservoir_add(bucket: list, seen_count: int, item: dict, cap: int,
                   rng: random.Random) -> None:
    if len(bucket) < cap:
        bucket.append(item)
        return
    j = rng.randrange(seen_count)
    if j < cap:
        bucket[j] = item


def _eligible_authors(conn, authors: list[str], channels: list[str],
                      min_messages: int, include_bots: bool) -> set[str]:
    if authors:
        return {chat_archive.normalize_author(author) for author in authors}

    channel_ph, channel_params = chat_archive._in_clause(channels)
    rows = conn.execute(
        "SELECT author, COUNT(*) FROM messages "
        f"WHERE channel IN ({channel_ph}) AND source IN ('chatterino', 'live') "
        "GROUP BY author",
        channel_params,
    ).fetchall()
    counts = Counter()
    for author, count in rows:
        canon = chat_archive.normalize_author(author)
        if not include_bots and chat_archive._is_noise_author(canon):
            continue
        counts[canon] += count
    return {author for author, count in counts.items() if count >= min_messages}


def _selected_channels(conn, requested: list[str]) -> list[str]:
    if requested:
        return [chat_archive.normalize_channel(channel) for channel in requested]
    configured = [chat_archive.normalize_channel(channel) for channel in config.CHANNELS]
    if configured:
        return configured
    return [
        row[0] for row in conn.execute(
            "SELECT channel FROM messages GROUP BY channel ORDER BY COUNT(*) DESC LIMIT 8"
        )
    ]


def sample_cases(args) -> list[dict]:
    rng = random.Random(args.seed)
    conn = chat_archive.connect()
    channels = _selected_channels(conn, _split_csv(args.channels))
    if not channels:
        raise SystemExit("No channels available for held-out sampling.")

    authors = _eligible_authors(
        conn,
        _split_csv(args.authors),
        channels,
        args.min_author_messages,
        args.include_bots,
    )
    if not authors:
        raise SystemExit("No eligible authors. Lower --min-author-messages or pass --authors.")

    channel_ph, channel_params = chat_archive._in_clause(channels)
    source_sql = "" if args.include_mirror else "AND source IN ('chatterino', 'live') "
    since_sql = "AND sent_at >= ? " if args.since else ""
    limit_sql = "LIMIT ?" if args.scan_limit > 0 else ""
    params = list(channel_params)
    if args.since:
        params.append(args.since)
    if args.scan_limit > 0:
        params.append(args.scan_limit)

    rows = conn.execute(
        "SELECT id, channel, author, sent_at, content, source FROM messages "
        f"WHERE channel IN ({channel_ph}) {source_sql}{since_sql}"
        "ORDER BY channel, sent_at, id "
        f"{limit_sql}",
        params,
    )

    context_by_channel = defaultdict(lambda: deque(maxlen=args.context))
    buckets = defaultdict(list)
    seen_by_author = Counter()
    skipped = Counter()

    for msg_id, channel, author, sent_at, content, source in rows:
        channel = chat_archive.normalize_channel(channel)
        persona = chat_archive.normalize_author(author)
        current_dt = _dt(sent_at)
        context = []
        for ctx in context_by_channel[channel]:
            if current_dt:
                ctx_dt = _dt(ctx["sent_at"])
                if ctx_dt and (current_dt - ctx_dt).total_seconds() > args.within_minutes * 60:
                    continue
            context.append(ctx)

        cleaned = _clean_line(content, args.max_target_chars)
        if persona not in authors:
            skipped["author"] += 1
        elif len(context) < args.min_context:
            skipped["context"] += 1
        elif not message_quality.usable_for_persona_exemplar(
            cleaned, max_chars=args.max_target_chars
        ):
            skipped["target_quality"] += 1
        else:
            seen_by_author[persona] += 1
            case = {
                "id": f"heldout-{msg_id}",
                "persona": persona,
                "channel": channel,
                "target": {
                    "id": msg_id,
                    "sent_at": sent_at,
                    "author": persona,
                    "content": cleaned,
                    "source": source,
                },
                "context": context[-args.context:],
            }
            _reservoir_add(
                buckets[persona], seen_by_author[persona], case,
                args.per_author, rng,
            )

        ctx_content = _clean_line(content, args.max_context_chars)
        if (
            ctx_content
            and not chat_archive._is_noise_author(persona)
            and not _bot_command_like(ctx_content)
            and message_quality.usable_for_snippet_context(
                ctx_content, max_chars=args.max_context_chars
            )
        ):
            context_by_channel[channel].append({
                "sent_at": sent_at,
                "author": persona,
                "content": ctx_content,
            })

    cases = [case for bucket in buckets.values() for case in bucket]
    rng.shuffle(cases)
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    for case in cases:
        case["sampling"] = {
            "context": args.context,
            "within_minutes": args.within_minutes,
            "source_policy": "all" if args.include_mirror else "chatterino_live_only",
        }
    print(
        f"sampled {len(cases)} held-out cases from {len(buckets)} authors "
        f"({dict(skipped.most_common(5))})"
    )
    return cases


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _terms(text: str) -> set[str]:
    return set(chat_archive.query_terms(text, max_terms=24))


def score_pair(target: str, generated: str, classifier=None, persona: str = "") -> dict:
    target_terms = _terms(target)
    generated_terms = _terms(generated)
    overlap = len(target_terms & generated_terms)
    target_len = max(1, len(target.split()))
    generated_len = max(1, len(generated.split()))
    length_ratio = min(target_len, generated_len) / max(target_len, generated_len)
    out = {
        "line_similarity": round(chat_archive.line_similarity(target, generated), 4),
        "term_recall": round(overlap / max(1, len(target_terms)), 4),
        "term_precision": round(overlap / max(1, len(generated_terms)), 4),
        "length_ratio": round(length_ratio, 4),
        "target_terms": sorted(target_terms),
        "generated_terms": sorted(generated_terms),
    }
    out["shape_score"] = round(
        (out["line_similarity"] * 0.45)
        + (out["term_recall"] * 0.25)
        + (out["term_precision"] * 0.10)
        + (length_ratio * 0.20),
        4,
    )
    if classifier is not None and generated:
        ranked = classifier.classify(generated, top_k=60)
        out["classifier_top"] = ranked[0][0] if ranked else None
        out["classifier_target_prob"] = round(
            next((prob for author, prob in ranked if author == persona), 0.0),
            4,
        ) if ranked else 0.0
        out["classifier_target_top1"] = bool(ranked and ranked[0][0] == persona)
    return out


def add_embedding_scores(results: list[dict]) -> None:
    try:
        from scripts.build_persona_embeddings import embed_batch
        import numpy as np
    except Exception as exc:
        print(f"embedding score skipped: {exc}")
        return

    texts = []
    pairs = []
    for idx, row in enumerate(results):
        generated = row.get("generated") or ""
        target = row.get("target", {}).get("content") or ""
        if generated and target:
            pairs.append(idx)
            texts.extend([target, generated])
    if not texts:
        return
    vectors = embed_batch(texts)
    for pair_idx, row_idx in enumerate(pairs):
        a = np.asarray(vectors[pair_idx * 2], dtype="float32")
        b = np.asarray(vectors[pair_idx * 2 + 1], dtype="float32")
        score = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
        results[row_idx].setdefault("scores", {})["embedding_cosine"] = round(score, 4)


async def run_generation(cases: list[dict], args) -> list[dict]:
    if not await llm.available():
        raise SystemExit("LM Studio endpoint not reachable; start it or run --sample-only.")

    classifier = None
    if args.classifier_score:
        try:
            from utils import persona_classifier
            classifier = persona_classifier
            classifier.load()
        except Exception as exc:
            print(f"classifier score skipped: {exc}")
            classifier = None

    results = []
    for idx, case in enumerate(cases, 1):
        persona = case["persona"]
        target = case["target"]["content"]
        recent = _context_rows_for_prompt(case["context"])
        generated = await persona_llm.generate(
            persona,
            case["channel"],
            mode=args.mode,
            context_count=len(recent),
            candidates=args.candidates,
            invoked_by="heldout-eval",
            model_override=args.model or None,
            recent_override=recent,
            exclude_examples=[target],
        )
        generated = generated or ""
        scores = score_pair(target, generated, classifier=classifier, persona=persona)
        row = {
            **case,
            "generated": generated,
            "mode": args.mode,
            "model": args.model or config.LLM_MODEL,
            "scores": scores,
        }
        results.append(row)
        print(
            f"{idx:03d}/{len(cases):03d} {persona} "
            f"shape={scores['shape_score']:.2f} sim={scores['line_similarity']:.2f}: "
            f"{generated[:100]!r}",
            flush=True,
        )
    if args.embed_score:
        add_embedding_scores(results)
    return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_report(path: Path, results: list[dict], args) -> None:
    scored = [row for row in results if row.get("generated")]
    by_author = defaultdict(list)
    for row in scored:
        by_author[row["persona"]].append(row)

    def avg(metric: str, rows: list[dict] = scored) -> float:
        vals = [float(row.get("scores", {}).get(metric, 0.0)) for row in rows]
        return _mean(vals)

    lines = [
        "# Persona Held-Out Reply Eval",
        "",
        f"Cases: {len(results)}",
        f"Generated: {len(scored)}",
        f"Mode: `{args.mode}`",
        f"Model: `{args.model or config.LLM_MODEL}`",
        "",
        "## Overall",
        "",
        f"- Shape score: {avg('shape_score'):.3f}",
        f"- Normalized line similarity: {avg('line_similarity'):.3f}",
        f"- Query-term recall: {avg('term_recall'):.3f}",
        f"- Query-term precision: {avg('term_precision'):.3f}",
        f"- Length ratio: {avg('length_ratio'):.3f}",
    ]
    if any("embedding_cosine" in row.get("scores", {}) for row in scored):
        lines.append(f"- Embedding cosine: {avg('embedding_cosine'):.3f}")
    if any("classifier_target_prob" in row.get("scores", {}) for row in scored):
        lines.append(f"- Classifier target prob: {avg('classifier_target_prob'):.3f}")
        top1 = _mean([
            1.0 if row.get("scores", {}).get("classifier_target_top1") else 0.0
            for row in scored
        ])
        lines.append(f"- Classifier target top-1: {top1:.3f}")

    lines.extend(["", "## By Author", ""])
    for author, rows in sorted(by_author.items(), key=lambda item: (-len(item[1]), item[0])):
        lines.append(
            f"- `{author}`: n={len(rows)}, shape={avg('shape_score', rows):.3f}, "
            f"sim={avg('line_similarity', rows):.3f}, recall={avg('term_recall', rows):.3f}"
        )

    examples = sorted(scored, key=lambda row: row["scores"]["shape_score"])
    lines.extend(["", "## Lowest Shape Scores", ""])
    for row in examples[: min(8, len(examples))]:
        lines.extend([
            f"### {row['persona']} in #{row['channel']} at {row['target']['sent_at']}",
            f"- score: {row['scores']['shape_score']:.3f}",
            f"- real: {row['target']['content']}",
            f"- generated: {row['generated']}",
            "",
        ])

    lines.extend(["", "## Notes", ""])
    lines.append(
        "This is a private diagnostic. The real target line is excluded from "
        "prompt evidence, but the persona's broader archive can still contain "
        "near-duplicate habits or recurring bits. Read scores as comparative "
        "trend signals across versions, not as a perfect human-likeness metric."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authors", default="", help="comma-separated personas; default = eligible regulars")
    parser.add_argument("--channels", default="", help="comma-separated channels; default = config channels")
    parser.add_argument("--since", default="", help="optional lower sent_at bound, e.g. 2025-01-01")
    parser.add_argument("--min-author-messages", type=int, default=500)
    parser.add_argument("--include-bots", action="store_true")
    parser.add_argument("--include-mirror", action="store_true",
                        help="also sample one-speaker mirror logs; off by default for valid chronology")
    parser.add_argument("--context", type=int, default=10)
    parser.add_argument("--min-context", type=int, default=4)
    parser.add_argument("--within-minutes", type=int, default=20)
    parser.add_argument("--per-author", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--scan-limit", type=int, default=0,
                        help="max chronological rows to scan; 0 = all selected rows")
    parser.add_argument("--max-target-chars", type=int, default=260)
    parser.add_argument("--max-context-chars", type=int, default=220)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cases-out", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case-in", type=Path, default=None)
    parser.add_argument("--results-out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample-only", action="store_true",
                        help="write held-out cases but do not call the LLM")
    parser.add_argument("--generate", action="store_true",
                        help="call the persona generator and score outputs")
    parser.add_argument("--mode", default="normal", choices=["normal", "hyper"])
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--model", default="", help="optional LM Studio model override")
    parser.add_argument("--classifier-score", action="store_true")
    parser.add_argument("--embed-score", action="store_true")
    return parser.parse_args()


async def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args()
    if args.case_in:
        cases = read_jsonl(args.case_in)
        print(f"loaded {len(cases)} held-out cases from {args.case_in}")
    else:
        cases = sample_cases(args)
        write_jsonl(args.cases_out, cases)
        print(f"cases -> {args.cases_out}")

    if args.sample_only or not args.generate:
        print("sample-only complete; pass --generate to run the local model eval.")
        return 0

    results = await run_generation(cases, args)
    write_jsonl(args.results_out, results)
    write_report(args.report, results, args)
    print(f"results -> {args.results_out}")
    print(f"report  -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
