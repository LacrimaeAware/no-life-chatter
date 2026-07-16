"""Build LLM-verified user profiles v5 - see utils/user_profiles.py.

Needs LM Studio up (uses the local chat model as the sincerity/extraction
judge). Incremental: judged messages are cached, so re-running only pays for
new evidence — safe to run repeatedly, e.g. as a dead-hours batch.

    python scripts/build_user_profiles.py name1 name2       # just these
    python scripts/build_user_profiles.py --roster 20       # top-N archive authors
    python scripts/build_user_profiles.py name --dry-run    # retrieval only, no LLM
    python scripts/build_user_profiles.py name --show       # print stored profile

Output: data/unsynced/user_profiles.json (gitignored — real people, owner's
eyes only; never paste into tracked docs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# chat lines contain emoji; never let a cp1252 console kill a paid-for run
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import chat_archive, user_profiles  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("authors", nargs="*", help="chatter names (any alias)")
    ap.add_argument("--roster", type=int, default=0,
                    help="add the top-N most-active archive authors")
    ap.add_argument("--cap", type=int, default=30,
                    help="max judged candidates per (author, slot)")
    ap.add_argument("--batch-size", type=int, default=6,
                    help="candidate contexts judged per local-model call")
    ap.add_argument("--dry-run", action="store_true",
                    help="show retrieval candidates; no LLM calls, no writes")
    ap.add_argument("--show", action="store_true",
                    help="print the stored profile for the given authors")
    args = ap.parse_args()

    authors = [chat_archive.normalize_author(a) for a in args.authors]
    if args.roster:
        from utils.fact_bank import _author_roster
        for a in _author_roster(args.roster, min_messages=2000):
            if a not in authors:
                authors.append(a)
    if not authors:
        ap.error("give author names or --roster N")

    if args.show:
        for author in authors:
            prof = user_profiles.profile_for(author)
            print(f"=== {author} ===")
            print(json.dumps(prof, indent=1, ensure_ascii=False) or "{}")
            line = user_profiles.profile_line(author)
            print(f"prompt line: {line!r}\n")
        return 0

    if args.dry_run:
        for author in authors:
            print(f"=== {author} ===")
            for slot in user_profiles.SLOTS:
                rows = user_profiles.candidate_rows(author, slot, cap=args.cap)
                print(f"  {slot}: {len(rows)} candidates")
                for r in rows[:3]:
                    print(f"    {r['sent_at'][:10]} #{r['channel']}: {r['content'][:90]}")
        return 0

    verdicts = {"n": 0}

    user_profiles.require_model_dependency()

    def progress(author, slot, verdict):
        verdicts["n"] += 1
        tag = "+" if verdict.get("asserts") and verdict.get("sincerity") == "sincere" else "-"
        print(f"[{verdicts['n']:>4}] {tag} {author}/{slot}: "
              f"{verdict.get('value')!r} ({verdict.get('sincerity')}) "
              f"<- {verdict['content'][:70]}")

    store = user_profiles.build_profiles(
        authors,
        per_slot_cap=args.cap,
        batch_size=args.batch_size,
        progress=progress,
    )
    candidate_rows = int((store.get("_meta") or {}).get("candidate_rows", 0))
    if not candidate_rows:
        print(
            "ERROR: retrieval returned zero profile candidates for the requested roster.",
            file=sys.stderr,
        )
        return 2
    print(f"\nCandidate verdicts cached this run: {verdicts['n']}")
    print(f"Retrieval candidates considered: {candidate_rows}")
    for author in authors:
        prof = store["profiles"].get(author, {})
        confirmed = sum(1 for s in prof.values()
                        if s.get("status") == "confirmed"
                        or any(v["status"] == "confirmed" for v in s.get("values", [])))
        print(f"{author}: {len(prof)} slots resolved, {confirmed} with confirmed facts")
        line = user_profiles.profile_line(author)
        if line:
            print(f"  prompt line: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
