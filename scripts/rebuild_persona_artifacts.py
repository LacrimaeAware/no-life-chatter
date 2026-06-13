"""Run the offline persona artifact rebuild pipeline in order.

Default pipeline:

1. Train the authorship classifier.
2. Rebuild lexical voice profiles in the classifier pickle.
3. Rebuild semantic person vectors through the configured local embedding model.
4. Rebuild the per-message semantic index used by burst leaderboards and RAG.
5. Rebuild the v2 `~iq` text-IQ ensemble cache.
6. Smoke-check trait axes/leaderboards.

Use --dry-run to print the commands without running them. Use --skip-embeddings
when LM Studio's embedding endpoint is not running; that also skips the message
index because it uses the same embedding endpoint.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, cmd: list[str], dry_run: bool = False) -> int:
    print(f"\n== {name} ==", flush=True)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return 0
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.monotonic() - started
    print(f"== {name} finished in {elapsed:.1f}s with exit {proc.returncode} ==", flush=True)
    return proc.returncode


def add_if(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned commands without running them")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="keep going after a failed step")

    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--authors", default="",
                        help="comma-separated classifier roster; default = top non-bot authors")
    parser.add_argument("--max-authors", type=int, default=31)
    parser.add_argument("--per-author", type=int, default=4000)
    parser.add_argument("--min-messages", type=int, default=300)
    parser.add_argument("--vocab-size", type=int, default=20000)

    parser.add_argument("--skip-style-profiles", action="store_true")
    parser.add_argument("--style-min-messages", type=int, default=2000)
    parser.add_argument("--style-max-roster", type=int, default=80)
    parser.add_argument("--style-words-top", type=int, default=300)
    parser.add_argument("--style-phrases-top", type=int, default=150)
    parser.add_argument("--style-bg-cap", type=int, default=120000)

    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--semantic-unit", choices=("utterance", "message"),
                        default="utterance",
                        help="unit for semantic embeddings/index; utterance merges same-author bursts")
    parser.add_argument("--embedding-per-author", type=int, default=1000)
    parser.add_argument("--embedding-report", action="store_true")

    parser.add_argument("--skip-message-index", action="store_true")
    parser.add_argument("--message-index-per-author", type=int, default=1500)
    parser.add_argument("--no-message-index-force", action="store_true",
                        help="leave existing per-author message index files in place")

    parser.add_argument("--skip-iq", action="store_true")
    parser.add_argument("--iq-max-utterances", type=int, default=600)
    parser.add_argument("--iq-min-utterances", type=int, default=80)
    parser.add_argument("--iq-author-cap", type=int, default=15000)
    parser.add_argument("--iq-judge", action="store_true",
                        help="also run the optional local-LLM judge layer")

    parser.add_argument("--skip-trait-smoke", action="store_true")
    parser.add_argument("--trait-smoke-n", type=int, default=3)
    args = parser.parse_args()

    py = sys.executable
    steps: list[tuple[str, list[str]]] = []

    if not args.skip_classifier:
        cmd = [py, "scripts/train_classifier.py",
               "--max-authors", str(args.max_authors),
               "--per-author", str(args.per_author),
               "--min-messages", str(args.min_messages),
               "--vocab-size", str(args.vocab_size)]
        if args.authors:
            cmd.extend(["--authors", args.authors])
        steps.append(("classifier", cmd))

    if not args.skip_style_profiles:
        steps.append(("style profiles", [
            py, "scripts/build_style_profiles.py",
            "--min-messages", str(args.style_min_messages),
            "--max-roster", str(args.style_max_roster),
            "--words-top", str(args.style_words_top),
            "--phrases-top", str(args.style_phrases_top),
            "--bg-cap", str(args.style_bg_cap),
        ]))

    if not args.skip_embeddings:
        cmd = [py, "scripts/build_persona_embeddings.py",
               "--per-author", str(args.embedding_per_author),
               "--unit", args.semantic_unit]
        if args.embedding_report:
            cmd.append("--report")
        steps.append(("semantic embeddings", cmd))

    if not args.skip_embeddings and not args.skip_message_index:
        cmd = [py, "scripts/build_message_index.py",
               "--per-author", str(args.message_index_per_author),
               "--unit", args.semantic_unit]
        if not args.no_message_index_force:
            cmd.append("--force")
        steps.append(("semantic message index", cmd))

    if not args.skip_iq:
        cmd = [py, "scripts/build_iq_v2.py",
               "--force",
               "--max-utterances", str(args.iq_max_utterances),
               "--min-utterances", str(args.iq_min_utterances),
               "--author-cap", str(args.iq_author_cap)]
        if args.skip_embeddings:
            cmd.append("--no-embeddings")
        if args.iq_judge:
            cmd.append("--judge")
        steps.append(("text-IQ v2", cmd))

    if not args.skip_trait_smoke:
        steps.append(("trait axis smoke", [
            py, "scripts/trait_axis_smoke.py",
            "--n", str(args.trait_smoke_n),
        ]))

    if not steps:
        print("No steps selected.")
        return 0

    for name, cmd in steps:
        code = run_step(name, cmd, dry_run=args.dry_run)
        if code and not args.continue_on_error:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
