"""Preview the LLM persona retrieval step without calling the model.

    python scripts/persona_rag_preview.py <user> "topic or message"
    python scripts/persona_rag_preview.py <user> --channel thickpoo

This is a local interpretability/debug tool. It prints which author keys are
being searched, how many random signature vs. relevant examples were selected,
and a small sample of the real lines that would be fed into the persona prompt.
Nothing is sent to Twitch and LM Studio does not need to be running.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils import chat_archive, persona_llm  # noqa: E402


def _print_lines(title: str, lines: list[str], limit: int) -> None:
    print(f"\n{title} ({min(len(lines), limit)} of {len(lines)}):")
    if not lines or limit <= 0:
        print("  (none)")
        return
    for i, line in enumerate(lines[:limit], 1):
        print(f"  {i:>2}. {line}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("user", help="persona/user to inspect")
    ap.add_argument("query", nargs="*", help="topic/message to retrieve against")
    ap.add_argument("--channel", default="", help="include recent context from this channel")
    ap.add_argument("--context", type=int, default=config.LLM_CONTEXT,
                    help="recent channel messages to use when --channel is set")
    ap.add_argument("--n", type=int, default=config.LLM_EXEMPLARS,
                    help="total exemplar budget")
    ap.add_argument("--show-relevant", type=int, default=20)
    ap.add_argument("--show-signature", type=int, default=8)
    args = ap.parse_args()

    query = " ".join(args.query).strip()
    recent = chat_archive.latest(args.channel, args.context) if args.channel else []
    retrieval_text = persona_llm._retrieval_text(recent, query)
    signature, relevant = persona_llm.select_exemplars(args.user, retrieval_text, n=args.n)

    keys = chat_archive.author_keys(args.user)
    conn = chat_archive.connect()
    counts = []
    for key in keys:
        n = conn.execute("SELECT COUNT(*) FROM messages WHERE author = ?", (key,)).fetchone()[0]
        counts.append((key, n))

    print(f"User: {args.user}")
    print(f"Canonical author: {chat_archive.normalize_author(args.user)}")
    print("Author keys searched: " + ", ".join(f"{key} ({n:,})" for key, n in counts))
    print(f"Exemplar budget: {args.n}")
    print(f"Selected: {len(signature)} random signature + {len(relevant)} relevant")
    if args.channel:
        print(f"Recent context: last {len(recent)} messages from #{chat_archive.normalize_channel(args.channel)}")
    if query:
        print(f"Directed query: {query}")
    if not retrieval_text.strip():
        print("Retrieval query text is empty; relevant examples will be empty.")

    _print_lines("Relevant examples", relevant, args.show_relevant)
    _print_lines("Random signature examples", signature, args.show_signature)


if __name__ == "__main__":
    main()
