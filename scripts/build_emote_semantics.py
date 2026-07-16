"""Usage-context emote semantics: what an emote MEANS, learned from our logs.

Emote names lie (deliberately irrelevant names), images aren't always
fetchable (dead emotes in old logs), and fake personal emotes exist. But
meaning-from-usage covers all of it: an emote's vector = mean embedding of
the messages it appears in, with the emote token itself removed. DansGame's
contexts are disgust, so its vector points at disgust — the stance-OPERATOR
meaning the name embedding can never carry.

Per emote: sample up to --contexts deduped messages containing it (FTS), strip
the emote, and mean-pool the remaining words. Bare-emote reactions use the
preceding human line as their context. Evidence is diversified across authors;
emotes with too few usable contexts keep their prior vector when available.

    python scripts/build_emote_semantics.py [--top 2000] [--contexts 160]
    python scripts/build_emote_semantics.py --emotes BatChest,KEKW --contexts 200 --refresh
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import config  # noqa: E402
from utils import atomic_file, chat_archive, message_quality, persona_classifier as pc  # noqa: E402
from scripts.build_persona_embeddings import embed_batch  # noqa: E402

OUT = os.path.join("data", "unsynced", "emote_semantics.pkl")
VERSION = 2


def _partial_path(path: str = OUT) -> str:
    stem, suffix = os.path.splitext(path)
    return stem + ".partial" + suffix


def _load_existing(path: str = OUT):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("emotes"), dict):
        return dict(payload["emotes"])
    return dict(payload) if isinstance(payload, dict) else {}


def _load_metadata(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return dict(payload.get("__meta__") or {}) if isinstance(payload, dict) else {}


def _request_signature(*, top: int, contexts: int, emotes: str, refresh: bool) -> str:
    requested = sorted(
        token.strip().casefold() for token in (emotes or "").split(",") if token.strip()
    )
    raw = json.dumps({
        "top": int(top),
        "contexts": int(contexts),
        "emotes": requested,
        "refresh": bool(refresh),
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _write(
    out,
    *,
    target_contexts: int,
    complete: bool,
    path: str = OUT,
    request_signature: str = "",
) -> None:
    payload = {
        "__meta__": {
            "version": VERSION,
            "model": config.LLM_EMBED_MODEL,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_contexts": int(target_contexts),
            "emotes": len(out),
            "complete": bool(complete),
            "request_signature": request_signature,
        },
        "emotes": out,
    }
    with atomic_file.open_atomic(path, "wb") as handle:
        pickle.dump(payload, handle)


def _strip_emote(text: str, emote: str) -> str:
    return re.sub(
        rf"(?<!\w){re.escape(emote)}(?!\w)",
        " ",
        text or "",
        flags=re.IGNORECASE,
    )


def _clean_context(message: str, previous: str, emote: str) -> tuple[str | None, bool]:
    own = message_quality.clean_text(
        _strip_emote(message, emote),
        strip_emotes=True,
        strip_urls=True,
    )
    own_words = own.split()
    if len(own_words) >= 3:
        return own[:300], False
    prior = message_quality.semantic_text(
        _strip_emote(previous or "", emote), min_words=3, max_words=70
    )
    if not prior:
        return None, False
    combined = f"{prior} {own}".strip() if own else prior
    return combined[:300], True


def _sample_rows(conn, emote: str, limit: int):
    salt = int(hashlib.sha1(emote.casefold().encode("utf-8")).hexdigest()[:8], 16)
    try:
        query = chat_archive._fts_phrase(emote)
        max_id = int(conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM messages"
        ).fetchone()[0])
        if not max_id:
            return []
        per_start = max(20, (limit + 1) // 2)
        first = salt % max_id
        starts = [first, (first + max_id // 2) % max_id]
        rows = []
        for start in starts:
            chunk = conn.execute(
                "SELECT m.id, m.sent_at, m.channel, m.author, m.content "
                "FROM messages_fts f CROSS JOIN messages m ON m.id=f.rowid "
                "WHERE f.messages_fts MATCH ? AND f.rowid>=? LIMIT ?",
                (query, start, per_start),
            ).fetchall()
            if len(chunk) < per_start and start:
                chunk.extend(conn.execute(
                    "SELECT m.id, m.sent_at, m.channel, m.author, m.content "
                    "FROM messages_fts f CROSS JOIN messages m ON m.id=f.rowid "
                    "WHERE f.messages_fts MATCH ? AND f.rowid<? LIMIT ?",
                    (query, start, per_start - len(chunk)),
                ).fetchall())
            rows.extend(chunk)
        return rows[:limit]
    except Exception:
        like = "%" + emote.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_") + "%"
        return conn.execute(
            "SELECT m.id, m.sent_at, m.channel, m.author, m.content "
            "FROM messages m WHERE m.content LIKE ? ESCAPE '\\' "
            "ORDER BY m.id LIMIT ?",
            (like, limit),
        ).fetchall()


def _previous_human_lines(conn, rows) -> dict[int, str]:
    """Fetch one preceding human line per target in one indexed SQL pass."""
    if not rows:
        return {}
    conn.execute(
        "CREATE TEMP TABLE IF NOT EXISTS emote_context_targets ("
        "id INTEGER PRIMARY KEY, sent_at TEXT, channel TEXT, author TEXT)"
    )
    conn.execute("DELETE FROM emote_context_targets")
    conn.executemany(
        "INSERT INTO emote_context_targets VALUES (?, ?, ?, ?)",
        [
            (int(row_id), sent_at, chat_archive.normalize_channel(channel), author)
            for row_id, sent_at, channel, author, _content in rows
        ],
    )
    found = {}
    query = (
        "SELECT t.id, (SELECT json_object('author', p.author, 'content', p.content) "
        "FROM messages p WHERE p.channel=t.channel AND p.author<>t.author "
        "AND p.sent_at >= datetime(t.sent_at, '-5 minutes') "
        "AND (p.sent_at<t.sent_at OR (p.sent_at=t.sent_at AND p.id<t.id)) "
        "AND ltrim(p.content) NOT GLOB '~*' AND ltrim(p.content) NOT GLOB '!*' "
        "AND ltrim(p.content) NOT GLOB '$*' AND ltrim(p.content) NOT GLOB '<*' "
        "ORDER BY p.sent_at DESC, p.id DESC LIMIT 1) "
        "FROM emote_context_targets t"
    )
    targets = {int(row[0]): chat_archive.normalize_author(row[3]) for row in rows}
    for row_id, blob in conn.execute(query):
        if not blob:
            continue
        try:
            item = json.loads(blob)
        except Exception:
            continue
        author = chat_archive.normalize_author(item.get("author") or "")
        content = item.get("content") or ""
        if author == targets.get(int(row_id)) or chat_archive._is_noise_author(author):
            continue
        if message_quality.semantic_text(content, min_words=3, max_words=70):
            found[int(row_id)] = content
    return found


def _contexts_for(conn, emote: str, target: int) -> tuple[list[str], int]:
    token_re = re.compile(rf"(?<!\w){re.escape(emote)}(?!\w)", re.IGNORECASE)
    slots = []
    pending = []
    seen_ids = set()
    candidate_goal = max(target + 40, target * 3)
    for row_id, sent_at, channel, author, content in _sample_rows(
        conn, emote, max(target * 12, 600)
    ):
        if int(row_id) in seen_ids:
            continue
        seen_ids.add(int(row_id))
        canon = chat_archive.normalize_author(author)
        if chat_archive._is_noise_author(canon) or message_quality.command_like(content or ""):
            continue
        if not token_re.search(content or ""):
            continue
        cleaned, used_previous = _clean_context(content or "", "", emote)
        if not cleaned:
            pending.append((row_id, sent_at, channel, author, content))
        slots.append({
            "id": int(row_id),
            "sent_at": sent_at,
            "author": canon,
            "content": content or "",
            "text": cleaned,
            "used_previous": used_previous,
        })

    own_keys = {
        chat_archive.line_match_key(slot["text"] or "")
        for slot in slots if slot["text"]
    }
    missing = max(0, target - len(own_keys))
    previous_budget = min(
        len(pending),
        max(target // 4, missing * 2),
    )
    if previous_budget and len(pending) > previous_budget:
        pending = [
            pending[(index * len(pending)) // previous_budget]
            for index in range(previous_budget)
        ]
    previous_by_id = _previous_human_lines(conn, pending)
    candidates = []
    seen = set()
    for slot in slots:
        cleaned = slot["text"]
        used_previous = slot["used_previous"]
        if not cleaned:
            cleaned, used_previous = _clean_context(
                slot["content"], previous_by_id.get(slot["id"], ""), emote
            )
        key = chat_archive.line_match_key(cleaned or "")
        if not cleaned or not key or key in seen:
            continue
        seen.add(key)
        candidates.append({
            "text": cleaned,
            "author": slot["author"],
            "day": (slot["sent_at"] or "")[:10],
            "used_previous": used_previous,
        })
        if len(candidates) >= candidate_goal:
            break

    # First pass limits one prolific chatter/day from defining community-wide
    # meaning; the second fills capacity for genuinely niche/personal emotes.
    selected = []
    selected_ids = set()
    author_counts = Counter()
    author_days = set()
    for index, row in enumerate(candidates):
        author_day = (row["author"], row["day"])
        if author_day in author_days or author_counts[row["author"]] >= 4:
            continue
        selected.append(row)
        selected_ids.add(index)
        author_counts[row["author"]] += 1
        author_days.add(author_day)
        if len(selected) >= target:
            break
    if len(selected) < target:
        for index, row in enumerate(candidates):
            if index in selected_ids:
                continue
            selected.append(row)
            if len(selected) >= target:
                break
    return [row["text"] for row in selected], sum(
        bool(row["used_previous"]) for row in selected
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=2000)
    ap.add_argument("--contexts", type=int, default=160)
    ap.add_argument("--emotes", default="",
                    help="comma-separated emotes to build instead of the frequency top-N")
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild existing vectors; with --emotes only those keys are replaced")
    args = ap.parse_args()

    request_signature = _request_signature(
        top=args.top,
        contexts=args.contexts,
        emotes=args.emotes,
        refresh=args.refresh,
    )

    dependency_probe = embed_batch(["emote semantic dependency check"])
    if len(dependency_probe) != 1 or not dependency_probe[0]:
        raise RuntimeError("embedding backend returned an empty dependency probe")

    # Source: the ground-truth registry UNION the most frequent emote-shaped
    # tokens in the archive. Sourcing from per-person DISTINCTIVE profiles
    # (the old way) filtered out the universal meaning-bearers (DansGame,
    # Sadge, KEKW) precisely because everyone uses them.
    reg = {}
    reg_path = os.path.join("data", "unsynced", "emote_registry.json")
    if os.path.exists(reg_path):
        with open(reg_path, encoding="utf-8") as handle:
            reg = json.load(handle)
    conn = chat_archive.connect()
    if args.emotes.strip():
        registry_case = {name.casefold(): name for name in reg}
        emotes = [
            registry_case.get(e.strip().casefold(), e.strip())
            for e in args.emotes.split(",") if e.strip()
        ]
    else:
        freq = Counter()
        total_rows = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        stride = max(1, total_rows // 400000)
        offset = 17 % stride
        registry_case = {name.casefold(): name for name in reg}
        for (m,) in conn.execute(
            "SELECT content FROM messages WHERE id % ? = ? ORDER BY id LIMIT 400000",
            (stride, offset),
        ):
            for tok in (m or "").split():
                raw = tok.strip(".,!?;:\"'()[]{}<>")
                canonical = registry_case.get(raw.casefold(), raw)
                if canonical in reg or pc._is_emote_token(canonical):
                    freq[canonical] += 1
        emotes = [e for e, n in freq.most_common(args.top) if n >= 5]
    print(f"building context vectors for {len(emotes)} emotes "
          f"(registry {len(reg)} + archive-frequent)...")

    conn = chat_archive.connect()
    partial = _partial_path()
    resuming = os.path.exists(partial)
    if resuming and _load_metadata(partial).get("request_signature") != request_signature:
        raise RuntimeError(
            "emote checkpoint belongs to different build arguments; "
            "resume that build before starting another"
        )
    out = _load_existing(partial) if resuming else _load_existing()
    done = 0
    if args.refresh and not resuming:
        if not args.emotes.strip():
            out = {}
    for i, emote in enumerate(emotes, 1):
        existing_key = next(
            (key for key in out if key.casefold() == emote.casefold()), emote
        )
        existing = out.get(existing_key) or {}
        if not args.refresh and int(existing.get("n") or 0) >= args.contexts:
            continue
        ctxs, previous_contexts = _contexts_for(conn, emote, args.contexts)
        if len(ctxs) < 8:
            continue
        embs = embed_batch(ctxs)
        v = np.asarray(embs, dtype="float32").mean(axis=0)
        if existing_key != emote:
            out.pop(existing_key, None)
        out[emote] = {
            "vector": (v / (np.linalg.norm(v) + 1e-9)).astype("float16"),
            "n": len(ctxs),
            "previous_contexts": previous_contexts,
        }
        done += 1
        if done % 25 == 0:
            _write(
                out,
                target_contexts=args.contexts,
                complete=False,
                path=partial,
                request_signature=request_signature,
            )
            print(f"  ({i}/{len(emotes)}) {len(out)} vectors saved...", flush=True)
    _write(
        out,
        target_contexts=args.contexts,
        complete=True,
        request_signature=request_signature,
    )
    try:
        os.remove(partial)
    except FileNotFoundError:
        pass
    print(f"done: {len(out)} emote context-vectors -> {OUT}")

    # sanity: do operator emotes point where they should?
    probes = {"disgust ew gross nasty": None, "sad crying unhappy": None,
              "happy nice wholesome": None}
    P = {k: np.asarray(v, dtype="float32") for k, v in
         zip(probes, embed_batch(list(probes)))}
    for k in P:
        P[k] /= np.linalg.norm(P[k])
    for emote in ["DansGame", "Sadge", "FeelsBadMan", "ApuDoomer", "FeelsOkayMan"]:
        d = out.get(emote)
        if not d:
            print(f"  {emote}: (no vector)")
            continue
        v = np.asarray(d["vector"], dtype="float32")
        best = max(P, key=lambda k: float(v @ P[k]))
        print(f"  {emote}: closest probe = '{best}' "
              + " ".join(f"{k.split()[0]}:{float(v @ P[k]):+.2f}" for k in P))


if __name__ == "__main__":
    main()
