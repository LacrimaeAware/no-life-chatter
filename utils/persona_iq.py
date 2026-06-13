"""Text-IQ v2: peak expressed cognition from chat logs.

This keeps the chat UX as `~iq`, but the internals are deliberately an
ensemble rather than one brittle proxy. The score is roster-relative and built
offline from:

- peak lexical/syntax/reasoning markers over merged utterances
- embedding projections onto cognitive-expression axes
- topic breadth plus niche specificity from utterance embeddings
- optional local-LLM atomic judging of top candidate utterances

The key aggregation rule is "median of the top 10%" so normal chat filler does
not drown out a person's strongest expressed-cognition moments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
import random
import re
import time
import urllib.request
from collections import Counter
from typing import Iterable

import config
from utils import chat_archive, message_quality as mq, persona_classifier as pc

CACHE = os.path.join("data", "unsynced", "iq_scores.pkl")
VERSION = 2

DEFAULT_MIN_UTTERANCES = 80
DEFAULT_MAX_UTTERANCES = 600
DEFAULT_AUTHOR_CAP = 15000
DEFAULT_WORD_FREQ_SAMPLE = 250000

_REASONING_MARKERS = {
    "because", "therefore", "whereas", "although", "though", "unless",
    "despite", "implies", "infer", "assuming", "assume", "hypothesis",
    "hypothetically", "specifically", "technically", "basically",
    "essentially", "counterexample", "base", "rates", "causal", "cause",
    "effect", "incentive", "incentives", "structure", "tradeoff",
    "evidence", "predict", "prediction", "model", "models", "pattern",
    "patterns", "relative", "context", "mechanism", "mechanisms",
}

_CLAUSE_RE = re.compile(
    r"\b(because|although|though|whereas|therefore|however|unless|despite|"
    r"implies|assuming|specifically|technically|hypothetically|relative to|"
    r"in terms of|on the other hand|the reason|it follows|as a result)\b",
    re.I,
)

_QUESTION_RE = re.compile(
    r"\b(why|how|what if|under what|in what sense|compared to|relative to|"
    r"what causes|what explains|how would|why would)\b",
    re.I,
)

# Axis examples are intentionally casual-chat register. They measure expressed
# reasoning moves, not school essay polish.
COG_AXES: dict[str, tuple[list[str], list[str]]] = {
    "abstraction": (
        [
            "lol that guy fell over",
            "this game looks bad",
            "i liked the old thing better",
            "chat is being weird again",
            "that was funny",
        ],
        [
            "the general pattern is incentives changing the behavior",
            "this is really a status game disguised as an argument",
            "the abstract version is scarcity creating fake value",
            "that example is one instance of a broader coordination problem",
            "the same structure shows up in politics and games",
        ],
    ),
    "causal": (
        [
            "it happened because it happened",
            "people are just dumb lol",
            "no reason, thats just how it is",
            "random stuff keeps happening",
            "it is bad because bad",
        ],
        [
            "the likely cause is that the reward system pushes everyone there",
            "if the first assumption is wrong then the conclusion collapses",
            "that outcome follows from the incentives, not from one person",
            "the missing variable is who benefits when the rule changes",
            "this predicts the opposite result once the sample changes",
        ],
    ),
    "nuance": (
        [
            "its obviously always true",
            "only an idiot would disagree",
            "there is literally no other explanation",
            "everyone who says otherwise is lying",
            "case closed, dont think about it",
        ],
        [
            "it depends which part of the claim you mean",
            "that is true in one sense but false under the other assumption",
            "the evidence points that way, but the sample is probably biased",
            "i think the weaker version is right and the strong version is not",
            "there are two different claims getting mixed together here",
        ],
    ),
    "technical": (
        [
            "thing does stuff and it breaks",
            "computer bad",
            "the game is laggy because vibes",
            "numbers went up somehow",
            "idk just click it",
        ],
        [
            "the bottleneck is probably memory bandwidth rather than compute",
            "that is a precision problem, not a recall problem",
            "the parser accepts the token but the downstream schema rejects it",
            "the rate limit changes the equilibrium once requests are batched",
            "the model is overfitting surface markers instead of the target",
        ],
    ),
    "connections": (
        [
            "that is unrelated",
            "new topic anyway",
            "i dont see any pattern",
            "every example is separate",
            "just a random thing",
        ],
        [
            "this is the same failure mode as the last system, just with names changed",
            "that connects to the earlier point about incentives",
            "the analogy is not perfect but the mechanism is similar",
            "both examples are really about information asymmetry",
            "this looks different on the surface but it is the same tradeoff",
        ],
    ),
    "problem_solving": (
        [
            "just give up then",
            "nothing can be done",
            "idk someone fix it",
            "complain until it works",
            "the solution is simply be better",
        ],
        [
            "first isolate whether the failure is input, retrieval, or generation",
            "try the simpler baseline, then add the complicated part back",
            "split it into a detection step and a scoring step",
            "measure the false positives before tuning the threshold",
            "make a held-out set so we can tell if the fix actually generalizes",
        ],
    ),
    "metacognition": (
        [
            "i know im right",
            "my first guess is definitely correct",
            "dont need evidence for this",
            "i never change my mind",
            "thinking about it more is pointless",
        ],
        [
            "my confidence is low because i only saw one example",
            "i might be anchoring on the funniest case instead of the common one",
            "the thing to check is whether my explanation predicts new cases",
            "i am separating what i know from what i am guessing",
            "if the counterexample holds then my model needs to change",
        ],
    ),
}

INTERPRETABLE_FEATURES = [
    "vocab_peak",
    "syntax_peak",
    "lexical_diversity",
    "reasoning_markers",
    "question_quality",
]

EMBED_FEATURES = [
    "abstraction",
    "causal",
    "nuance",
    "technical",
    "connections",
    "problem_solving",
    "metacognition",
    "topic_breadth",
    "niche_depth",
]

GROUPS = {
    "vocab": ("vocab_peak", "lexical_diversity"),
    "syntax": ("syntax_peak", "reasoning_markers"),
    "reasoning": ("causal", "nuance", "connections", "problem_solving", "metacognition", "llm_judge"),
    "abstraction": ("abstraction", "technical"),
    "breadth": ("topic_breadth", "question_quality"),
    "depth": ("niche_depth", "technical", "vocab_peak"),
}

GROUP_WEIGHTS = {
    "reasoning": 0.30,
    "abstraction": 0.20,
    "vocab": 0.15,
    "syntax": 0.15,
    "breadth": 0.10,
    "depth": 0.10,
}


def _read_cache():
    if not os.path.exists(CACHE):
        return None
    with open(CACHE, "rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict) and payload.get("__meta__", {}).get("version") == VERSION:
        return payload
    return None


def cache_info() -> dict:
    payload = _read_cache()
    return payload.get("__meta__", {}) if payload else {}


def _stable_rng(label: str, seed: int = 17) -> random.Random:
    h = hashlib.blake2b(str(label).encode("utf-8"), digest_size=8).hexdigest()
    return random.Random(seed + int(h, 16))


def _sample(items: list, cap: int, label: str) -> list:
    if cap <= 0 or len(items) <= cap:
        return list(items)
    items = list(items)
    _stable_rng(label).shuffle(items)
    return items[:cap]


def _clean(text: str) -> str:
    return mq.clean_text(text)


def _letter_count(text: str) -> int:
    return mq.letter_count(text)


def _symbol_count(text: str) -> int:
    return mq.symbol_count(text)


def _junk_count(text: str) -> int:
    return mq.junk_count(text)


def _is_low_quality_token(tok: str) -> bool:
    return mq.low_quality_token(tok)


def _tokens(text: str) -> list[str]:
    return mq.tokens(text)


def _command_like(text: str) -> bool:
    return mq.command_like(text)


def _spam_like(text: str, toks: list[str]) -> bool:
    return mq.spam_like(text, toks)


def _usable_iq_utterance(raw: str, clean: str, toks: list[str]) -> bool:
    return mq.usable_for_iq(raw, clean, toks)


def _top_tail_median(vals: Iterable[float], pct: float = 0.90) -> float:
    vals = sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not vals:
        return 0.0
    start = max(0, int(len(vals) * pct))
    tail = vals[start:] or vals[-1:]
    mid = len(tail) // 2
    if len(tail) % 2:
        return tail[mid]
    return (tail[mid - 1] + tail[mid]) / 2.0


def _mattr(tokens: list[str], window: int = 50) -> float:
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    vals = []
    for i in range(0, len(tokens) - window + 1, max(1, window // 3)):
        chunk = tokens[i:i + window]
        vals.append(len(set(chunk)) / window)
    return sum(vals) / len(vals)


def _word_freqs(sample: int = DEFAULT_WORD_FREQ_SAMPLE):
    conn = chat_archive.connect()
    counts = Counter()
    for (content,) in conn.execute(
            "SELECT content FROM messages ORDER BY RANDOM() LIMIT ?", (sample,)):
        counts.update(_tokens(_clean(content or "")))
    total = sum(counts.values()) or 1
    return counts, total


def _rarity_fn(freqs: Counter, total: int):
    def rarity(word: str) -> float | None:
        if freqs.get(word, 0) < 3:
            return None
        return -math.log((freqs.get(word, 0) + 1) / total)
    return rarity


def _mean_rarity(toks: list[str], rarity) -> float | None:
    vals = [rarity(t) for t in toks]
    vals = [v for v in vals if v is not None]
    if len(vals) < 3:
        return None
    return sum(vals) / len(vals)


def _has_reasoning_shape(text: str, toks: list[str]) -> bool:
    if _QUESTION_RE.search(text or ""):
        return True
    if _CLAUSE_RE.search(text or ""):
        return True
    marker_count = sum(1 for t in toks if t in _REASONING_MARKERS)
    return marker_count >= 2


def _row_has_corpus_signal(row: dict, rarity) -> bool:
    if _mean_rarity(row["tokens"], rarity) is not None:
        return True
    return _has_reasoning_shape(row["clean"], row["tokens"])


def _utterance_rows(author: str, author_cap: int) -> list[dict]:
    rows = []
    for raw in chat_archive.utterances_for(author):
        clean = _clean(raw)
        toks = _tokens(clean)
        if not _usable_iq_utterance(raw, clean, toks):
            continue
        rows.append({"raw": raw, "clean": clean, "tokens": toks})
    if len(rows) > author_cap:
        rows = _sample(rows, author_cap, f"{author}:utterance-cap")
    return rows


def _interpretable_features(rows: list[dict], rarity) -> dict[str, float]:
    if not rows:
        return {name: 0.0 for name in INTERPRETABLE_FEATURES}
    rarity_scores = []
    syntax_scores = []
    marker_scores = []
    question_scores = []
    all_tokens = []
    for row in rows:
        toks = row["tokens"]
        text = row["clean"]
        all_tokens.extend(toks)
        r = _mean_rarity(toks, rarity)
        if r is not None:
            rarity_scores.append(r)
        n = len(toks)
        clause = len(_CLAUSE_RE.findall(text))
        marker_count = sum(1 for t in toks if t in _REASONING_MARKERS)
        marker_density = marker_count / math.sqrt(max(1, n))
        syntax_scores.append(math.log1p(n) * (1.0 + clause + marker_density))
        marker_scores.append(marker_density + clause)
        q = 0.0
        if "?" in text:
            q += 0.35
        if _QUESTION_RE.search(text):
            q += 1.0
        if q and n >= 7:
            q += min(0.6, math.log1p(n) / 6)
        question_scores.append(q)
    return {
        "vocab_peak": _top_tail_median(rarity_scores),
        "syntax_peak": _top_tail_median(syntax_scores),
        "lexical_diversity": _mattr(all_tokens),
        "reasoning_markers": _top_tail_median(marker_scores),
        "question_quality": _top_tail_median(question_scores),
    }


def _embed_batch(texts: list[str]):
    if not texts:
        return []
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    body = json.dumps({"model": config.LLM_EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(
        base + "/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return [d["embedding"] for d in data["data"]]


def _normalize_matrix(vecs):
    import numpy as np

    mat = np.asarray(vecs, dtype="float32")
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat


def _axis_vectors():
    import numpy as np

    labels = []
    spans = {}
    for axis, (neg, pos) in COG_AXES.items():
        start = len(labels)
        labels.extend(neg + pos)
        spans[axis] = (start, start + len(neg), start + len(neg) + len(pos))
    embs = _normalize_matrix(_embed_batch(labels))
    out = {}
    for axis, (start, split, end) in spans.items():
        neg = embs[start:split].mean(axis=0)
        pos = embs[split:end].mean(axis=0)
        vec = pos - neg
        out[axis] = vec / (np.linalg.norm(vec) + 1e-9)
    return out


def _embedding_features(author_rows: dict[str, list[dict]], max_utterances: int):
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    axis = _axis_vectors()
    matrices = {}
    aligned_rows = {}
    all_vecs = []
    owners = []
    for author, rows in author_rows.items():
        eligible = [
            r for r in rows
            if 4 <= len(r["tokens"]) <= 80 and len(r["clean"]) <= 700
        ]
        eligible = _sample(eligible, max_utterances, f"{author}:iq-embed")
        if len(eligible) < 20:
            continue
        vecs = []
        for i in range(0, len(eligible), 64):
            vecs.extend(_embed_batch([r["clean"] for r in eligible[i:i + 64]]))
        mat = _normalize_matrix(vecs)
        matrices[author] = mat
        aligned_rows[author] = eligible
        all_vecs.append(mat)
        owners.extend([author] * len(mat))

    if not all_vecs:
        return {}, "no utterances embedded"

    all_mat = np.vstack(all_vecs)
    global_mean = all_mat.mean(axis=0)
    centered = all_mat - global_mean
    specificity = np.linalg.norm(centered, axis=1)
    k = min(24, max(6, int(math.sqrt(len(all_mat) / 80)) + 6))
    km = MiniBatchKMeans(n_clusters=k, random_state=23, batch_size=2048, n_init=5)
    labels = km.fit_predict(all_mat)

    out = {}
    offset = 0
    for author, mat in matrices.items():
        n = len(mat)
        lab = labels[offset:offset + n]
        spec = specificity[offset:offset + n]
        offset += n
        counts = np.bincount(lab, minlength=k).astype("float64")
        share = counts / max(1.0, counts.sum())
        nz = share[share > 0]
        feats = {
            "topic_breadth": float(-(nz * np.log(nz)).sum() / math.log(k)),
            "niche_depth": float(_top_tail_median(spec)),
        }
        for name, vec in axis.items():
            feats[name] = float(_top_tail_median(mat @ vec))
        out[author] = feats
    return out, f"embedded {len(all_mat)} utterances"


def _chat_sync(prompt: str, max_tokens: int = 500, model: str | None = None) -> str:
    body = json.dumps({
        "model": model or getattr(config, "LLM_MODEL", ""),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": False,
    }).encode()
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def _judge_author(author: str, rows: list[dict], rarity, items: int = 8) -> float | None:
    if not getattr(config, "LLM_MODEL", ""):
        return None
    scored = []
    for row in rows:
        toks = row["tokens"]
        text = row["clean"]
        if len(toks) < 6 or len(text) > 500:
            continue
        r = _mean_rarity(toks, rarity)
        if r is None:
            continue
        markers = sum(1 for t in toks if t in _REASONING_MARKERS)
        score = r + 0.8 * markers + 0.4 * len(_CLAUSE_RE.findall(text)) + 0.12 * math.log1p(len(toks))
        scored.append((score, text))
    scored.sort(key=lambda kv: -kv[0])
    examples = [text for _score, text in scored[:items]]
    if not examples:
        return None
    prompt = (
        "Rate these Twitch-chat utterances for expressed cognitive density. "
        "Ignore whether you agree, ignore politeness, and do not reward mere verbosity. "
        "For each item, score 0-4 on: reasoning, abstraction, precision, nuance, novelty. "
        "Reply ONLY JSON: {\"items\":[{\"reasoning\":0,\"abstraction\":0,"
        "\"precision\":0,\"nuance\":0,\"novelty\":0}, ...]}.\n\n"
        + "\n".join(f"{i + 1}. {text}" for i, text in enumerate(examples))
    )
    raw = _chat_sync(prompt)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    data = json.loads(re.sub(r",\s*([\]}])", r"\1", match.group(0)))
    vals = []
    for item in data.get("items", []):
        if not isinstance(item, dict):
            continue
        nums = [float(item.get(k, 0)) for k in ("reasoning", "abstraction", "precision", "nuance", "novelty")]
        vals.append(sum(nums) / (4.0 * len(nums)))
    return sum(vals) / len(vals) if vals else None


def _canonical_roster(names: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for name in names:
        canon = chat_archive.normalize_author(name)
        if not canon or canon in seen or chat_archive._is_noise_author(canon):
            continue
        seen.add(canon)
        out.append(canon)
    return out


def _roster(max_authors: int | None = None) -> list[str]:
    try:
        profiles = (pc.load().get("profiles") or {})
        names = _canonical_roster(sorted(profiles))
    except Exception:
        names = []
    if not names:
        msg_dir = os.path.join("data", "unsynced", "msg_index")
        if os.path.isdir(msg_dir):
            names = _canonical_roster(sorted(f[:-4] for f in os.listdir(msg_dir) if f.endswith(".npz")))
    if max_authors:
        names = names[:max_authors]
    return names


def _z_table(raw: dict[str, dict[str, float]], features: list[str]):
    import numpy as np

    names = list(raw)
    z = {a: {} for a in names}
    stats = {}
    for feat in features:
        vals = np.array([raw[a].get(feat, 0.0) for a in names], dtype="float64")
        mu = float(vals.mean())
        sd = float(vals.std()) or 1.0
        stats[feat] = (mu, sd)
        vals = np.clip((vals - mu) / sd, -3.5, 3.5)
        for i, a in enumerate(names):
            z[a][feat] = float(vals[i])
    return z, stats


def _group_scores(z_feats: dict[str, float]) -> dict[str, float]:
    groups = {}
    for group, feats in GROUPS.items():
        vals = [z_feats[f] for f in feats if f in z_feats]
        if vals:
            groups[group] = sum(vals) / len(vals)
    return groups


def _weighted(groups: dict[str, float]) -> float:
    parts = [(GROUP_WEIGHTS[g], v) for g, v in groups.items() if g in GROUP_WEIGHTS]
    total = sum(w for w, _v in parts) or 1.0
    return sum(w * v for w, v in parts) / total


def _percentiles(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    n = len(ordered)
    if n <= 1:
        return {a: 50 for a, _ in ordered}
    return {a: int(round(100 * i / (n - 1))) for i, (a, _v) in enumerate(ordered)}


def _half_delta(
    split_raw: tuple[dict[str, float], dict[str, float]],
    stats: dict[str, tuple[float, float]],
) -> float | None:
    left, right = split_raw
    vals = []
    for feat in INTERPRETABLE_FEATURES:
        if feat not in left or feat not in right or feat not in stats:
            continue
        mu, sd = stats[feat]
        vals.append(abs(((left[feat] - mu) / sd) - ((right[feat] - mu) / sd)))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _confidence(n_utterances: int, split_delta: float | None, used_embeddings: bool) -> str:
    if split_delta is None:
        return "low"
    if n_utterances >= 700 and split_delta <= 0.45 and used_embeddings:
        return "high"
    if n_utterances >= 200 and split_delta <= 0.90:
        return "medium"
    return "low"


def compute_all(
    force: bool = False,
    *,
    use_embeddings: bool = True,
    use_llm: bool = False,
    max_utterances: int = DEFAULT_MAX_UTTERANCES,
    min_utterances: int = DEFAULT_MIN_UTTERANCES,
    author_cap: int = DEFAULT_AUTHOR_CAP,
    sample_word_freq: int = DEFAULT_WORD_FREQ_SAMPLE,
    max_authors: int | None = None,
    judge_items: int = 8,
) -> dict[str, dict]:
    """Build or load the roster-relative text-IQ scores."""
    if not force:
        payload = _read_cache()
        if payload:
            return payload["scores"]

    started = time.time()
    authors = _roster(max_authors=max_authors)
    freqs, total = _word_freqs(sample_word_freq)
    rarity = _rarity_fn(freqs, total)

    author_rows: dict[str, list[dict]] = {}
    raw: dict[str, dict[str, float]] = {}
    split_raw: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for author in authors:
        rows = _utterance_rows(author, author_cap=author_cap)
        rows = [row for row in rows if _row_has_corpus_signal(row, rarity)]
        if len(rows) < min_utterances:
            continue
        author_rows[author] = rows
        raw[author] = _interpretable_features(rows, rarity)
        mid = max(1, len(rows) // 2)
        split_raw[author] = (
            _interpretable_features(rows[:mid], rarity),
            _interpretable_features(rows[mid:], rarity),
        )

    embed_note = "disabled"
    used_embeddings = False
    if use_embeddings and raw and getattr(config, "LLM_EMBED_MODEL", ""):
        try:
            embed_raw, embed_note = _embedding_features(author_rows, max_utterances)
            for author, feats in embed_raw.items():
                if author in raw:
                    raw[author].update(feats)
            used_embeddings = bool(embed_raw)
        except Exception as exc:
            embed_note = f"failed: {exc}"
            logging.warning("iq v2 embedding features failed: %s", exc)

    llm_note = "disabled"
    if use_llm and raw:
        judged = 0
        for author, rows in author_rows.items():
            try:
                val = _judge_author(author, rows, rarity, items=judge_items)
            except Exception as exc:
                logging.warning("iq v2 judge failed for %s: %s", author, exc)
                val = None
            if val is not None:
                raw[author]["llm_judge"] = val
                judged += 1
        llm_note = f"judged {judged} authors"

    if not raw:
        return {}

    features = sorted({feat for feats in raw.values() for feat in feats})
    z_feats, stats = _z_table(raw, features)
    group_raw = {author: _group_scores(z_feats[author]) for author in raw}
    composite_raw = {author: _weighted(groups) for author, groups in group_raw.items()}

    import numpy as np

    vals = np.array(list(composite_raw.values()), dtype="float64")
    mu = float(vals.mean())
    sd = float(vals.std()) or 1.0
    percentiles = _percentiles(composite_raw)

    out = {}
    for author, groups in group_raw.items():
        final_z = (composite_raw[author] - mu) / sd
        delta = _half_delta(split_raw[author], stats)
        n = len(author_rows[author])
        row = {
            "iq": int(max(62, min(158, round(100 + 15 * final_z)))),
            "z": round(float(final_z), 2),
            "percentile": percentiles[author],
            "confidence": _confidence(n, delta, used_embeddings),
            "n_utterances": n,
            "split_delta": None if delta is None else round(float(delta), 2),
            "components": {k: round(float(v), 2) for k, v in sorted(groups.items())},
        }
        # Back-compat and compact command output.
        for key in ("vocab", "syntax", "reasoning", "abstraction", "breadth", "depth"):
            if key in groups:
                row[key] = round(float(groups[key]), 2)
        out[author] = row

    payload = {
        "__meta__": {
            "version": VERSION,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "authors": len(out),
            "features": features,
            "embed_model": getattr(config, "LLM_EMBED_MODEL", ""),
            "embedding_features": embed_note,
            "llm_judge": llm_note,
            "max_utterances": max_utterances,
            "min_utterances": min_utterances,
            "author_cap": author_cap,
            "elapsed_sec": round(time.time() - started, 1),
        },
        "scores": out,
    }
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(payload, fh)
    return out


def score(author: str):
    data = _canonicalized_scores(compute_all())
    return data.get(chat_archive.normalize_author(author))


def leaderboard(n: int = 5, reverse: bool = False):
    data = _canonicalized_scores(compute_all())
    ranked = sorted(data.items(), key=lambda kv: kv[1]["iq"], reverse=not reverse)
    return ranked[:n]


def _canonicalized_scores(data: dict) -> dict:
    """Collapse stale cached score labels through the current alias map.

    IQ scores are cached on disk, so after adding an alias the cache can still
    contain both `oldname` and `currentname`. Until the next rebuild, display the
    current canonical name once. If the current name already has a row, prefer
    it; otherwise keep the row with the most utterances.
    """
    grouped: dict[str, tuple[str, dict]] = {}
    for author, row in data.items():
        canon = chat_archive.normalize_author(author)
        if chat_archive._is_noise_author(canon):
            continue
        prev = grouped.get(canon)
        if prev is None:
            grouped[canon] = (author, row)
            continue
        prev_author, prev_row = prev
        prefer_new = (
            (author == canon and prev_author != canon)
            or (
                author != canon
                and prev_author != canon
                and row.get("n_utterances", 0) > prev_row.get("n_utterances", 0)
            )
        )
        if prefer_new:
            grouped[canon] = (author, row)
    return {canon: row for canon, (_source, row) in grouped.items()}
