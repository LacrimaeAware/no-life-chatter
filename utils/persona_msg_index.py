"""Consumers of the per-message embedding index (scripts/build_message_index.py).

Two capabilities person-level mean vectors can't provide:

- **burst traits**: a chatter who is extremely doomer in 10% of messages and
  neutral otherwise averages to mild. The 90th-percentile of per-message axis
  projections catches the tail — "how doomer are their doomer moments".
- **semantic retrieval**: the author's archived messages nearest in MEANING
  to the live conversation — evidence for the persona prompt that FTS keyword
  matching can't find (paraphrases, same topic in different words).

Per-author .npz files load lazily with a small LRU; nothing here runs unless
the index exists, so the bot is unaffected until a build completes.
"""

import json
import os
import urllib.request
from collections import OrderedDict

import config
from utils import chat_archive

DIR = os.path.join("data", "unsynced", "msg_index")
_CACHE = OrderedDict()  # author -> (vectors fp32, texts)
_CACHE_MAX = 8
_burst_cache = {}  # axis_name -> {author: percentile_score}
_contra_cache = {}  # axis_name -> {author: contradiction_z}


def _path(author):
    return os.path.join(DIR, f"{chat_archive.normalize_author(author)}.npz")


def available(author=None) -> bool:
    if author is None:
        return os.path.isdir(DIR) and any(f.endswith(".npz") for f in os.listdir(DIR))
    return os.path.exists(_path(author))


def _load(author):
    import numpy as np
    canon = chat_archive.normalize_author(author)
    if canon in _CACHE:
        _CACHE.move_to_end(canon)
        return _CACHE[canon]
    d = np.load(_path(canon), allow_pickle=True)
    pair = (d["vectors"].astype("float32"), list(d["texts"]))
    _CACHE[canon] = pair
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return pair


def _embed_one(text):
    import numpy as np
    base = config.LLM_ENDPOINT.split("/v1/")[0]
    body = json.dumps({"model": config.LLM_EMBED_MODEL, "input": [text]}).encode()
    req = urllib.request.Request(base + "/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        v = np.asarray(json.load(r)["data"][0]["embedding"], dtype="float32")
    return v / (float((v ** 2).sum()) ** 0.5 + 1e-9)


def semantic_hits(author, query_text, k=8):
    """The author's k archived messages nearest in meaning to query_text:
    [(score, text), ...]. [] when the index or query is missing."""
    if not query_text or not available(author):
        return []
    V, texts = _load(author)
    q = _embed_one(query_text[:1000])
    sims = V @ q
    order = sims.argsort()[::-1][:k]
    return [(float(sims[i]), texts[i]) for i in order]


def burst_scores(axis_name, pct=90):
    """{author: z} where the raw score is the pct-th percentile of the
    author's per-message projections on the axis — 'how strong are their
    strongest moments', immune to dilution by neutral filler."""
    import numpy as np
    if axis_name in _burst_cache:
        return _burst_cache[axis_name]
    # same decorrelated dial as axis_scores/traits_for: built-ins use the Löwdin
    # ortho directions, custom axes project on their own raw direction.
    from utils.persona_axes import _ortho_builtin, _all_axis_vectors
    ortho = _ortho_builtin()
    av = ortho[axis_name] if axis_name in ortho else _all_axis_vectors()[axis_name][0]
    raw = {}
    for f in os.listdir(DIR):
        if not f.endswith(".npz"):
            continue
        author = f[:-4]
        V, _texts = _load(author)
        raw[author] = float(np.percentile(V @ av, pct))
    if not raw:
        return {}
    vals = np.array(list(raw.values()))
    z = (vals - vals.mean()) / (vals.std() or 1.0)
    out = dict(zip(raw.keys(), z))
    _burst_cache[axis_name] = out
    return out


def contradiction_scores(axis_name, lo_pct=10, hi_pct=90):
    """{author: z} for self-contradiction on an axis: how much a person occupies
    BOTH poles at once. A sincere chatter clusters at one end; a performative /
    ironic one has messages at both (the user's 'high in both feminism and
    misogyny' signal). Computed as sqrt(frac above the global hi-pole percentile
    * frac below the global lo-pole percentile), z-scored across the roster.

    No oracle, no single-message intent call: it reads the per-message cloud the
    mean-pool throws away. A high score means the person's CHARGED-axis mean is
    unreliable — they perform both ends. Same axis selection as burst_scores."""
    import numpy as np
    if axis_name in _contra_cache:
        return _contra_cache[axis_name]
    from utils.persona_axes import _ortho_builtin, _all_axis_vectors
    ortho = _ortho_builtin()
    av = ortho[axis_name] if axis_name in ortho else _all_axis_vectors()[axis_name][0]
    proj = {}
    for f in os.listdir(DIR):
        if not f.endswith(".npz"):
            continue
        author = f[:-4]
        V, _texts = _load(author)
        proj[author] = np.asarray(V @ av, dtype="float32")
    if not proj:
        return {}
    allp = np.concatenate(list(proj.values()))
    lo, hi = float(np.percentile(allp, lo_pct)), float(np.percentile(allp, hi_pct))
    raw = {a: float(np.sqrt(((p > hi).mean()) * ((p < lo).mean()))) for a, p in proj.items()}
    vals = np.array(list(raw.values()))
    z = (vals - vals.mean()) / (vals.std() or 1.0)
    out = dict(zip(raw.keys(), z))
    _contra_cache[axis_name] = out
    return out
