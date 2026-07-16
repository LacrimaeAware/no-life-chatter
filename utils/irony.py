"""Evidence-layered irony reading for an arbitrary chat message.

Surface sarcasm/extremity embeddings are weak evidence. Community repetition,
literal plausibility, and confirmed personal history can strengthen or overturn
that first read. Emotes are resolved to usage-meaning words before embedding.
"""

import numpy as np

from utils import chat_archive, emote_meaning
from utils.persona_traits import _axis_vectors, _embed

_harm = None
_cal = {}


def _harm_axis():
    global _harm
    if _harm is None:
        pos = ['children deserve to suffer', 'i would enjoy hurting someone vulnerable',
               'causing pain to innocent people is good', 'i hope something terrible happens to them',
               'making others miserable brings me joy']
        neg = ['children deserve to be safe', 'i would enjoy helping someone vulnerable',
               'protecting innocent people is good', 'i hope something wonderful happens to them',
               'making others happy brings me joy']
        e = _embed(pos + neg)
        P = np.asarray(e[:5], dtype="float32").mean(axis=0)
        N = np.asarray(e[5:], dtype="float32").mean(axis=0)
        v = P - N
        _harm = v / (np.linalg.norm(v) + 1e-9)
    return _harm


def _calib(name, av):
    if name not in _cal:
        conn = chat_archive.connect()
        max_id = int(conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM messages"
        ).fetchone()[0])
        stride = max(1, max_id // 300)
        offset = 17 % stride
        msgs = [r[0] for r in conn.execute(
            "SELECT content FROM messages WHERE LENGTH(content) > 12 "
            "AND id % ? = ? ORDER BY id LIMIT 300", (stride, offset))]
        if len(msgs) < 100:
            msgs = [r[0] for r in conn.execute(
                "SELECT content FROM messages WHERE LENGTH(content) > 12 "
                "ORDER BY id LIMIT 300")]
        E = np.asarray(_embed(msgs), dtype="float32")
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        s = E @ av
        _cal[name] = (float(s.mean()), float(s.std()) or 1.0)
    return _cal[name]


def _resolve_emotes(text):
    """Replace recognized emotes with their usage-meaning words so the
    embedder sees the operator (DansGame -> 'disgust gross')."""
    out = []
    for tok in text.split():
        words = emote_meaning.meaning_words(tok, n=1)
        if words:
            out.append(tok + " (" + words[0][0].split()[0] + ")")
        else:
            out.append(tok)
    return " ".join(out)


def read(message, context=""):
    iron = np.asarray(_axis_vectors()["ironic"], dtype="float32")
    harm = _harm_axis()
    text = _resolve_emotes(message)
    if context:
        text = f"{context}. {text}"
    e = np.asarray(_embed([text])[0], dtype="float32")
    e /= (np.linalg.norm(e) + 1e-9)
    mi, si = _calib("ironic", iron)
    mh, sh = _calib("harm", harm)
    zi = (float(e @ iron) - mi) / si
    zh = (float(e @ harm) - mh) / sh
    if zi > 0.6:
        verdict = "reads IRONIC (marked sarcasm)"
    elif zh > 1.2:
        verdict = "reads DEADPAN-IRONIC (calm surface, extreme content)"
    elif zi < -0.6:
        verdict = "reads sincere"
    else:
        verdict = "unclear / mild"
    return verdict, zi, zh


def analyze(message: str, context: str = "", *, author: str | None = None,
            channel: str | None = None) -> dict:
    """Layer history/community evidence over the weak surface-axis read."""
    verdict, zi, zh = read(message, context)
    echo = chat_archive.community_echo_stats(
        message, author=author, channel=channel
    )
    try:
        from utils import user_profiles
        consistency = user_profiles.claim_consistency(author, message) if author else {
            "claims": user_profiles.claims_in_text(message),
            "conflicts": [],
            "agreements": [],
        }
    except Exception:
        consistency = {"claims": [], "conflicts": [], "agreements": []}

    unusual = [
        claim for claim in consistency.get("claims", [])
        if claim.get("plausibility") in {"unusual", "impossible"}
    ]
    conflicts = consistency.get("conflicts", [])
    reasons = []
    if echo.get("authors", 0) >= 2:
        reasons.append(f"near-copy used by {echo['authors']} other chatters")
    elif echo.get("matches", 0) >= 2:
        reasons.append(f"repeated {echo['matches']} times in the archive")
    if conflicts:
        slots = ",".join(sorted({item["slot"] for item in conflicts}))
        reasons.append(f"conflicts with confirmed {slots} history")
    if unusual:
        level = "impossible" if any(
            claim.get("plausibility") == "impossible" for claim in unusual
        ) else "unusual"
        reasons.append(f"{level} literal claim")

    if conflicts and (unusual or echo.get("authors", 0) >= 1):
        verdict = "likely a bit / nonliteral"
        confidence = "high"
    elif echo.get("authors", 0) >= 3 and echo.get("exact", 0) >= 3:
        verdict = "likely a repeated bit"
        confidence = "medium"
    elif unusual and (zi > 0.3 or echo.get("matches", 0) >= 2):
        verdict = "probably nonliteral"
        confidence = "medium"
    else:
        confidence = "low" if not reasons else "medium"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "sarcasm": zi,
        "extremity": zh,
        "reasons": reasons,
        "echo": echo,
        "consistency": consistency,
    }


def format_analysis(result: dict, max_chars: int = 470) -> str:
    reasons = list(result.get("reasons") or [])
    reasons.append(
        f"surface sarcasm {result.get('sarcasm', 0.0):+.1f}, "
        f"extremity {result.get('extremity', 0.0):+.1f}"
    )
    text = (
        f"{result.get('verdict', 'unclear')} [{result.get('confidence', 'low')} confidence]"
        f" | " + "; ".join(reasons)
    )
    return text if len(text) <= max_chars else text[:max_chars - 3] + "..."
