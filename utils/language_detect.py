"""Free, local language detection — no API, no per-message cost.

Uses `langdetect`, which runs in-process (like the romanizers) wherever the bot
runs. This removes the paid Google "detect language" call; only the actual
translation still uses a paid API.

It returns codes in the same uppercase scheme the rest of the bot uses
(e.g. "EN", "JA", "ZH-CN"), so detection and translation agree on codes.
"""

from langdetect import DetectorFactory, LangDetectException, detect_langs

# Make detection deterministic (langdetect is randomized by default).
DetectorFactory.seed = 0

# langdetect emits ISO 639-1 codes (plus zh-cn / zh-tw). Uppercasing them lines
# them up with the bot's code scheme; this table covers the few special cases.
_ALIASES = {
    "ZH": "ZH-CN",
}


def detect_language(text: str) -> tuple[str, float]:
    """Detect the language of ``text``.

    Returns ``(LANG_CODE_UPPER, confidence)`` where confidence is 0..1, or
    ``("", 0.0)`` if the language can't be determined.
    """
    if not text or not text.strip():
        return "", 0.0
    try:
        results = detect_langs(text)  # already sorted, highest probability first
    except LangDetectException:
        return "", 0.0
    if not results:
        return "", 0.0
    top = results[0]
    code = top.lang.upper()
    return _ALIASES.get(code, code), float(top.prob)
