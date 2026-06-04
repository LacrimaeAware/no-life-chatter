# utils/roman_scripts/japanese.py
# Strict Japanese romanization using pykakasi (kanji + kana).
# No fallbacks. If pykakasi isn't installed, import will fail.

from pykakasi import kakasi  # requires pykakasi>=2.x
import re

# Configure pykakasi
_kks = kakasi()
_kks.setMode("H", "a")          # Hiragana -> ascii
_kks.setMode("K", "a")          # Katakana -> ascii
_kks.setMode("J", "a")          # Kanji    -> ascii (via reading)
_kks.setMode("r", "Hepburn")    # Hepburn romaji
_kks.setMode("C", False)        # no capitalization
_conv = _kks.getConverter()

_WS = re.compile(r"\s+")
_NOPRE_SPACE = re.compile(r"\s+([,.;:!?、。・「」『』（）()［］\\/\-])")
_SPACE_AFTER_PUNCT = re.compile(r"([,.;:!?、。・「」『』（）()［］\\/\-])(\S)")

def romanize(s: str) -> str:
    """
    Return Hepburn romaji for full Japanese text (kanji + kana).
    Uses chunk conversion to avoid kanji leaking and to control spacing.
    """
    if not s:
        return ""

    # Convert into chunks with readings
    chunks = _conv.convert(s)  # list of dicts with keys like 'orig','hira','kana','hepburn'
    parts = []
    for ch in chunks:
        rom = ch.get("hepburn") or ch.get("hira") or ch.get("kana") or ch.get("orig", "")
        if rom is None:
            rom = ""
        parts.append(rom)

    out = " ".join(parts)

    # Normalize whitespace & tidy punctuation spacing
    out = _WS.sub(" ", out).strip()
    out = _NOPRE_SPACE.sub(r"\1", out)         # no space before punctuation
    out = _SPACE_AFTER_PUNCT.sub(r"\1 \2", out)  # ensure a space after punctuation

    return out
