# utils/roman_scripts/thai.py
# Minimal Thai helpers using PyThaiNLP only.
# - segment(text): Thai word segmentation via newmm
# - romanize(text): romanize with thai2rom (fallback to royin)
#
# If anything raises, we return the input (no custom fallbacks).

from pythainlp.transliterate import romanize as _thai_romanize
from pythainlp.tokenize import word_tokenize as _thai_word_tokenize

def segment(text: str) -> str:
    """Return Thai text segmented with spaces (newmm)."""
    if not text:
        return ""
    try:
        return " ".join(_thai_word_tokenize(text, engine="newmm"))
    except Exception:
        # keep it simple: just return input on any issue
        return text

def romanize(text: str) -> str:
    """Romanize Thai using thai2rom; fallback to royin if needed."""
    if not text:
        return ""
    seg = segment(text) or text

    # Preferred engine: thai2rom (DL-based). Some versions need extra corpora.
    try:
        out = _thai_romanize(seg, engine="thai2rom")
        if out and out != seg:
            return out
    except Exception:
        pass

    # Fallback engine: royin (RTGS/official)
    try:
        out = _thai_romanize(seg, engine="royin")
        if out and out != seg:
            return out
    except Exception:
        pass

    # If engines return unchanged or error, just return segmented Thai
    return seg
