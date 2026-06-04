# utils/romanize.py
# Public helpers:
#   supported_languages() -> list[str]       # romanizable langs only
#   is_supported(lang_code: str) -> bool
#   romanize(lang_code: str, text: str) -> str | None
#   thai_segment(text: str) -> str | None

from typing import List, Optional

# Import per-language romanizers
from utils.roman_scripts import thai as _th      # requires PyThaiNLP on the server
from utils.roman_scripts import korean as _ko    # your KO romanizer
from utils.roman_scripts import japanese as _ja  # your JA romanizer

# >>> Only these languages get an L-ROM whisper <<<
_SUPPORTED_ROMAN = {"TH", "KO", "JA"}

def supported_languages() -> List[str]:
    return sorted(_SUPPORTED_ROMAN)

def is_supported(lang_code: str) -> bool:
    return (lang_code or "").upper() in _SUPPORTED_ROMAN

def romanize(lang_code: str, text: str) -> Optional[str]:
    if not text:
        return ""
    lang = (lang_code or "").upper()
    if lang == "TH":
        return _th.romanize(text)
    if lang == "KO":
        return _ko.romanize(text)
    if lang == "JA":
        return _ja.romanize(text)
    return None

def thai_segment(text: str) -> Optional[str]:
    """Thai word segmentation (spaces) or None if unavailable."""
    if not text:
        return ""
    try:
        seg = _th.segment(text)
        # _th.segment returns a string (segmented) or original; treat empty as None
        return seg if isinstance(seg, str) and seg else None
    except Exception:
        return None
