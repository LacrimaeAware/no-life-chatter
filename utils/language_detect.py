"""Local language detection via lingua — free, no API.

lingua is far more accurate than the old langdetect on short text, and crucially
its confidence scores are *calibrated*: junk/emotes/short-English score low, real
foreign sentences score high. That makes a confidence threshold actually work.

Detection is restricted to the languages the bot supports (config.SUPPORTED_LANGS),
which keeps it from guessing obscure languages. Codes are returned in the bot's
uppercase scheme (e.g. "EN", "ES", "ZH-CN").
"""

from lingua import Language, LanguageDetectorBuilder

import config

# bot code -> lingua Language
_CODE_TO_LANG = {
    "EN": Language.ENGLISH, "ES": Language.SPANISH, "FR": Language.FRENCH,
    "DE": Language.GERMAN, "PT": Language.PORTUGUESE, "IT": Language.ITALIAN,
    "RU": Language.RUSSIAN, "PL": Language.POLISH, "CS": Language.CZECH,
    "SK": Language.SLOVAK, "UK": Language.UKRAINIAN, "EL": Language.GREEK,
    "TR": Language.TURKISH, "HU": Language.HUNGARIAN, "VI": Language.VIETNAMESE,
    "AR": Language.ARABIC, "JA": Language.JAPANESE, "KO": Language.KOREAN,
    "ZH-CN": Language.CHINESE, "LA": Language.LATIN,
}

# lingua iso code -> bot code (only the ones that differ)
_ISO_TO_CODE = {"ZH": "ZH-CN"}

# Build a detector restricted to the configured supported languages. Always keep
# English in the set so English is a candidate (and gets correctly identified
# rather than forced into a foreign language).
_langs = {_CODE_TO_LANG[c] for c in config.SUPPORTED_LANGS if c in _CODE_TO_LANG}
_langs.add(Language.ENGLISH)
_detector = LanguageDetectorBuilder.from_languages(*_langs).build()


def detect_language(text: str) -> tuple[str, float]:
    """Return (LANG_CODE_UPPER, confidence 0..1), or ("", 0.0) if undetectable."""
    if not text or not text.strip():
        return "", 0.0
    values = _detector.compute_language_confidence_values(text)
    if not values:
        return "", 0.0
    top = values[0]
    code = top.language.iso_code_639_1.name  # e.g. "EN", "ZH"
    return _ISO_TO_CODE.get(code, code), float(top.value)


def detect_confidences(text: str) -> dict[str, float]:
    """Return {LANG_CODE_UPPER: confidence} across the supported languages.

    Used to ask "is this confidently NOT the target language?" — comparing the
    best foreign score against the target's own score, rather than trusting a
    single top guess (which is unreliable when several languages are close, e.g.
    Spanish vs Portuguese).
    """
    if not text or not text.strip():
        return {}
    out: dict[str, float] = {}
    for v in _detector.compute_language_confidence_values(text):
        code = v.language.iso_code_639_1.name
        out[_ISO_TO_CODE.get(code, code)] = float(v.value)
    return out
