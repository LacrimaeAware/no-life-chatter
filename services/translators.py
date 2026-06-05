"""Translation providers.

Right now: DeepL (free tier, header-based auth). Kept separate so a second
provider (e.g. Google) can be added and rotated later to spread free-tier usage.
"""

import logging

import requests

import config

_DEEPL_URL = "https://api-free.deepl.com/v2/translate"

# DeepL target codes differ slightly from the bot's internal codes.
_DEEPL_TARGET = {
    "EN": "EN-US",
    "PT": "PT-PT",
    "ZH-CN": "ZH",
    "ZH": "ZH",
}


def deepl_translate(text: str, target: str) -> str | None:
    """Translate via DeepL. Returns None if no key is set or on any failure."""
    key = config.DEEPL_API_KEY
    if not key:
        return None
    tgt = _DEEPL_TARGET.get(target.upper(), target.upper())
    try:
        resp = requests.post(
            _DEEPL_URL,
            headers={"Authorization": "DeepL-Auth-Key " + key},
            data={"text": text, "target_lang": tgt},
            timeout=15,
        )
    except Exception as e:
        logging.warning(f"DeepL request error: {e}")
        return None

    if resp.status_code == 200:
        try:
            return resp.json()["translations"][0]["text"]
        except Exception:
            return None

    logging.warning(f"DeepL failed ({resp.status_code}): {resp.text[:200]}")
    return None
