# utils/roman_scripts/korean.py
# Korean Revised Romanization via Hangul decomposition (basic).

def romanize(text: str) -> str:
    if not text:
        return ""
    out = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:  # Hangul syllables
            idx = code - 0xAC00
            c = idx // 588
            v = (idx % 588) // 28
            f = idx % 28
            out.append(_CHOSEONG[c] + _JUNGSEONG[v] + _JONGSEONG[f])
        else:
            out.append(ch)
    return ''.join(out)

_CHOSEONG = [
    "g","kk","n","d","tt","r","m","b","pp","s","ss","","j","jj","ch","k","t","p","h"
]
_JUNGSEONG = [
    "a","ae","ya","yae","eo","e","yeo","ye","o","wa","wae","oe",
    "yo","u","wo","we","wi","yu","eu","ui","i"
]
_JONGSEONG = [
    "", "k","k","k","n","n","n","t","l","lk","lm","lb","ls","lt","lp","lh",
    "m","p","p","t","t","ng","t","t","k","t","p","t"
]
