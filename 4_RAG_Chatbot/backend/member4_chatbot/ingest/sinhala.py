"""Sinhala text checks: how much of a text is Unicode Sinhala, whether it is readable, and legacy-font repair.

Many Sri Lankan PDFs are not Unicode. Two cases matter:
1. **Legacy fonts** (FM Abhaya and similar): Sinhala letters are drawn with Latin code points, so the text layer
   reads like `,xld` instead of `ලංකා`. A character map can turn it back into Unicode.
2. **Scanned pages**: the "text" is a picture. Some scanners add their own Latin OCR layer, which is garbage for
   Sinhala (`tslbOrD I OcolOd ...`). Only OCR with a Sinhala model (Tesseract `sin`) can read these.

`assess()` decides which case a text is in. Mapping is tried first; it is kept only if the result reads as
Sinhala (enough common Sinhala words), otherwise the caller falls back to OCR.
"""

from __future__ import annotations

import re
import unicodedata

SINHALA_START, SINHALA_END = 0x0D80, 0x0DFF
ZWJ = "\u200d"

# Very common Sinhala function words and paddy words. A readable Sinhala text hits several of these per 100
# words; legacy-font mojibake and scanner garbage hit none.
COMMON_SI_WORDS = {
    "සහ", "හා", "වේ", "වන", "ඇති", "කර", "කරන", "කිරීම", "සඳහා", "මෙම", "ලෙස", "විට", "නම්",
    "නොවේ", "පසු", "පෙර", "තුළ", "මගින්", "අතර", "වැනි", "යුතුය", "හැක", "හෝ", "එම", "මෙය", "සිට", "දක්වා",
    "වී", "වගාව", "කුඹුරු", "පොහොර", "පස", "ජලය", "යූරියා", "දින", "සති", "කලාපය", "වල්", "රෝග", "පළිබෝධ",
    "කිලෝග්\u200dරෑම්", "හෙක්ටයාරයට", "අක්කරයට", "බීජ", "ශාක", "පත්\u200dර", "මාස", "අවශ්\u200dය", "ප්\u200dරමාණය",
}

# Partial FM-Abhaya-style map (Latin code point -> Sinhala). Only letters whose FM Abhaya positions are well
# established are listed (e.g. `Y%S ,xld` = ශ්\u200dරී ලංකා). It is NOT a complete converter: none of the current
# documents use a legacy font (see phase-5.md), so it has not been tested on a real FM document. `assess()`
# rejects a mapping that does not produce readable Sinhala, so an incomplete map cannot slip garbage through.
FM_PREFIX_VOWEL = "f"  # ෙ (kombuva) is typed BEFORE the consonant in FM fonts; Unicode puts it after.
FM_MAP = {
    "%": "්" + ZWJ + "ර",  # rakaransaya ්\u200dර
    "w": "අ", "l": "ක", ".": "ග", "p": "ච", "c": "ජ", "g": "ට", "v": "ඩ", "K": "ණ", "y": "ත", "o": "ද",
    "O": "ධ", "k": "න", "m": "ප", "n": "බ", "N": "භ", "u": "ම", "h": "ය", "r": "ර", ",": "ල", "j": "ව",
    "Y": "ශ", "i": "ස", "<": "ළ", "x": "ං",
    "a": "්", "d": "ා", "e": "ැ", "E": "ෑ", "s": "ි", "S": "ී", "q": "ු", "Q": "ූ", "f": "ෙ",
}


def letters(text: str) -> list[str]:
    return [c for c in text if unicodedata.category(c)[0] in "LM"]


def sinhala_share(text: str) -> float:
    """Share of letters (and combining marks) that are in the Sinhala block U+0D80-U+0DFF."""
    ls = letters(text)
    if not ls:
        return 0.0
    return sum(SINHALA_START <= ord(c) <= SINHALA_END for c in ls) / len(ls)


def common_word_rate(text: str) -> float:
    """Share of Sinhala-script words that are in COMMON_SI_WORDS (readable Sinhala: about 0.1-0.3)."""
    words = [w for w in re.split(r"[\s\.,;:!\?\(\)\[\]\"'“”‘’\-–|/]+", text) if w]
    si_words = [w for w in words if sinhala_share(w) > 0.5 and len(w) >= 2]
    if not si_words:
        return 0.0
    return sum(w in COMMON_SI_WORDS for w in si_words) / len(si_words)


def invalid_sign_rate(text: str) -> float:
    """Share of Sinhala vowel signs / virama that do not follow a consonant (or virama + ZWJ).

    In real Sinhala a dependent sign always sits on a consonant, so this is near 0. Legacy-font mojibake and
    mapped scanner garbage put signs at word starts and after other signs, giving 0.2 or more.
    """
    signs = bad = 0
    prev = " "
    for c in text:
        o = ord(c)
        if 0x0DCA <= o <= 0x0DDF or o in (0x0DF2, 0x0DF3, 0x0D82, 0x0D83):
            signs += 1
            p = ord(prev)
            if not (0x0D9A <= p <= 0x0DC6 or (o in (0x0D82, 0x0D83) and 0x0DCF <= p <= 0x0DDF)):
                bad += 1
        prev = c if c != ZWJ else prev
    return bad / signs if signs else 0.0


def fm_to_unicode(text: str) -> str:
    """Map FM-Abhaya-style legacy text to Unicode Sinhala (partial map, see FM_MAP)."""
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == FM_PREFIX_VOWEL and i + 1 < len(text) and text[i + 1] in FM_MAP:
            # f + consonant [+ d] -> consonant + ෙ [ා => ො]
            cons = FM_MAP[text[i + 1]]
            j = i + 2
            if j < len(text) and text[j] == "d":
                out.append(cons + "ො")
                j += 1
            elif j < len(text) and text[j] == "a":
                out.append(cons + "ේ")
                j += 1
            else:
                out.append(cons + "ෙ")
            i = j
            continue
        out.append(FM_MAP.get(ch, ch))
        i += 1
    return unicodedata.normalize("NFC", "".join(out))


def assess(text: str, expected_lang: str) -> dict:
    """How readable `text` is for a document expected to be in `expected_lang` ('si', 'en' or 'unknown').

    Returns {"ok": bool, "sinhala_share": float, "common_word_rate": float, "reason": str}.
    """
    share = sinhala_share(text)
    rate = common_word_rate(text)
    n_letters = len(letters(text))
    if n_letters < 30:
        return {"ok": False, "sinhala_share": share, "common_word_rate": rate, "reason": "no text layer"}
    if expected_lang == "si" or share > 0.2:
        bad = invalid_sign_rate(text)
        ok = share >= 0.5 and rate >= 0.05 and bad <= 0.05
        reason = ("readable Sinhala" if ok else "low Sinhala share" if share < 0.5
                  else f"broken letter sequences ({bad:.0%} of vowel signs)" if bad > 0.05
                  else "Sinhala letters but few real words")
        return {"ok": ok, "sinhala_share": share, "common_word_rate": rate, "reason": reason}
    en_rate = english_word_rate(text)
    ok = en_rate >= 0.10
    return {"ok": ok, "sinhala_share": share, "common_word_rate": en_rate,
            "reason": "readable English" if ok else "Latin letters but few English words (legacy font or scanner OCR)"}


COMMON_EN_WORDS = {
    "the", "of", "and", "to", "in", "a", "is", "for", "be", "on", "with", "are", "as", "by", "at", "or", "from",
    "this", "that", "it", "rice", "paddy", "should", "can", "which", "after", "before", "fertilizer", "plant",
    "plants", "field", "water", "weeds", "soil", "days", "weeks", "per", "applied", "application", "variety",
}


def english_word_rate(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return 0.0
    return sum(w in COMMON_EN_WORDS for w in words) / len(words)
