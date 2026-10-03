"""Language detection for farmer questions: Sinhala script, English, or Sinhala typed in Latin letters ("Singlish").

Script counting is enough to separate Sinhala from English and is exact, fast and explainable, so no model is
used. Singlish ("pohora kiyada danna one") is common in Sri Lankan chat; it is recognised with a small list of
frequent romanised Sinhala words and reported as `si_latn`, so later phases can answer in Sinhala.
"""

from __future__ import annotations

import re
import unicodedata

from member4_chatbot.ingest.sinhala import SINHALA_END, SINHALA_START

SINGLISH_WORDS = {
    "mata", "mage", "mama", "oya", "oyage", "kohomada", "mokakda", "mokadda", "monawada", "kiyada", "keeyada",
    "karanna", "karanne", "danna", "denna", "dannada", "one", "ona", "oney", "thiyenawa", "tiyenawa", "thiyanawa",
    "pohora", "kumbura", "kumburu", "wee", "vee", "goyam", "wagawa", "wagaawa", "yuriya", "pasa", "wathura",
    "watura", "wal", "rogaya", "rogayak", "kollo", "kawda", "kawuda", "ayubowan", "kiyanna", "puluwanda",
    "puluwan", "nadda", "neda", "da", "dha", "eka", "ekak", "kiyak", "kochchara", "kochcharada", "wela", "welawa",
    "yanna", "giya", "hoda", "naha", "nehe", "ow", "ou", "bohoma", "sthuthi", "isthuthi", "monada", "ai",
}


def script_counts(text: str) -> tuple[int, int]:
    si = latin = 0
    for c in text:
        if SINHALA_START <= ord(c) <= SINHALA_END:
            si += 1
        elif c.isascii() and c.isalpha():
            latin += 1
        elif unicodedata.category(c).startswith("L") and "LATIN" in unicodedata.name(c, ""):
            latin += 1
    return si, latin


def detect_language(text: str) -> str:
    """'si' (Sinhala script), 'en' (English), 'si_latn' (Singlish) or 'unknown' (no letters)."""
    si, latin = script_counts(text)
    if si == 0 and latin == 0:
        return "unknown"
    if si >= latin * 0.5 or si >= 4 and si > latin * 0.3:
        return "si"
    words = re.findall(r"[a-z]+", text.lower())
    if words and sum(w in SINGLISH_WORDS for w in words) / len(words) >= 0.3:
        return "si_latn"
    return "en"
