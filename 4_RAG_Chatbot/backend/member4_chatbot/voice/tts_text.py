"""Text normalisation before speaking: what the TTS reads must be plain words in one language per sentence.

Steps (for any answer text, Sinhala or English):
1. Strip markup: URLs, markdown (**, #, `), citation markers ([1], (p. 3)), footnote asterisks.
2. Expand dotted Sinhala abbreviations (කි.ග්‍රෑ., ටී.එස්.පී.), so their full stops do not end a sentence.
3. Split into sentences: each line, and each full stop / question mark followed by a space. A table row
   "a | b | c" becomes one sentence "a, b, c".
4. Decide each sentence's language from its letters. A Sinhala answer can quote an English DOA passage; that part
   is spoken by the English voice.
5. Normalise each sentence in its language:
   - codes (node A3-06, escalation ESC-C5F12D, varieties Bg 300) are spelled out
   - number + unit (kg/ac, mg/kg, t/ha, dS/m, °C, ...) is read with the unit, from data/voice/tts_terms.yaml
   - terms (urea, TSP, MOP, dolomite, pH, N/P/K, ...)
   - numbers, decimals, fractions, ranges, %, dates and times become words (sinhala_numbers.py / num2words)
6. Report anything left over that the voice cannot read (Latin letters in a Sinhala sentence, digits).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import yaml

from member4_chatbot.lang import script_counts
from member4_chatbot.paths import VOICE_DATA_DIR
from member4_chatbot.voice.sinhala_numbers import _NUMF, digits_to_words, numbers_to_words

TERMS_YAML = VOICE_DATA_DIR / "tts_terms.yaml"
ZWJ = "\u200d"
_NUMBER = r"(?:[<>]\s*)?\d+(?:[.,]\d+)*(?:\s+\d+/\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?(?:\s+\d+/\d+)?)?"


@dataclass
class Sentence:
    lang: str  # "si" or "en": which voice speaks it
    source: str  # the sentence as it was in the answer
    text: str  # what the voice reads
    leftovers: list[str] = field(default_factory=list)  # characters / words the voice cannot read


@lru_cache(maxsize=1)
def _terms() -> dict:
    data = yaml.safe_load(TERMS_YAML.read_text(encoding="utf-8"))
    data["_units"] = [(re.compile(rf"(?<![\w.])({_NUMBER})\s*(?:{u['pattern']})"), u) for u in data["units"]]
    data["_alone"] = [(re.compile(rf"(?<![A-Za-z])(?:{u['pattern']})"), u["alone"]) for u in data["units"] if u.get("alone")]
    data["_terms"] = [(re.compile(t["match"]), t) for t in data["terms"]]
    data["_abbr"] = [(re.compile(a["match"]), a["say"]) for a in data["sinhala_abbreviations"]]
    return data


@lru_cache(maxsize=1)
def terms_version() -> str:
    """Changes when tts_terms.yaml or the normalisation code changes (part of the TTS cache key)."""
    from pathlib import Path

    from member4_chatbot.voice import sinhala_numbers

    h = hashlib.sha1(TERMS_YAML.read_bytes())
    for src in (Path(__file__), Path(sinhala_numbers.__file__)):
        h.update(src.read_bytes())
    return h.hexdigest()[:10]


# ---- 1-3: markup, abbreviations, sentences ---------------------------------------------------------------
_URL = re.compile(r"https?://\S+|www\.\S+")
_MD = [(re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),  # [text](link) -> text
       (re.compile(r"\[\d+(?:[,–-]\s*\d+)*\]"), ""),  # [1], [2-3] citation markers
       (re.compile(r"\((?:p|pp|page)\.?\s*\d+[^)]*\)", re.I), ""),  # (p. 3)
       (re.compile(r"(\*\*|__|`+|^#+\s*|^\s*[-*•]\s+)", re.M), ""),  # bold, code, headings, bullets
       (re.compile(r"(?<=\w)\*+|\*+(?=\s|$)"), "")]  # footnote asterisks: "Time *"


def clean_markup(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _URL.sub("", text)
    for pattern, repl in _MD:
        text = pattern.sub(repl, text)
    return text


def expand_sinhala_abbreviations(text: str) -> str:
    for pattern, say in _terms()["_abbr"]:
        text = pattern.sub(say + " ", text)
    return text


def split_sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line:  # table row
            cells = [c.strip(" :") for c in line.split("|")]
            line = ", ".join(c for c in cells if c) + "."
        for part in re.split(r"(?<=[.?!।])\s+|(?<=:)\s+(?=[^\d\s])(?=.{40,})", line):
            if re.search(r"[\w\u0D80-\u0DFF]", part):
                out.append(part.strip())
    return out


def sentence_lang(sentence: str) -> str:
    si, latin = script_counts(sentence)
    return "si" if si and si >= latin * 0.5 else "en"


# ---- 5: per-language normalisation ----------------------------------------------------------------------
def _spell(code: str, lang: str) -> str:
    letters = _terms()["letters_si"]
    words = []
    for ch in code:
        if ch.isdigit():
            words.append(digits_to_words(ch) if lang == "si" else _en_digit(ch))
        elif ch.isalpha():
            words.append(letters.get(ch.upper(), ch) if lang == "si" else ch.upper())
    return " ".join(words)


def _en_digit(d: str) -> str:
    return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][int(d)]


def _number_words(n: str, lang: str) -> str:
    return numbers_to_words(n) if lang == "si" else en_numbers_to_words(n)


_CODES = [
    # escalation reference ESC-C5F12D
    (re.compile(r"\bESC-([0-9A-F]{4,8})\b"), lambda m, lang: _spell("ESC", lang) + " " + _spell(m[1], lang)),
    # node ids A3-06 and the plot name A-3
    (re.compile(r"\b([A-Z])(\d)-(\d{2})\b"), lambda m, lang: _spell(m[1] + m[2] + m[3], lang)),
    (re.compile(r"\b([A-Z])-(\d{1,2})\b"), lambda m, lang: _spell(m[1], lang) + " " + _number_words(m[2], lang)),
    # rice varieties: Bg 300, Bg 94-1, At 362, Ld 365, Bw 367, H 4
    (re.compile(r"\b(Bg|BG|Bw|BW|Ld|LD|At|AT|Bl|BL|H)\s?(\d{1,4})(?:\s?-\s?(\d{1,2}))?\b"),
     lambda m, lang: _spell(m[1], lang) + " " + _number_words(m[2], lang)
     + (" " + _number_words(m[3], lang) if m[3] else "")),
]


@lru_cache(maxsize=1)
def _one_pass() -> tuple[re.Pattern, list]:
    """Terms as ONE alternation, so a replacement is never matched again by a later rule
    (e.g. "TSP" -> "T S P" must not then turn its "P" into "phosphorus")."""
    rules = [(p, (lambda t: lambda m, lang: t[lang])(t)) for p, t in _terms()["_terms"]]
    big = re.compile("|".join(f"(?P<r{i}>{p.pattern})" for i, (p, _) in enumerate(rules)))
    return big, rules


def _apply_common(s: str, lang: str) -> str:
    # codes first: "ESC-3F9A1C" must not have its "1C" read as 1 degree Celsius by the unit rules
    for pattern, fn in _CODES:
        s = pattern.sub(lambda m: fn(m, lang), s)
    t = _terms()
    for pattern, unit in t["_units"]:
        s = pattern.sub(lambda m: unit[lang].replace("{n}", _number_words(m[1], lang)), s)
    for pattern, alone in t["_alone"]:
        s = pattern.sub(alone[lang], s)
    big, rules = _one_pass()

    def repl(m: re.Match) -> str:
        i = int(m.lastgroup[1:]) if m.lastgroup and m.lastgroup.startswith("r") else next(
            k for k in range(len(rules)) if m.group(f"r{k}") is not None)
        pattern, fn = rules[i]
        return fn(pattern.match(m.group(0)), lang)

    return big.sub(repl, s)


def normalize_si(sentence: str) -> tuple[str, list[str]]:
    s = expand_sinhala_abbreviations(sentence)
    s = _apply_common(s, "si")
    s = numbers_to_words(s)
    # unknown acronyms (2-6 capitals) are spelled with Sinhala letter names
    s = re.sub(r"\b[A-Z]{2,6}\b", lambda m: _spell(m[0], "si"), s)
    s = s.replace("/", " ").replace("–", ", ").replace("—", ", ")
    s = re.sub(r"\s+", " ", s).strip()
    leftovers = re.findall(r"[A-Za-z]+|\d+", s)
    return s, leftovers


def normalize_en(sentence: str) -> tuple[str, list[str]]:
    s = _apply_common(sentence, "en")
    s = en_numbers_to_words(s)
    s = re.sub(r"\b([A-Z]{2,6})\b", lambda m: " ".join(m[1]) if m[1] not in {"OK", "AM", "PM"} else m[1], s)
    s = s.replace("/", " ").replace("–", ", ").replace("—", ", ").replace("&", " and ")
    s = re.sub(r"\s+", " ", s).strip()
    leftovers = re.findall(r"\d+|[\u0D80-\u0DFF]+", s)
    return s, leftovers


# ---- English numbers ---------------------------------------------------------------------------------------
_EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]


def _n2w(x: str) -> str:
    from num2words import num2words

    x = x.replace(",", "")
    return num2words(float(x) if "." in x else int(x), lang="en").replace(",", "")


def _en_fraction(whole: str | None, num: int, den: int) -> str:
    named = {(1, 2): "a half", (1, 4): "a quarter", (3, 4): "three quarters"}.get((num, den))
    frac = named or f"{_n2w(str(num))} over {_n2w(str(den))}"
    if whole:
        return f"{_n2w(whole)} and {frac}"
    return {"a half": "one half", "a quarter": "one quarter"}.get(frac, frac)


_EN_PATTERNS: list[tuple[re.Pattern, object]] = [
    (re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{1,2}):(\d{2}))?(?!\d)"),
     lambda m: f"{_n2w(str(int(m[3])))} {_EN_MONTHS[int(m[2]) - 1]} {_year(m[1])}"
     + (f" at {_en_time(int(m[4]), int(m[5]))}" if m[4] else "") if 1 <= int(m[2]) <= 12 else m[0]),
    (re.compile(r"(?<![\d.:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])"), lambda m: _en_time(int(m[1]), int(m[2]))),
    (re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d,])"), lambda m: m[0].replace(",", "")),
    (re.compile(rf"(?<![\d.\-/])({_NUMF})\s*[-–]\s*({_NUMF})(?![\d.\-/])"),
     lambda m: f"{_en_any(m[1])} to {_en_any(m[2])}"),
    (re.compile(r"(?<![\d.])(?:(\d+)\s*)?([½¼¾])"),
     lambda m: _en_fraction(m[1], *{"½": (1, 2), "¼": (1, 4), "¾": (3, 4)}[m[2]])),
    (re.compile(r"(?<![\d.])(?:(\d+)\s+)?(\d+)/(\d+)(?![\d/])"),
     lambda m: _en_fraction(m[1], int(m[2]), int(m[3])) if int(m[3]) else m[0]),
    (re.compile(r"(\d+(?:\.\d+)?)\s*%"), lambda m: f"{_n2w(m[1])} percent"),
    (re.compile(r"([<>])\s*(\d+(?:\.\d+)?)"), lambda m: f"{'more' if m[1] == '>' else 'less'} than {_n2w(m[2])}"),
    (re.compile(r"(?<![\d.\-])(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)(?![\d.\-])"),
     lambda m: f"{_n2w(m[1])} to {_n2w(m[2])}"),
    (re.compile(r"\d+(?:\.\d+)?"), lambda m: _n2w(m[0])),
]


def _en_any(x: str) -> str:
    """'4', '5.5', '4 1/2', '1/2', '2½' -> English words."""
    x = x.strip()
    if x[-1] in "½¼¾":
        return _en_fraction(x[:-1].strip() or None, *{"½": (1, 2), "¼": (1, 4), "¾": (3, 4)}[x[-1]])
    m = re.fullmatch(r"(?:(\d+)\s+)?(\d+)/(\d+)", x)
    if m and int(m[3]):
        return _en_fraction(m[1], int(m[2]), int(m[3]))
    return _n2w(x)


def _year(y: str) -> str:
    from num2words import num2words

    return num2words(int(y), lang="en", to="year")


def _en_time(h: int, mi: int) -> str:
    return f"{_n2w(str(h))} hundred hours" if mi == 0 else f"{_n2w(str(h))} {_n2w(str(mi)) if mi >= 10 else 'oh ' + _n2w(str(mi))}"


def en_numbers_to_words(text: str) -> str:
    for pattern, repl in _EN_PATTERNS:
        text = pattern.sub(repl, text)
    return text


# ---- everything together --------------------------------------------------------------------------------------
def prepare(text: str) -> list[Sentence]:
    """Answer text -> sentences ready to speak, each with its language and any leftovers."""
    text = expand_sinhala_abbreviations(clean_markup(text))
    answer_lang = sentence_lang(text)
    out = []
    for source in split_sentences(text):
        lang = sentence_lang(source)
        # a sentence with no Sinhala letters and no English word of 4+ letters ("pH 5.9.", "Bg 300", "45 kg") is
        # a code or a number: it is read in the language of the whole answer ("2 Weeks, 50." stays English)
        if lang == "en" and answer_lang == "si" and (re.search(r"[඀-෿]", source)
                                                     or not re.search(r"[A-Za-z]{4,}", source)):
            lang = "si"
        spoken, leftovers = normalize_si(source) if lang == "si" else normalize_en(source)
        spoken = spoken.replace(ZWJ + " ", " ")
        if re.search(r"[\w\u0D80-\u0DFF]", spoken):
            out.append(Sentence(lang, source, spoken, leftovers))
    return out


def speak_parts(text: str, per_part: int = 2) -> tuple[list[str], int]:
    """The text split for playback in parts (Phase 7): the first part is one sentence (sound starts quickly), then
    `per_part` sentences each. The split is the one prepare() uses (markup removed, abbreviations expanded so their
    dots do not end a sentence), so every part spoken alone sounds as it would inside the whole answer.
    Returns (parts, number of sentences)."""
    sentences = split_sentences(expand_sinhala_abbreviations(clean_markup(text)))
    if not sentences:
        return [], 0
    parts = [sentences[0]]
    for i in range(1, len(sentences), per_part):
        parts.append(" ".join(sentences[i:i + per_part]))
    return parts, len(sentences)
