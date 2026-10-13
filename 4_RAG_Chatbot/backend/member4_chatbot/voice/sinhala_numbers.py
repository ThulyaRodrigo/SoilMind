"""Numbers, decimals, fractions, ranges, percentages, dates and times written out as Sinhala words, for TTS.

SinhalaVITS reads Sinhala letters only, so "45" or "2026-10-15" must become words before it speaks. Sinhala
numerals have two forms:
- a final form, used at the end of a number (පහ "five", විස්ස "twenty", සියය "hundred", දහස "thousand")
- a combining form, used before another part (පන්සිය "five hundred", විසිපහ "twenty-five", දහස් පහ "1,005")

So 45 is හතළිස්පහ, 500 පන්සියය, 2026 දෙදහස් විසිහය, 25,000 විසිපන්දහස.

Wording drafted by the AI coding assistant from standard Sinhala numerals; it must be checked by a Sinhala speaker
(as all Sinhala wording in this chatbot, see chat/text.py SINHALA_VERIFIED).
"""

from __future__ import annotations

import re

FINAL_0_19 = ["බිංදුව", "එක", "දෙක", "තුන", "හතර", "පහ", "හය", "හත", "අට", "නවය", "දහය", "එකොළහ", "දොළහ",
              "දහතුන", "දාහතර", "පහළොව", "දාසය", "දාහත", "දහඅට", "දහනවය"]
# combining form of 1-19, before සිය / දහස (index 0 unused)
COMB_1_19 = ["", "එක", "දෙ", "තුන්", "හාර", "පන්", "හය", "හත්", "අට", "නව", "දස", "එකොළොස්", "දොළොස්",
             "දහතුන්", "දාහතර", "පහළොස්", "දාසය", "දාහත්", "දහඅට", "දහනව"]
FINAL_TENS = {2: "විස්ස", 3: "තිහ", 4: "හතළිහ", 5: "පනහ", 6: "හැට", 7: "හැත්තෑව", 8: "අසූව", 9: "අනූව"}
PREFIX_TENS = {2: "විසි", 3: "තිස්", 4: "හතළිස්", 5: "පනස්", 6: "හැට", 7: "හැත්තෑ", 8: "අසූ", 9: "අනූ"}

MONTHS = ["ජනවාරි", "පෙබරවාරි", "මාර්තු", "අප්‍රේල්", "මැයි", "ජූනි", "ජූලි", "අගෝස්තු", "සැප්තැම්බර්",
          "ඔක්තෝබර්", "නොවැම්බර්", "දෙසැම්බර්"]
POINT = "දශම"
HALF = "හමාර"  # "and a half", joined to the final form: තුනහමාර = 3 1/2


def _below_100(n: int, combining: bool) -> str:
    if n < 20:
        return COMB_1_19[n] if combining else FINAL_0_19[n]
    tens, unit = divmod(n, 10)
    if unit == 0:
        return PREFIX_TENS[tens] if combining else FINAL_TENS[tens]
    return PREFIX_TENS[tens] + (COMB_1_19[unit] if combining else FINAL_0_19[unit])


def _below_1000(n: int, combining: bool = False) -> str:
    hundreds, rest = divmod(n, 100)
    if hundreds == 0:
        return _below_100(rest, combining)
    head = COMB_1_19[hundreds] + "සිය"
    if rest == 0:
        return head if combining else head.removeprefix("එක") + "ය"  # 100 = සියය, 200 = දෙසියය
    return f"{head} {_below_100(rest, combining)}"


def number_to_words(n: int) -> str:
    """A whole number 0 .. 9,999,999 in Sinhala words; larger numbers are read digit by digit."""
    if n < 0:
        return "ඍණ " + number_to_words(-n)
    if n >= 10_000_000:
        return digits_to_words(str(n))
    if n < 1000:
        return _below_1000(n)
    lakhs, rest = divmod(n, 100_000)
    if lakhs:
        head = f"ලක්ෂ {FINAL_0_19[lakhs] if lakhs < 20 else _below_100(lakhs, False)}"
        return head if rest == 0 else f"{head}යි {number_to_words(rest)}"
    thousands, rest = divmod(n, 1000)
    if thousands == 1:
        head = "දහස" if rest == 0 else "එක්දහස්"
    else:
        head = _below_1000(thousands, combining=True) + ("දහස" if rest == 0 else "දහස්")
    return head if rest == 0 else f"{head} {_below_1000(rest)}"


def digits_to_words(digits: str) -> str:
    return " ".join(FINAL_0_19[int(d)] for d in digits if d.isdigit())


def decimal_to_words(text: str) -> str:
    """'5.5' -> 'පහ දශම පහ'; '0.42' -> 'බිංදුව දශම හතර දෙක' (digits after the point one by one)."""
    whole, _, frac = text.partition(".")
    out = number_to_words(int(whole or "0"))
    return f"{out} {POINT} {digits_to_words(frac)}" if frac else out


def fraction_to_words(whole: int | None, num: int, den: int) -> str:
    if (num, den) == (1, 2):
        return number_to_words(whole) + HALF if whole else "භාගය"
    if (num, den) == (1, 4):
        return f"{number_to_words(whole)}යි කාල" if whole else "කාල"
    if (num, den) == (3, 4):
        return f"{number_to_words(whole)}යි තුන්කාල" if whole else "තුන්කාල"
    head = f"{number_to_words(whole)}යි " if whole else ""
    return f"{head}{number_to_words(den)}න් {number_to_words(num)}"


def date_to_words(year: int, month: int, day: int) -> str:
    """2026-10-15 -> 'දෙදහස් විසිහය ඔක්තෝබර් පහළොව' (year, month name, day)."""
    return f"{number_to_words(year)} {MONTHS[month - 1]} {number_to_words(day)}"


def time_to_words(hour: int, minute: int) -> str:
    """14:30 -> 'පැය දාහතර මිනිත්තු තිහ'; 14:00 -> 'පැය දාහතර'."""
    out = f"පැය {number_to_words(hour)}"
    return out if minute == 0 else f"{out} මිනිත්තු {number_to_words(minute)}"


_NUM = r"\d+(?:\.\d+)?"
# a number that may carry a fraction: 4, 5.5, 4 1/2, 1/2, 2½
_NUMF = r"\d+(?:\.\d+)?(?:\s+\d+/\d+|[½¼¾])?|\d+/\d+|[½¼¾]"
_VULGAR = {"½": (1, 2), "¼": (1, 4), "¾": (3, 4)}


def any_number_to_words(text: str) -> str:
    """'45', '5.5', '4 1/2', '1/2', '2½' -> words."""
    text = text.strip()
    if text[-1] in _VULGAR:
        whole = text[:-1].strip()
        return fraction_to_words(int(whole) if whole else None, *_VULGAR[text[-1]])
    m = re.fullmatch(r"(?:(\d+)\s+)?(\d+)/(\d+)", text)
    if m and int(m[3]):
        return fraction_to_words(int(m[1]) if m[1] else None, int(m[2]), int(m[3]))
    return decimal_to_words(text)


_PATTERNS: list[tuple[re.Pattern, object]] = [
    # ISO date (optionally with time), then plain times
    (re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{1,2}):(\d{2}))?(?!\d)"),
     lambda m: date_to_words(int(m[1]), int(m[2]), int(m[3]))
     + (f" {time_to_words(int(m[4]), int(m[5]))}" if m[4] else "")
     if 1 <= int(m[2]) <= 12 and 1 <= int(m[3]) <= 31 else m[0]),
    (re.compile(r"(?<![\d.:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])"), lambda m: time_to_words(int(m[1]), int(m[2]))),
    # thousands separators: 1,000 -> 1000
    (re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d,])"), lambda m: m[0].replace(",", "")),
    # "out of 100" in the health score: 100න් -> සියයෙන්
    (re.compile(r"(?<![\d.])100\s?න්"), lambda m: "සියයෙන්"),
    # ranges: 20-25 -> විස්සත් විසිපහත් අතර; 4-4 1/2 -> හතරත් හතරහමාරත් අතර
    (re.compile(rf"(?<![\d.\-/])({_NUMF})\s*[-–]\s*({_NUMF})(?![\d.\-/])"),
     lambda m: f"{any_number_to_words(m[1])}ත් {any_number_to_words(m[2])}ත් අතර"),
    # percentages: 30% -> සියයට තිහ
    (re.compile(rf"({_NUM})\s*%"), lambda m: f"සියයට {decimal_to_words(m[1])}"),
    # >10 -> දහයට වැඩි, <5 -> පහට අඩු
    (re.compile(rf"([<>])\s*({_NUM})"), lambda m: f"{decimal_to_words(m[2])}ට {'වැඩි' if m[1] == '>' else 'අඩු'}"),
    # 3 1/2, 1/2, 2½
    (re.compile(rf"(?<![\d.])(?:{_NUMF})(?![\d/])"), lambda m: any_number_to_words(m[0])),
]


def numbers_to_words(text: str) -> str:
    """Replace every number in a sentence (dates, times, fractions, %, ranges, decimals, integers) with words."""
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text
