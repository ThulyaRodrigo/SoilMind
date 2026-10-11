"""Field context: prefer documents that match the plot over look-alike pages.

RRDI publishes near-identical fertilizer pages for irrigated / rainfed fields in the Intermediate and Dry Zone /
the Wet Zone, and tables per variety age class. The retriever cannot always tell them apart, so for "my field"
and fertilizer questions the ranking is nudged towards the plot's profile:
- zone and water regime: from Member 2's advice, whose rules cite the RRDI page they apply
  (`.../rrdi_fertilizerrecomendation_irrigated_izdz/` -> irrigated, Intermediate and Dry Zone)
- variety age class: from Member 2's season (`age_class` "3_5_month" -> 3 1/2 month)
Anything the question states explicitly ("rainfed", "wet zone", "Gampaha", "4 month", වැසි, තෙත් කලාපය, මාස 4)
overrides the plot's value, so a farmer can still ask about another situation.
How strongly a match counts is chosen on the validation split (notebook 05).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from member4_chatbot.dataset.templates import DISTRICTS

PAGE_SLUG = re.compile(r"rrdi_fertilizerrecomendation_(?:[a-z]+_)?(irrigated|rainfed)_(izdz|wz)")
AGE_CLASS = {"2_5_month": "2.5", "3_month": "3", "3_5_month": "3.5", "4_month": "4", "4_5_month": "4.5"}

WATER_WORDS = {
    "irrigated": [r"\birrigat", r"\bunder irrigation\b", "වාරි"],
    "rainfed": [r"\brain[- ]?fed\b", r"\bwithout irrigation\b", r"\bno irrigation\b", "වැසි දියෙන්", "වැස්සෙන්", "වැසි කුඹුර"],
}
ZONE_WORDS = {
    "izdz": [r"\bdry zone\b", r"\bintermediate zone\b", r"\bdry and intermediate\b", "වියළි කලාප", "අතරමැදි කලාප"],
    "wz": [r"\bwet zone\b", "තෙත් කලාප"],
}
AGE_RE = [
    (re.compile(r"\b(\d)\s*(?:1/2|½|\.5)\s*month", re.I), lambda m: f"{m.group(1)}.5"),
    (re.compile(r"මාස\s*(\d)\s*(?:1/2|½)"), lambda m: f"{m.group(1)}.5"),
    (re.compile(r"\b(\d)\s*month", re.I), lambda m: m.group(1)),
    (re.compile(r"මාස\s*(\d)(?!\s*(?:1/2|½))"), lambda m: m.group(1)),
]


@dataclass(frozen=True)
class Profile:
    zone: str | None = None  # "izdz" | "wz"
    water: str | None = None  # "irrigated" | "rainfed"
    age: str | None = None  # "2.5" | "3" | "3.5" | "4" ...

    def merged_with(self, question: "Profile") -> "Profile":
        """The question's explicit values win; the plot fills the rest."""
        return Profile(question.zone or self.zone, question.water or self.water, question.age or self.age)


def plot_profile(recommendations: dict | None, season: dict | None) -> Profile:
    zone = water = age = None
    for rule in (recommendations or {}).get("rules", []):
        m = PAGE_SLUG.search(rule.get("source_url") or "")
        if m:
            water, zone = m.group(1), m.group(2)
            break
    if season and season.get("age_class") in AGE_CLASS:
        age = AGE_CLASS[season["age_class"]]
    return Profile(zone, water, age)


def question_profile(text: str) -> Profile:
    low = text.lower()

    def first(words: dict[str, list[str]]) -> str | None:
        hits = [k for k, pats in words.items() if any(re.search(p, low) for p in pats)]
        return hits[0] if len(hits) == 1 else None  # both mentioned -> no preference

    zone = first(ZONE_WORDS)
    if zone is None:  # a district name tells the zone
        for z, names in DISTRICTS.items():
            if any(n.lower() in low for n in names["en"] + names["si"]):
                zone = z
                break
    age = None
    for rx, val in AGE_RE:
        m = rx.search(text)
        if m:
            age = val(m)
            break
    return Profile(zone, first(WATER_WORDS), age)


def passage_tags(doc: str) -> Profile:
    """Zone / water of a passage, from its RRDI page (fertilizer pages and the crop calendar carry them)."""
    water = "irrigated" if "irrigated" in doc else "rainfed" if "rainfed" in doc else None
    zone = "izdz" if "izdz" in doc else "wz" if "_wz" in doc else None
    return Profile(zone, water, None)


def match_score(profile: Profile, tags: Profile) -> int:
    """+1 for each attribute that matches, -1 for each that conflicts, 0 when either side does not say."""
    s = 0
    for a in ("zone", "water"):
        p, t = getattr(profile, a), getattr(tags, a)
        if p and t:
            s += 1 if p == t else -1
    return s


AGE_GROUP_EN = re.compile(r"^For\s+(\d)(\s*1/2)?\s*Month", re.I)
AGE_GROUP_SI = re.compile(r"^මාස\s*(\d)(\s*1/2)?\s*(?:සඳහා|සදහා)")


def row_age(line: str) -> str | None:
    """Age class of a table group row ("For 3 1/2 Month | ...", "මාස 3 1/2 සඳහා | ...")."""
    m = AGE_GROUP_EN.match(line.strip()) or AGE_GROUP_SI.match(line.strip())
    if not m:
        return None
    return f"{m.group(1)}.5" if m.group(2) else m.group(1)
