"""Template questions generated from structured data in the documents (Phase 5, task 3b-i).

Every number in a generated answer is read from a downloaded document, so each pair is exact and checkable:
- RRDI fertilizer recommendation tables (4 zone / water pages, English and Sinhala versions of each)
- the RRDI "fertilizer by soil test report" table
- RRDI variety sheets (maturity, yield, release year, recommended region, pest / disease resistance)
- the NAICC paddy crop calendar (fertilizer kg/ac, pests by month, seed paddy per acre)

Questions are written in English and, separately, in Sinhala with Sinhala templates (not translations).
Wording varies through several templates per question type, chosen with a fixed seed.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ACRES_PER_HA = 2.471  # 1 ha = 2.471 acres; unit conversion only

ZONE = {
    "izdz": {"en": "Dry and Intermediate Zone", "si": "වියළි හා අතරමැදි කලාපයේ"},
    "wz": {"en": "Wet Zone", "si": "තෙත් කලාපයේ"},
}
WATER = {
    "irrigated": {"en": "irrigated", "en_long": "under irrigation", "si": "වාරි ජලයෙන් වගා කරන"},
    "rainfed": {"en": "rainfed", "en_long": "rainfed (rain-fed)", "si": "වැසි දියෙන් වගා කරන"},
}
# Districts listed on each RRDI page (used only to phrase questions the way farmers ask them).
DISTRICTS = {
    "izdz": {"en": ["Anuradhapura", "Polonnaruwa", "Kurunegala", "Hambantota", "Monaragala", "Trincomalee"],
             "si": ["අනුරාධපුර", "පොළොන්නරුව", "කුරුණෑගල", "හම්බන්තොට", "මොණරාගල", "ත්‍රිකුණාමලය"]},
    "wz": {"en": ["Gampaha", "Colombo", "Kalutara", "Kegalle"],
           "si": ["ගම්පහ", "කොළඹ", "කළුතර", "කෑගල්ල"]},
}
FERT = {
    "urea": {"en": "urea", "si": "යූරියා"},
    "tsp": {"en": "TSP", "si": "ටී.එස්.පී."},
    "mop": {"en": "MOP", "si": "එම්.ඕ.පී."},
    "zinc": {"en": "zinc sulphate", "si": "සින්ක් සල්ෆේට්"},
}
AGE = {
    "3": {"en": "3 month", "si": "මාස 3"},
    "3.5": {"en": "3 1/2 month", "si": "මාස 3 1/2"},
    "4": {"en": "4 month", "si": "මාස 4"},
}
FERT_PAGES = {  # doc id -> (zone, water, lang)
    "rrdi_fert_irrigated_izdz_en": ("izdz", "irrigated", "en"), "rrdi_fert_irrigated_izdz_si": ("izdz", "irrigated", "si"),
    "rrdi_fert_rainfed_izdz_en": ("izdz", "rainfed", "en"), "rrdi_fert_rainfed_izdz_si": ("izdz", "rainfed", "si"),
    "rrdi_fert_irrigated_wz_en": ("wz", "irrigated", "en"), "rrdi_fert_irrigated_wz_si": ("wz", "irrigated", "si"),
    "rrdi_fert_rainfed_wz_en": ("wz", "rainfed", "en"), "rrdi_fert_rainfed_wz_si": ("wz", "rainfed", "si"),
}


@dataclass
class Pair:
    question: str
    answer: str
    lang: str
    intent: str
    doc: str
    anchor: str  # regex (case-insensitive) that identifies the answer passage inside `doc`
    template: str
    also: list[tuple[str, str]] = field(default_factory=list)  # (doc, anchor) of passages that also answer it


def per_acre(kg_ha: float) -> str:
    v = kg_ha / ACRES_PER_HA
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


def table_rows(txt: str) -> list[list[str]]:
    m = re.search(r"\[TABLE\]\n(.*?)\n\[/TABLE\]", txt, re.S)
    if not m:
        raise ValueError("no table")
    return [[c.strip() for c in row.split("|")] for row in m.group(1).split("\n")]


def parse_fert_table(txt: str) -> dict:
    """{age: {time: {fert: kg/ha}}} where time is 'basal', '2', '4', ... (weeks) or 'total'."""
    out: dict = {}
    age = None
    for cells in table_rows(txt):
        first = cells[0]
        m = re.match(r"^(?:For\s+)?(?:මාස\s+)?(\d)(?:\s*1/2)?\s*(?:Month)?\s*(?:සඳහා|සදහා)?$", first.replace("  ", " "))
        if m and len(cells) >= 6:
            age = "3.5" if "1/2" in first else m.group(1)
            cells = cells[1:]
        if age is None or len(cells) < 5:
            continue
        t = cells[0]
        if re.search(r"Basic|මූලික", t):
            key = "basal"
        elif re.search(r"Total|එකතුව", t):
            key = "total"
        else:
            w = re.search(r"(\d+)", t)
            if not w:
                continue
            key = w.group(1)
        vals = {}
        for f, c in zip(("urea", "tsp", "mop", "zinc"), cells[1:5]):
            if c and re.match(r"^\d+(\.\d+)?$", c):
                vals[f] = float(c)
        out.setdefault(age, {})[key] = vals
    return out


def time_phrase(key: str, lang: str) -> str:
    if key == "basal":
        return "the basal application (at establishment)" if lang == "en" else "මූලික පොහොර ලෙස"
    return f"{key} weeks after establishment" if lang == "en" else f"වගාව ස්ථාපනය කර සති {key} දී"


def fert_pairs(raw_dir: Path, rng: random.Random) -> list[Pair]:
    tables = {doc: parse_fert_table((raw_dir / f"{doc}.txt").read_text(encoding="utf-8")) for doc in FERT_PAGES}
    # sanity: the English and Sinhala page of each zone must give the same numbers
    for doc, (zone, water, lang) in FERT_PAGES.items():
        if lang == "en":
            twin = doc[:-3] + "_si"
            if tables[doc] != tables[twin]:
                raise ValueError(f"{doc} and {twin} disagree")
    pairs: list[Pair] = []
    for doc, (zone, water, lang) in FERT_PAGES.items():
        twin = doc[:-3] + ("_si" if lang == "en" else "_en")
        Z, W = ZONE[zone][lang], WATER[water]
        anchor = "Urea" if lang == "en" else "යූරියා"
        twin_anchor = [(twin, "යූරියා" if lang == "en" else "Urea")]
        for age, rows in tables[doc].items():
            A = AGE[age][lang]
            for key, vals in rows.items():
                for f, v in vals.items():
                    F = FERT[f][lang]
                    if key == "total":
                        if lang == "en":
                            q = rng.choice([
                                f"What is the total {F} recommendation for a {A} variety in the {Z} ({W['en']})?",
                                f"How much {F} altogether per hectare for {A} paddy, {W['en']}, {Z}?",
                                f"For the whole season, how much {F} does a {A} rice crop need in the {Z} {W['en_long']}?",
                            ])
                            a = (f"RRDI recommends {v:.0f} kg/ha of {F} in total for {A} varieties in the {Z} "
                                 f"({W['en']}) - about {per_acre(v)} kg per acre.")
                        else:
                            q = rng.choice([
                                f"{Z} {W['si']} කුඹුරු සඳහා {A} වී වලට මුළු {F} ප්‍රමාණය කීයද?",
                                f"{A} ප්‍රභේදයකට {F} හෙක්ටයාරයකට මුළු කිලෝ කීයක් ඕනද? ({Z}, {W['si']})",
                            ])
                            a = (f"{Z} {W['si']} කුඹුරු සඳහා {A} ප්‍රභේදවලට {F} හෙක්ටයාරයකට කිලෝග්‍රෑම් {v:.0f}ක් "
                                 f"(අක්කරයකට කිලෝ {per_acre(v)}ක් පමණ) නිර්දේශිතය.")
                        pairs.append(Pair(q, a, lang, "fertilizer_amount", doc, anchor, "fert_total", twin_anchor))
                        continue
                    T = time_phrase(key, lang)
                    if lang == "en":
                        d = rng.choice(DISTRICTS[zone]["en"])
                        q = rng.choice([
                            f"How much {F} should I apply at {T} for a {A} variety in the {Z} ({W['en']})?",
                            f"{Z}, {W['en']} paddy, {A} variety: how many kg of {F} at {T}?",
                            f"I farm {W['en']} paddy in {d}. How much {F} do I put at {T} for {A} rice?",
                        ])
                        a = (f"For {A} varieties in the {Z} ({W['en']}), RRDI recommends {v:.0f} kg/ha of {F} at {T} "
                             f"- about {per_acre(v)} kg per acre.")
                    else:
                        d = rng.choice(DISTRICTS[zone]["si"])
                        q = rng.choice([
                            f"{Z} {W['si']} කුඹුරේ {A} වී වලට {T} {F} කොච්චර දාන්න ඕනද?",
                            f"{d} {W['si']} කුඹුරට {A} වී සඳහා {T} {F} කිලෝ කීයද?",
                            f"{A} ප්‍රභේදයකට {T} යොදන {F} ප්‍රමාණය කීයද? ({Z}, {W['si']})",
                        ])
                        a = (f"{Z} {W['si']} කුඹුරු සඳහා {A} ප්‍රභේදවලට {T} {F} හෙක්ටයාරයකට කිලෝග්‍රෑම් {v:.0f}ක් "
                             f"(අක්කරයකට කිලෝ {per_acre(v)}ක් පමණ) යෙදිය යුතුය.")
                    pairs.append(Pair(q, a, lang, "fertilizer_amount", doc, anchor, "fert_split", twin_anchor))
            # timing: when is each fertilizer applied for this age?
            for f in ("urea", "mop", "tsp"):
                times = [(k, r[f]) for k, r in rows.items() if k != "total" and f in r]
                if not times:
                    continue
                F = FERT[f][lang]
                if lang == "en":
                    parts = [("as basal" if k == "basal" else f"at {k} weeks") + f" ({v:.0f} kg/ha)" for k, v in times]
                    q = rng.choice([
                        f"When should I apply {F} for a {A} variety in the {Z} ({W['en']})?",
                        f"At what stages is {F} applied for {A} paddy in {rng.choice(DISTRICTS[zone]['en'])} ({W['en']})?",
                        f"What is the {F} schedule for {A} rice, {W['en']}, {Z}?",
                    ])
                    a = f"Apply {F} " + ", ".join(parts) + " - weeks are counted from the date of establishment (RRDI)."
                else:
                    parts = [("මූලික ලෙස" if k == "basal" else f"සති {k} දී") + f" (කිලෝ {v:.0f})" for k, v in times]
                    q = rng.choice([
                        f"{Z} {W['si']} කුඹුරේ {A} වී වලට {F} දාන්නේ කවදද?",
                        f"{A} වී සඳහා {F} යොදන කාලසටහන කුමක්ද? ({Z}, {W['si']})",
                    ])
                    a = f"{F} " + ", ".join(parts) + " යෙදිය යුතුය (සති ගණන් කරන්නේ වගාව ස්ථාපනය කළ දින සිට)."
                pairs.append(Pair(q, a, lang, "fertilizer_timing", doc, anchor, "fert_timing", twin_anchor))
    return pairs


def soil_report_pairs(rng: random.Random) -> list[Pair]:
    """RRDI 'fertilizer as per soil analysis report' (irrigated IZ/DZ): per-acre TSP by soil P, MOP by soil K."""
    p_rows = [("0-5", 22), ("5-10", 12), ("more than 10", 0)]
    k_rows = [("40-80", 8, 10), ("80-160", 4, 4), ("more than 160", 0, 0)]
    p_si = {"0-5": "0-5", "5-10": "5-10", "more than 10": "10ට වැඩි"}
    k_si = {"40-80": "40-80", "80-160": "80-160", "more than 160": "160ට වැඩි"}
    out = []
    en_doc, si_doc = "rrdi_fert_soilreport_irrigated_izdz_en", "rrdi_fert_soilreport_irrigated_izdz_si"
    for rng_txt, tsp in p_rows:
        q = rng.choice([f"Soil test phosphorus is {rng_txt} mg/kg. How much TSP per acre?",
                        f"How much TSP should I use if my soil report shows P {rng_txt} mg/kg?"])
        a = (f"For soil P {rng_txt} mg/kg the RRDI soil-report recommendation is {tsp} kg TSP per acre."
             if tsp else f"For soil P {rng_txt} mg/kg no TSP is needed.")
        out.append(Pair(q, a, "en", "fertilizer_amount", en_doc, "Phosphorus", "soil_report", [(si_doc, "පොස්පරස්")]))
        q = f"පස් වාර්තාවේ පොස්පරස් මි.ග්‍රෑ./කි.ග්‍රෑ. {p_si[rng_txt]} නම් අක්කරයකට ටී.එස්.පී. කොච්චරද?"
        a = (f"පොස්පරස් {p_si[rng_txt]} නම් අක්කරයකට ටී.එස්.පී. කිලෝ {tsp}ක් යෙදිය යුතුය."
             if tsp else f"පොස්පරස් {p_si[rng_txt]} නම් ටී.එස්.පී. අවශ්‍ය නැත.")
        out.append(Pair(q, a, "si", "fertilizer_amount", si_doc, "පොස්පරස්", "soil_report", [(en_doc, "Phosphorus")]))
    for rng_txt, w4, w6 in k_rows:
        q = rng.choice([f"My soil potassium is {rng_txt} mg/kg. How much MOP should I apply?",
                        f"Soil test K {rng_txt} mg/kg - what MOP rate per acre?"])
        a = (f"For soil K {rng_txt} mg/kg apply {w4} kg MOP per acre at 4 weeks and {w6} kg at 6 weeks."
             if w4 else f"For soil K {rng_txt} mg/kg no MOP is needed.")
        out.append(Pair(q, a, "en", "fertilizer_amount", en_doc, "Potassium", "soil_report", [(si_doc, "පොටෑසියම්")]))
        q = f"පසේ පොටෑසියම් මි.ග්‍රෑ./කි.ග්‍රෑ. {k_si[rng_txt]} නම් එම්.ඕ.පී. කීයක් දාන්නද?"
        a = (f"පොටෑසියම් {k_si[rng_txt]} නම් අක්කරයකට සති 4 දී එම්.ඕ.පී. කිලෝ {w4}ක් සහ සති 6 දී කිලෝ {w6}ක් යොදන්න."
             if w4 else f"පොටෑසියම් {k_si[rng_txt]} නම් එම්.ඕ.පී. අවශ්‍ය නැත.")
        out.append(Pair(q, a, "si", "fertilizer_amount", si_doc, "පොටෑසියම්", "soil_report", [(en_doc, "Potassium")]))
    return out


VARIETY_FIELDS_EN = {
    "year": r"Year of (?:release|Lease)\s*\|?\s*:\s*(\d{4})",
    "yield": r"Average yield\s*\|?\s*:\s*([\d.]+)\s*t/ha",
    "days": r"Maturity\s*\|?\s*:\s*([\d\s\-–]+?)\s*[Dd]ays",
    "region": r"(?:Regions recommended for farming|Recommendation)\s*\|?\s*:\s*([^\n|]+?)\s*$",
}
VARIETY_FIELDS_SI = {
    "year": r"(?:හඳුන්වාදුන් වර්ෂය|නිර්දේශීත වර්ෂය)\s*\|?\s*:\s*(\d{4})",
    "yield": r"සාමාන්‍ය අස්වැන්න\s*\|?\s*:\s*හෙක්ටයාරයකට ටොන්\s*([\d.]+)",
    "days": r"අස්වනු නෙලීමට (?:ගතවෙන|යන) කාලය\s*\|?\s*:\s*දින\s*([\d\s\-–]+?\d)",
    "region": r"නිර්දේශය\s*\|?\s*:\s*([^\n|]+?)\s*$",
}
RESIST_EN = {"blast": r"Blast", "bacterial leaf blight": r"Bacterial Leaf Blight", "gall midge": r"Gall Midge",
             "brown planthopper": r"Brown Planthopper"}
RESIST_SI = {"කොළ පාළුව": r"කොළ පාළුව(?: රෝගය)?", "බැක්ටීරියා කොළ අංගමාරය": r"බැක්ටීරියා කොළ අංගමාරයt?",
             "ගොයම් මැස්සා": r"ගොයම් (?:මැක්කා|මැස්සා)", "දුඹුරු පැළ කීඩෑවා": r"දුඹුරු පැළ කීඩෑවා"}
LEVEL_EN = re.compile(r"resistant|susceptible", re.I)
LEVEL_SI = re.compile(r"ප්‍රතිරෝධී|ග්‍රාහී|අවදානම")
FIELD_ANCHOR = {
    "en": {"days": "Maturity", "yield": "Average yield", "year": "Year of", "region": "Recommendation|Regions recommended"},
    "si": {"days": "අස්වනු නෙලීමට", "yield": "සාමාන්‍ය අස්වැන්න", "year": "වර්ෂය", "region": "නිර්දේශය"},
}
RESIST_TWIN = dict(zip(RESIST_EN, RESIST_SI))  # English label -> Sinhala label
NAME_RE = re.compile(r"Variety name\s*\|?\s*:\s*((?:[A-Za-z]{1,3}\s*)?\d+(?:\s*[\-–]\s*\d+[a-z]?)?(?:\s*\(GSR\))?)")


def clean_name(n: str) -> str:
    n = re.sub(r"\s*[\-–]\s*", "-", n.strip())
    return re.sub(r"^([A-Za-z]+)\s*(\d)", r"\1 \2", n)


def resistance(text: str, labels: dict, level_re: re.Pattern) -> dict:
    """{label: level} from 'Label | Level' rows, or a row of labels followed by a row of levels."""
    out = {}
    lines = [[c.strip() for c in ln.split("|")] for ln in text.split("\n")]
    for i, cells in enumerate(lines):
        if len(cells) == 2:
            for key, pat in labels.items():
                if re.fullmatch(pat, cells[0]) and level_re.search(cells[1]):
                    out.setdefault(key, cells[1])
        elif len(cells) >= 3 and i + 1 < len(lines) and len(lines[i + 1]) == len(cells):
            for lab, lev in zip(cells, lines[i + 1]):
                for key, pat in labels.items():
                    if re.fullmatch(pat, lab) and level_re.search(lev):
                        out.setdefault(key, lev)
    return {k: re.sub(r"\s*/\s*", " / ", v) for k, v in out.items()}


def variety_pairs(passages: pd.DataFrame, rng: random.Random) -> list[Pair]:
    out = []
    vdocs = passages[passages.doc.str.match(r"rrdi_rice_(bg|at|bw|ld|h|62)", case=False)]
    docs_lower = {d.lower(): d for d in vdocs.doc.unique()}
    texts = {d: "\n".join(g.text) for d, g in vdocs.groupby("doc")}
    names = {}
    for d, t in texts.items():
        m = NAME_RE.search(t)
        if m and not d.lower().endswith("_si"):
            names[d.lower()] = clean_name(m.group(1))
    for doc, text in texts.items():
        lang = "si" if doc.lower().endswith("_si") else "en"
        base = doc.lower()[:-3] if lang == "si" else doc.lower()
        name = names.get(base)
        if not name:  # Sinhala page without an English twin: its own name, if written in Latin letters
            m = NAME_RE.search(text) or re.search(r"ප්‍රභේදයේ නම\s*\|?\s*:\s*((?:[A-Za-z]{1,3}\s*)?\d+(?:\s*[\-–]\s*\d+[a-z]?)?)", text)
            if not m:
                continue
            name = clean_name(m.group(1))
        twin = docs_lower.get(base + "_si") if lang == "en" else docs_lower.get(base)
        other = "si" if lang == "en" else "en"

        def also_for(field_key: str) -> list[tuple[str, str]]:
            return [(twin, FIELD_ANCHOR[other][field_key])] if twin else []
        fields = VARIETY_FIELDS_EN if lang == "en" else VARIETY_FIELDS_SI
        vals = {}
        for k, pat in fields.items():
            m = re.search(pat, text, re.M)
            vals[k] = m.group(1).strip() if m else None
        A = FIELD_ANCHOR[lang]
        if vals["days"]:
            d = re.sub(r"\s*[\-–]\s*", "-", vals["days"])
            if lang == "en":
                q = rng.choice([f"How many days does {name} take to mature?", f"What is the age of {name}?",
                                f"When can I harvest {name} - how many days?"])
                a = f"{name} matures in about {d} days (RRDI variety sheet)."
            else:
                q = rng.choice([f"{name} අස්වනු නෙලීමට දින කීයක් යනවද?", f"{name} වී වර්ගයේ වයස කීයද?"])
                a = f"{name} අස්වනු නෙලීමට දින {d}ක් පමණ ගතවේ."
            out.append(Pair(q, a, lang, "crop_choice", doc, A["days"], "variety_days", also_for("days")))
        if vals["yield"]:
            if lang == "en":
                q = rng.choice([f"What is the average yield of {name}?", f"How much yield can I expect from {name}?"])
                a = f"The average yield of {name} is {vals['yield']} t/ha."
            else:
                q = rng.choice([f"{name} සාමාන්‍ය අස්වැන්න කීයද?", f"{name} වැව්වොත් අස්වැන්න කොච්චර ලැබෙයිද?"])
                a = f"{name} සාමාන්‍ය අස්වැන්න හෙක්ටයාරයකට ටොන් {vals['yield']}කි."
            out.append(Pair(q, a, lang, "crop_choice", doc, A["yield"], "variety_yield", also_for("yield")))
        if vals["year"]:
            q = f"When was {name} released?" if lang == "en" else f"{name} හඳුන්වා දුන්නේ කවදද?"
            a = f"{name} was released in {vals['year']}." if lang == "en" else f"{name} {vals['year']} දී හඳුන්වා දෙන ලදී."
            out.append(Pair(q, a, lang, "crop_choice", doc, A["year"], "variety_year", also_for("year")))
        if vals["region"]:
            r = vals["region"].strip(" .")
            if lang == "en":
                q = rng.choice([f"Where is {name} recommended for cultivation?",
                                f"Can I grow {name} in my area - where is it recommended?"])
                a = f"{name} is recommended for: {r}."
            else:
                q = f"{name} වගා කිරීමට නිර්දේශ කර ඇත්තේ කොහෙටද?"
                a = f"{name} නිර්දේශය: {r}."
            out.append(Pair(q, a, lang, "crop_choice", doc, A["region"], "variety_region", also_for("region")))
        labels = RESIST_EN if lang == "en" else RESIST_SI
        res = resistance(text, labels, LEVEL_EN if lang == "en" else LEVEL_SI)
        twin_label = RESIST_TWIN if lang == "en" else {v: k for k, v in RESIST_TWIN.items()}
        twin_pats = RESIST_SI if lang == "en" else RESIST_EN
        for short, level in res.items():
            also = [(twin, twin_pats[twin_label[short]])] if twin else []
            if lang == "en":
                q = rng.choice([f"Is {name} resistant to {short}?", f"How does {name} do against {short}?"])
                a = f"RRDI rates {name} as {level.lower()} to {short}."
            else:
                q = f"{name} {short}ට ඔරොත්තු දෙනවද?"
                a = f"{name} {short} සඳහා: {level}."
            out.append(Pair(q, a, lang, "crop_choice", doc, labels[short], "variety_resistance", also))
    return out


CAL_DOC = "naicc_cal_paddy_izdz_irrigated_en"
CAL_FERT = {  # crop calendar (kg/ac), Dry & Intermediate Zone irrigated
    "2 1/2": [("10 days after broadcasting", 20, 0), ("25 days after broadcasting", 30, 10),
              ("35 days after broadcasting", 26, 14), ("40 days after broadcasting", 14, 0)],
    "3": [("2 weeks after broadcasting or planting", 20, 0), ("4 weeks after broadcasting (3 weeks after planting)", 30, 10),
          ("6 weeks after broadcasting (5 weeks after planting)", 26, 14), ("7 weeks after broadcasting (6 weeks after planting)", 14, 0)],
    "3 1/2": [("3 weeks after broadcasting (2 weeks after planting)", 20, 0), ("5 weeks after broadcasting (4 weeks after planting)", 30, 10),
              ("7 weeks after broadcasting (6 weeks after planting)", 26, 14), ("8 weeks after broadcasting (7 weeks after planting)", 14, 0)],
}
CAL_PESTS = {  # month -> pests (crop calendar, Dry & Intermediate Zone irrigated)
    "October": "thrips", "November": "thrips and gall midge",
    "December": "thrips, gall midge, leaf rolling caterpillar, stem borer, white back hopper, sheath and leaf mites, blast and bacterial sheath blight",
    "January": "leaf rolling caterpillar, brown plant hopper, paddy bug, white back hopper, sheath and leaf mites, blast, bacterial sheath blight and brown spot",
    "February": "brown plant hopper and paddy bug", "April": "thrips", "May": "thrips and gall midge",
    "June": "thrips, paddy bug, gall midge, leaf rolling caterpillar, stem borer, brown plant hopper, sheath and leaf mites and bacterial sheath blight",
    "July": "leaf rolling caterpillar, gall midge, brown plant hopper, sheath and leaf mites and brown spot",
    "August": "brown plant hopper and paddy bug",
}
MONTH_SI = {"October": "ඔක්තෝබර්", "November": "නොවැම්බර්", "December": "දෙසැම්බර්", "January": "ජනවාරි",
            "February": "පෙබරවාරි", "April": "අප්‍රේල්", "May": "මැයි", "June": "ජූනි", "July": "ජූලි", "August": "අගෝස්තු"}


def calendar_pairs(rng: random.Random) -> list[Pair]:
    """The crop calendar was transcribed into data/docs/manual_text (see its README); values copied from there."""
    out = []
    for age, rows in CAL_FERT.items():
        for when, urea, mop in rows:
            q = rng.choice([f"Crop calendar: how much urea per acre {when} for a {age} month variety?",
                            f"How many kg of urea per acre {when} for {age} month paddy in the irrigated Dry Zone?"])
            a = f"The NAICC crop calendar gives {urea} kg/ac urea {when} for {age} month varieties."
            if mop:
                a += f" MOP {mop} kg/ac is applied at the same time."
            out.append(Pair(q, a, "en", "fertilizer_amount", CAL_DOC, f"For {age} month variety", "calendar_fert"))
            q = f"මාස {age} වී වලට {when} අක්කරයකට යූරියා කීයද? (වියළි කලාපය, වාරි)"
            a = f"බෝග දින දර්ශනය අනුව මාස {age} ප්‍රභේදවලට අක්කරයකට යූරියා කිලෝ {urea}ක් ({when})."
            out.append(Pair(q, a, "si", "fertilizer_amount", CAL_DOC, f"For {age} month variety", "calendar_fert"))
    for month, pests in CAL_PESTS.items():
        q = rng.choice([f"Which pests and diseases should I watch for in {month}?",
                        f"What pest problems come in {month} in irrigated Dry Zone paddy?"])
        a = f"For {month} the NAICC crop calendar (Dry and Intermediate Zone, irrigated) lists {pests}."
        out.append(Pair(q, a, "en", "pest_disease", CAL_DOC, "Pests and diseases by month", "calendar_pests"))
        q = rng.choice([f"{MONTH_SI[month]} මාසයේ වී වගාවට එන පළිබෝධ සහ රෝග මොනවද?",
                        f"{MONTH_SI[month]} මාසයේදී කුඹුරේ බලාගත යුතු පළිබෝධ මොනවද?"])
        a = f"බෝග දින දර්ශනය (වියළි හා අතරමැදි කලාපය, වාරි) අනුව {MONTH_SI[month]} මාසයට: {pests}."
        out.append(Pair(q, a, "si", "pest_disease", CAL_DOC, "Pests and diseases by month", "calendar_pests"))
    seed = [("broadcasting nadu (medium elongate and elongate)", 40, "නාඩු වපුරන්න"), ("broadcasting samba", 30, "සම්බා වපුරන්න"),
            ("transplanting nadu", 20, "නාඩු පැල සිටුවන්න"), ("transplanting samba", 15, "සම්බා පැල සිටුවන්න"),
            ("parachute seedling broadcasting", 12, "පැරෂුට් ක්‍රමයට")]
    for what, kg, si in seed:
        out.append(Pair(f"How much seed paddy per acre for {what}?",
                        f"The crop calendar gives {kg} kg of seed paddy per acre for {what}.",
                        "en", "crop_establishment", CAL_DOC, "Seed paddy requirement", "calendar_seed"))
        out.append(Pair(f"{si} අක්කරයකට බිත්තර වී කොච්චර ඕනද?",
                        f"බෝග දින දර්ශනය අනුව අක්කරයකට බිත්තර වී කිලෝ {kg}ක්.",
                        "si", "crop_establishment", CAL_DOC, "Seed paddy requirement", "calendar_seed"))
    return out


def subsample(pairs: list[Pair], caps: dict[tuple[str, str], int], rng: random.Random) -> list[Pair]:
    """Keep at most caps[(template, lang)] pairs of each kind, chosen with the seeded RNG."""
    by: dict = {}
    for p in pairs:
        by.setdefault((p.template, p.lang), []).append(p)
    out = []
    for key, group in sorted(by.items()):
        cap = caps.get(key)
        if cap is not None and len(group) > cap:
            group = rng.sample(group, cap)
        out.extend(group)
    return out


TEMPLATE_CAPS = {
    ("fert_split", "en"): 80, ("fert_split", "si"): 80,
    ("fert_total", "en"): 30, ("fert_total", "si"): 30,
    ("fert_timing", "en"): 30, ("fert_timing", "si"): 30,
    ("variety_resistance", "en"): 50, ("variety_resistance", "si"): 50,
}


def all_template_pairs(raw_dir: Path, passages: pd.DataFrame, seed: int) -> list[Pair]:
    rng = random.Random(seed)
    pairs = fert_pairs(raw_dir, rng) + soil_report_pairs(rng) + variety_pairs(passages, rng) + calendar_pairs(rng)
    return subsample(pairs, TEMPLATE_CAPS, rng)
