"""Cleaning: Unicode normalisation and removal of headers, footers and page furniture.

- **NFC** normalisation, so the same Sinhala letter always has the same code points.
- **ZWJ (U+200D) is kept**: Sinhala needs it for yansaya / rakaransaya and touching letters (ශ්\u200dරී, ප්\u200dරමාණය).
  Other invisible characters (ZWSP, BOM, soft hyphen, ZWNJ) are removed.
- **Headers and footers**:
  - in PDFs, a line that repeats on at least half of the pages (digits ignored) is a running header / footer
  - in web pages, a line that appears on at least a quarter of all saved pages is site furniture (address,
    phone, language switch, "Read More >>")
  - lines that are only a page number, and lines in Tamil script (language-switch links) are dropped
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

TABLE_MARKERS = {"[TABLE]", "[/TABLE]"}  # structure, never furniture
ZWJ = "\u200d"
DROP_CHARS = dict.fromkeys(map(ord, "\u200b\u200c\ufeff\u00ad\u2060"), None)
TAMIL = re.compile(r"[஀-௿]")
PAGE_NUMBER = re.compile(r"^\s*(page\s*)?[\divxlcIVXLC]{1,6}\s*$", re.I)
FURNITURE = re.compile(
    r"^(read more\s*>>|වැඩිදුර කියවන්න\s*>>|facebook|e|සි|english|සිංහල|\*|proceed\s*>>|premium banner|ඉදිරියට යන්න\s*>>"
    r"|.*\S+@doa\.gov\.lk.*|(telephone|fax|ෆැක්ස්|දුරකථන අංකය|දුරකථන|address|ලිපිනය)\s*:.*)$", re.I)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(DROP_CHARS)
    text = text.replace("\u00a0", " ").replace("\r", "")
    return re.sub(r"[ \t]+", " ", text)


def line_key(line: str, ignore_digits: bool = False) -> str:
    key = line.strip().lower()
    return re.sub(r"\d+", "#", key) if ignore_digits else key


def repeated_lines(units: list[list[str]], min_share: float, min_count: int = 2, ignore_digits: bool = False) -> set[str]:
    """Line keys that occur in at least `min_share` of the units (pages of one PDF, or web pages)."""
    counts: Counter = Counter()
    for unit in units:
        counts.update({line_key(ln, ignore_digits) for ln in unit if ln.strip()})
    n = len(units)
    if n < 2:
        return set()
    return {k for k, c in counts.items() if c >= max(min_count, min_share * n)}


def clean_lines(text: str, drop: set[str], edge_drop: set[str] | None = None, edge: int = 0) -> str:
    """Drop furniture lines. `edge_drop` keys (digits ignored) are dropped only among the first / last `edge`
    non-empty lines, where running headers and footers sit."""
    lines = [ln.strip() for ln in normalise(text).split("\n")]
    filled = [i for i, ln in enumerate(lines) if ln]
    edges = set(filled[:edge] + filled[-edge:]) if edge else set()
    out = []
    for i, s in enumerate(lines):
        if s and (line_key(s) in drop or PAGE_NUMBER.match(s) or TAMIL.search(s) or FURNITURE.match(s)
                  or (i in edges and edge_drop and line_key(s, True) in edge_drop)):
            continue
        out.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


HEADER_LINES = 2  # running headers / footers are looked for in the first and last 2 lines of a page


def clean_pdf_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Clean every page of one PDF. A running header / footer is a short line among the first or last two lines
    of a page that repeats (digits ignored, e.g. "Page 3") on at least half of the pages; body text that merely
    looks alike ("Average yield : 5.0 t/ha" on every page) is never removed."""
    norm = [(p, normalise(t)) for p, t in pages]
    edges = []
    for _, t in norm:
        filled = [ln for ln in t.split("\n") if ln.strip()]
        edges.append([ln for ln in filled[:HEADER_LINES] + filled[-HEADER_LINES:] if len(ln.split()) <= 12])
    drop = repeated_lines(edges, 0.5, min_count=3, ignore_digits=True) if len(pages) >= 4 else set()
    return [(p, clean_lines(t, set(), drop, HEADER_LINES)) for p, t in norm]


def site_furniture(html_texts: list[str]) -> set[str]:
    """Line keys repeated across many saved web pages (header / footer of doa.gov.lk)."""
    units = [[ln for ln in normalise(t).split("\n") if ln.strip() not in TABLE_MARKERS] for t in html_texts]
    return repeated_lines(units, 0.25, min_count=3)
