"""Extractive answers: pick the 1-3 sentences (or table rows) of a passage that best answer the question.

Nothing is generated. Every word of the answer is copied from a DOA / RRDI passage, so numbers can never be
invented. Sentences are scored with the same fine-tuned retriever model (cosine similarity between the question
and each sentence). The best sentence is kept, plus up to `max_sentences - 1` more whose score is within
`margin` of the best. They are shown in their original order.

A table row on its own ("4 Weeks | 75 | | 25 |") is meaningless, so a selected row is shown with the table's
header row and the nearest group row above it ("For 3 Month | Basic | ..."), as in the document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

SENTENCE_END = re.compile(r"(?<=[.!?෴])\s+")
MIN_WORDS = 3


@dataclass
class Unit:
    text: str
    index: int  # position in the passage
    kind: str  # "sentence" | "row"
    cells: int = 0


@dataclass
class Extract:
    passage_id: str
    lines: list[str]  # the answer, in passage order (header rows included for tables)
    unit_scores: list[float] = field(default_factory=list)  # scores of the selected units
    best_score: float = 0.0


def split_units(text: str, title: str = "") -> list[Unit]:
    """Sentences and table rows of a passage. The passage's title line is context, not an answer."""
    units: list[Unit] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or (title and line == title):
            continue
        if "|" in line:
            cells = [c.strip() for c in line.split("|")]
            units.append(Unit(line, len(units), "row", len(cells)))
            continue
        for sent in SENTENCE_END.split(line):
            sent = sent.strip()
            if len(sent.split()) >= MIN_WORDS:
                units.append(Unit(sent, len(units), "sentence"))
    return units


MAX_TABLE_ROWS = 30
LIST_LEAD = re.compile(r"(following|as follows|පහත|:\s*$)", re.I)
MAX_LIST_ITEMS = 6


def _table_context(units: list[Unit], chosen: set[int]) -> set[int]:
    """A chosen table row brings its whole table (up to MAX_TABLE_ROWS rows): a single row such as
    "4 Weeks | 75 | | 25 |" means nothing without the header and age-group rows, and short rows are hard to
    rank, so the farmer is shown the table as printed in the document."""
    extra = set()
    for i in chosen:
        if units[i].kind != "row":
            continue
        j = i
        while j - 1 >= 0 and units[j - 1].kind == "row":
            j -= 1
        k = i
        while k + 1 < len(units) and units[k + 1].kind == "row":
            k += 1
        if k - j + 1 <= MAX_TABLE_ROWS:
            extra.update(range(j, k + 1))
        else:  # very long table: header + the chosen row's group row
            extra.add(j)
            for g in range(i - 1, j, -1):
                if units[g].cells >= units[j].cells:
                    extra.add(g)
                    break
    return extra


def _list_items(units: list[Unit], chosen: set[int]) -> set[int]:
    """A sentence that introduces a list ("the following fungicides could be applied", "... පහත ...") brings the
    short lines that follow it, up to MAX_LIST_ITEMS."""
    extra = set()
    for i in chosen:
        if units[i].kind == "sentence" and LIST_LEAD.search(units[i].text):
            for j in range(i + 1, min(i + 1 + MAX_LIST_ITEMS, len(units))):
                if len(units[j].text.split()) > 25:
                    break
                extra.add(j)
    return extra


def select(question_emb: np.ndarray, units: list[Unit], unit_embs: np.ndarray, *, max_sentences: int,
           margin: float) -> tuple[list[int], list[float]]:
    """Indices of the chosen units (best first) and their scores."""
    if not units:
        return [], []
    scores = unit_embs @ question_emb
    order = np.argsort(-scores)
    best = float(scores[order[0]])
    chosen = [int(order[0])]
    for i in order[1:]:
        if len(chosen) >= max_sentences or best - float(scores[i]) > margin:
            break
        chosen.append(int(i))
    return chosen, [float(scores[i]) for i in chosen]


def extract(passage_id: str, text: str, title: str, question_emb: np.ndarray, encode, *, max_sentences: int,
            margin: float) -> Extract:
    """`encode(list[str]) -> np.ndarray` returns normalised embeddings (the retriever model)."""
    units = split_units(text, title)
    if not units:
        return Extract(passage_id, [text.strip()], [], 0.0)
    embs = encode([u.text for u in units])
    chosen, scores = select(question_emb, units, embs, max_sentences=max_sentences, margin=margin)
    shown = sorted(set(chosen) | _table_context(units, set(chosen)) | _list_items(units, set(chosen)))
    return Extract(passage_id, [units[i].text for i in shown], scores, scores[0] if scores else 0.0)


NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER.findall(text)}


def _is_rounding(n: str, source: set[str]) -> bool:
    """True if `n` is a source number rounded to n's decimals (4.549 -> 4.55, 20.23 -> 20)."""
    decimals = len(n.split(".")[1]) if "." in n else 0
    v = float(n)
    half = 0.5 * 10 ** -decimals
    return any(abs(float(s) - v) <= half + 1e-9 for s in source)


def unsupported_numbers(answer: str, *sources: str, allowed_extra: set[str] | None = None) -> set[str]:
    """Numbers in `answer` that come from none of `sources` (passages, live values), neither as written nor as a
    rounding of a source number. Should always be empty."""
    allowed = set().union(*(numbers(s) for s in sources)) if sources else set()
    allowed |= allowed_extra or set()
    return {n for n in numbers(answer)
            if n not in allowed and n.rstrip("0").rstrip(".") not in allowed and not _is_rounding(n, allowed)}
