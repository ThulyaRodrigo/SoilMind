"""Sentence reranker: our fine-tuned cross-encoder picks the sentences / table rows that answer the question.

The retriever finds passages; the reranker reads (question, sentence) pairs together and scores how well each
sentence answers the question. Trained in notebook 05 on the auto-generated question-passage pairs (train split
only): a sentence is labelled relevant if it contains a key fact of the expected answer (a number or variety code
that is in both the expected answer and the expected passage). Sentences of the other top-ranked passages are hard
negatives.

This module is shared by notebook 05 (labels, inputs, selection) and the chat pipeline, so both do exactly the
same thing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from member4_chatbot.chat.extractive import LIST_LEAD, MAX_LIST_ITEMS, Unit, split_units
from member4_chatbot.chat.profile import row_age

FACT = re.compile(r"(?:Bg|At|Bw|Ld|H)\s?\d+(?:[-/]\d+)?|\d+(?:\.\d+)?")


def key_facts(answer: str, passage_texts: list[str]) -> set[str]:
    """Numbers and variety codes of an expected answer that also appear in its expected passage(s)."""
    ptxt = " ".join(passage_texts)
    return {f for f in FACT.findall(answer) if f in ptxt}


def contains_fact(text: str, fact: str) -> bool:
    if re.fullmatch(r"\d+(?:\.\d+)?", fact):
        return re.search(rf"(?<![\d.]){re.escape(fact)}(?![\d.])", text) is not None
    return fact in text


def table_context(units: list[Unit], i: int) -> list[int]:
    """Indices of the header row and (for a short row) the group row that give table row `i` its meaning."""
    if units[i].kind != "row":
        return []
    j = i
    while j - 1 >= 0 and units[j - 1].kind == "row":
        j -= 1
    out = [j] if j != i else []
    if units[i].cells < units[j].cells:
        for g in range(i - 1, j, -1):
            if units[g].cells >= units[j].cells:
                out.append(g)
                break
    return out


def group_age(units: list[Unit], i: int) -> str | None:
    """Age class of the table group a row belongs to (the row itself or the nearest group row above)."""
    if units[i].kind != "row":
        return None
    for g in range(i, -1, -1):
        if units[g].kind != "row":
            return None
        age = row_age(units[g].text.split("|")[0] + "|")
        if age:
            return age
    return None


def unit_inputs(units: list[Unit], title: str) -> list[str]:
    """What the cross-encoder reads for each unit: the passage title, and for table rows the header and group row."""
    out = []
    for i, u in enumerate(units):
        ctx = [units[c].text for c in table_context(units, i)]
        body = " ; ".join(ctx + [u.text]) if ctx else u.text
        out.append(f"{title} :: {body}" if title else body)
    return out


@dataclass
class Choice:
    passage_id: str
    lines: list[str]  # shown answer, in passage order (with table header / group rows)
    selected: list[str]  # the chosen units themselves (used for highlighting key numbers)
    best_score: float
    scores: dict = field(default_factory=dict)


def choose(question: str, passages: list[dict], scorer, *, answer_lang: str, max_units: int, margin: float,
           lang_delta: float, age: str | None = None, age_bonus: float = 0.0) -> Choice | None:
    """`passages`: dicts with id, text, title, lang (ranked best first). `scorer(question, texts) -> list[float]`.

    The passage holding the best-scoring unit is used (a passage in the answer language if its best unit is within
    `lang_delta`); within it, units scoring within `margin` of its best are kept, up to `max_units`. Table rows of
    the plot's age class get +age_bonus, rows of another age class -age_bonus.
    """
    per = []
    for p in passages:
        units = split_units(p["text"], p["title"])
        if units:
            per.append((p, units, unit_inputs(units, p["title"])))
    if not per:
        return None
    flat = [t for _, _, inputs in per for t in inputs]
    raw = list(scorer(question, flat))
    k = 0
    scored = []
    for p, units, inputs in per:
        s = raw[k:k + len(units)]
        k += len(units)
        if age and age_bonus:
            s = [v + (age_bonus if a == age else -age_bonus if a else 0.0)
                 for v, a in ((v, group_age(units, i)) for i, v in enumerate(s))]
        scored.append((p, units, s))
    best_p = max(scored, key=lambda x: max(x[2]))
    if best_p[0]["lang"] != answer_lang:
        same = [x for x in scored if x[0]["lang"] == answer_lang]
        if same:
            alt = max(same, key=lambda x: max(x[2]))
            if max(alt[2]) >= max(best_p[2]) - lang_delta:
                best_p = alt
    p, units, s = best_p
    order = sorted(range(len(units)), key=lambda i: -s[i])
    top = s[order[0]]
    chosen = [i for i in order[:max_units] if s[i] >= top - margin]
    shown = set(chosen)
    for i in chosen:
        shown.update(table_context(units, i))
        if units[i].kind == "sentence" and LIST_LEAD.search(units[i].text):
            for j in range(i + 1, min(i + 1 + MAX_LIST_ITEMS, len(units))):
                if len(units[j].text.split()) > 25:
                    break
                shown.add(j)
    return Choice(p["id"], [units[i].text for i in sorted(shown)], [units[i].text for i in sorted(chosen)], top,
                  {units[i].text: round(s[i], 4) for i in chosen})


class Reranker:
    """The fine-tuned cross-encoder from artifacts/reranker/model (loaded once)."""

    def __init__(self, path, max_length: int = 256):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(str(path), device="cpu", max_length=max_length)

    def __call__(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(x) for x in self.model.predict([(question, t) for t in texts], batch_size=64,
                                                     show_progress_bar=False)]
