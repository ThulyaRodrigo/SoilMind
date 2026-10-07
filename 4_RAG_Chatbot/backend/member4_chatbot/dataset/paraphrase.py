"""Rule-based English paraphrases (Phase 5, task 3b-iii).

No model is used: each rule is a word / phrase substitution or a farmer-style framing, applied with a seeded
random generator. A paraphrase is kept only if it differs from the original. Meaning-changing substitutions are
avoided (numbers, variety names, nutrients and units are never touched).
"""

from __future__ import annotations

import random
import re

# (pattern, replacements); applied case-insensitively on word boundaries, first match only
SWAPS: list[tuple[str, list[str]]] = [
    (r"\bhow much\b", ["what amount of", "how many kg of"]),
    (r"\bhow do i\b", ["how can i", "what is the way to", "how should i"]),
    (r"\bwhat should i do\b", ["what can i do", "what do you advise", "what is the remedy"]),
    (r"\bwhen should i\b", ["at what time should i", "when do i", "when is the right time to"]),
    (r"^should i\b", ["do i need to", "is it good to"]),
    (r"\bapply\b", ["put", "use", "add"]),
    (r"\bpaddy\b", ["rice"]),
    (r"\brice\b", ["paddy"]),
    (r"\bmy field\b", ["my paddy field", "my plot", "my land"]),
    (r"\bfertilizer\b", ["fertiliser", "pohora"]),
    (r"\bvariety\b", ["type of rice", "rice variety"]),
    (r"\bvarieties\b", ["rice types", "kinds of rice"]),
    (r"\bwhat are\b", ["tell me", "list"]),
    (r"\bwhat is\b", ["tell me", "explain"]),
    (r"\bwhich\b", ["what"]),
    (r"\bcontrol\b", ["manage", "get rid of"]),
    (r"\bsymptoms\b", ["signs"]),
    (r"\bper acre\b", ["for one acre"]),
    (r"\bper hectare\b", ["for one hectare"]),
]
PREFIXES = ["Please tell me, ", "Can you tell me ", "I want to know ", "Sir, ", "Question: ", "Hello, ", "Excuse me, "]
SUFFIXES = [" Please help.", " Thanks.", " for my paddy?", ""]


def _apply_swap(q: str, rng: random.Random) -> str:
    options = [(pat, reps) for pat, reps in SWAPS if re.search(pat, q, re.I)]
    if not options:
        return q
    pat, reps = rng.choice(options)
    rep = rng.choice(reps)

    def keep_case(m: re.Match) -> str:
        return rep[0].upper() + rep[1:] if m.group(0)[0].isupper() else rep

    return re.sub(pat, keep_case, q, count=1, flags=re.I)


KEEP_CASE = re.compile(r"\b(?:[A-Z][a-z]{0,2}(?=\s*\d)|[A-Z]{2,})")  # variety codes (Bg 300) and acronyms (TSP)


def lower_keep_names(text: str) -> str:
    keep = {m.start(): m.group(0) for m in KEEP_CASE.finditer(text)}
    out, i = [], 0
    while i < len(text):
        if i in keep:
            out.append(keep[i]); i += len(keep[i])
        else:
            out.append(text[i].lower()); i += 1
    return "".join(out)


def paraphrase(q: str, rng: random.Random) -> str | None:
    """One rule-based paraphrase of `q`, or None if the rules produce nothing new."""
    out = q
    for _ in range(rng.choice([1, 2])):
        out = _apply_swap(out, rng)
    style = rng.random()
    if style < 0.3:
        p = rng.choice(PREFIXES)
        out = p + out[0].lower() + out[1:] if not out[:2].isupper() else p + out
    elif style < 0.5:
        out = lower_keep_names(out.rstrip("?."))  # farmer typing: no capitals, no question mark
    elif style < 0.6:
        s = rng.choice(SUFFIXES)
        out = out.rstrip("?.") + ("?" + s if s and not s.startswith(" for") else s + ("" if s else "?"))
    out = re.sub(r"\s+", " ", out).strip()
    return out if out.lower().rstrip("?. ") != q.lower().rstrip("?. ") else None
