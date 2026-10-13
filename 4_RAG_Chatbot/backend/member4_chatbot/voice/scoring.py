"""WER / CER for Sinhala transcripts, with the same normalisation applied to reference and hypothesis.

Before scoring, both texts are put in one canonical form, so that only real recognition errors count:
- Unicode NFC; zero-width joiners removed (ප්‍ර and ප්ර are the same word to a reader)
- digits written as Sinhala words (a model may write "45" or "හතළිස්පහ"; both are right)
- dotted abbreviations joined (ටී.එස්.පී. -> ටීඑස්පී) and Latin codes written in Sinhala letters (Bg -> බීජී,
  TSP -> ටීඑස්පී), because a speaker says the letters
- punctuation removed, spaces collapsed

WER counts words (split at spaces). Sinhala spacing varies (හතළිස් පහ / හතළිස්පහ, වී වලට / වීවලට), so CER is
computed with spaces removed and is the more reliable of the two for Sinhala; both are reported.
"""

from __future__ import annotations

import re
import unicodedata

import jiwer

from member4_chatbot.voice.sinhala_numbers import numbers_to_words

LETTERS = {"A": "ඒ", "B": "බී", "C": "සී", "D": "ඩී", "E": "ඊ", "F": "එෆ්", "G": "ජී", "H": "එච්", "I": "අයි",
           "J": "ජේ", "K": "කේ", "L": "එල්", "M": "එම්", "N": "එන්", "O": "ඕ", "P": "පී", "Q": "කියු", "R": "ආර්",
           "S": "එස්", "T": "ටී", "U": "යූ", "V": "වී", "W": "ඩබ්ලිව්", "X": "එක්ස්", "Y": "වයි", "Z": "සෙඩ්"}
_INVISIBLE = dict.fromkeys(map(ord, "\u200d\u200c\u200b\ufeff"), None)


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    # dotted Sinhala abbreviations: ටී.එස්.පී. -> ටීඑස්පී (before ZWJ removal, which does not touch dots)
    t = re.sub(r"(?<=[\u0D80-\u0DFF\u200d])\.(?=[\u0D80-\u0DFF])", "", t)
    # Latin codes / acronyms of up to 4 letters (Bg, TSP, T.S.P, pH) -> Sinhala letter names
    t = re.sub(r"\b(?:[A-Za-z]\.?){1,4}\b(?!\w)",
               lambda m: "".join(LETTERS[c.upper()] for c in m[0] if c.isalpha()), t)
    t = numbers_to_words(t)
    t = t.translate(_INVISIBLE)
    t = re.sub(r"[^\w\u0D80-\u0DFF\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def wer(refs: list[str], hyps: list[str]) -> float:
    r, h = [normalize(x) for x in refs], [normalize(x) for x in hyps]
    keep = [i for i, x in enumerate(r) if x]
    return float(jiwer.wer([r[i] for i in keep], [h[i] or "∅" for i in keep]))


def cer(refs: list[str], hyps: list[str]) -> float:
    r = [normalize(x).replace(" ", "") for x in refs]
    h = [normalize(x).replace(" ", "") for x in hyps]
    keep = [i for i, x in enumerate(r) if x]
    return float(jiwer.cer([r[i] for i in keep], [h[i] or "∅" for i in keep]))


def word_substitutions(ref: str, hyp: str) -> list[tuple[str, str]]:
    """Aligned (reference word, hypothesis word) pairs that differ, for building the term dictionary."""
    r, h = normalize_keep_spelling(ref), normalize_keep_spelling(hyp)
    if not r or not h:
        return []
    out = jiwer.process_words(r, h)
    pairs = []
    for chunk in out.alignments[0]:
        if chunk.type == "substitute":
            rw = r.split()[chunk.ref_start_idx:chunk.ref_end_idx]
            hw = h.split()[chunk.hyp_start_idx:chunk.hyp_end_idx]
            pairs += list(zip(rw, hw))
    return pairs


def normalize_keep_spelling(text: str) -> str:
    """Like normalize() but keeps ZWJ and digits, so dictionary entries match what the model really writes."""
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"[^\w\u0D80-\u0DFF\u200d\s.]", " ", t)
    t = re.sub(r"(?<![\u0D80-\u0DFF\u200d\w])\.|\.(?![\u0D80-\u0DFF\w])", " ", t)
    return re.sub(r"\s+", " ", t).strip()
