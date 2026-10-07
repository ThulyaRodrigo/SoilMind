"""English -> Sinhala machine translation with NLLB-200 (Phase 5, task 3b, "Sinhala"), run locally.

    python -m member4_chatbot.dataset.translate        # translate every English text the dataset needs

Model: facebook/nllb-200-distilled-600M (open weights, CC-BY-NC 4.0, runs on CPU), eng_Latn -> sin_Sinh,
greedy decoding (num_beams=1 is deterministic; beams=4 used here, also deterministic).
Results are cached in data/training/sources/nllb_translations.csv (source, translation, model), so the
dataset build is reproducible without re-running the model, and a Sinhala speaker can correct a row by editing
`translation` and setting `checked` to yes. A checked row is never overwritten.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

from member4_chatbot.paths import TRAINING_SOURCES_DIR

MODEL = "facebook/nllb-200-distilled-600M"
CACHE = TRAINING_SOURCES_DIR / "nllb_translations.csv"
COLUMNS = ["source", "translation", "model", "checked"]
BATCH = 16


VIRAMA, ZWJ = chr(0x0DCA), chr(0x200D)
SPLIT_CONJUNCT = re.compile(r"(\S*" + VIRAMA + r") ([රය]\S*)")


PUNCT = ".,;:!?()\"'"


def sinhala_vocabulary(texts: list[str]) -> tuple[set[str], set[str]]:
    """(words, conjuncts): every Sinhala word in the corpus, and every 'consonant + virama + ZWJ + ර/ය' cluster."""
    words = {w.strip(PUNCT) for t in texts for w in t.split()}
    conj = {m.group(0) for t in texts for m in re.finditer("." + VIRAMA + ZWJ + "[රය]", t)}
    return words, conj


def repair_zwj(text: str, vocab: tuple[set[str], set[str]]) -> str:
    """NLLB's tokenizer has no ZWJ, so 'ප්‍රභේද' comes out as 'ප් රභේද'. Re-join the halves with a ZWJ when the
    left half is a lone consonant or never occurs as a word on its own in our Sinhala corpus (so it is a broken word) and the cluster
    (e.g. ප්‍ර) does occur inside real words. A genuine word boundary ('සහල් රෝග') is left alone."""
    words, conj = vocab

    def join(m: re.Match) -> str:
        left, right = m.group(1), m.group(2)
        bare = left.strip(PUNCT)
        if (len(bare) > 2 and bare in words) or (left[-2:] + ZWJ + right[0]) not in conj:
            return m.group(0)
        return left + ZWJ + right

    prev = None
    while prev != text:
        prev, text = text, SPLIT_CONJUNCT.sub(join, text)
    return text


def load_cache(path: Path = CACHE) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def translate_missing(texts: list[str], path: Path = CACHE) -> dict[str, str]:
    """Translate the texts not in the cache, append them, and return {source: translation} for all `texts`."""
    cache = load_cache(path)
    have = dict(zip(cache.source, cache.translation))
    todo = sorted({t for t in texts if t not in have})
    if todo:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.manual_seed(0)
        tok = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
        target = tok.convert_tokens_to_ids("sin_Sinh")
        t0 = time.time()
        new_rows = []
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=200)
            with torch.inference_mode():
                out = model.generate(**enc, forced_bos_token_id=target, num_beams=4, max_new_tokens=200)
            for src, tr in zip(batch, tok.batch_decode(out, skip_special_tokens=True)):
                new_rows.append({"source": src, "translation": tr.strip(), "model": MODEL, "checked": "no"})
            done = i + len(batch)
            print(f"  NLLB {done}/{len(todo)} ({(time.time() - t0) / done:.2f} s per text)", flush=True)
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.to_csv(path, index=False, encoding="utf-8")
        have = dict(zip(cache.source, cache.translation))
    return {t: have[t] for t in texts}


if __name__ == "__main__":
    from member4_chatbot.dataset.build import texts_to_translate

    todo = texts_to_translate()
    got = translate_missing(todo)
    print(f"{len(got)} texts translated (cache: {CACHE})")
