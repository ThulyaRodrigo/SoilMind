"""Unicode check, cleaning and chunking (Phase 5, task 7)."""

import unicodedata

from member4_chatbot.ingest.chunk import MAX_WORDS, chunk_pages
from member4_chatbot.ingest.clean import clean_lines, clean_pdf_pages, normalise, site_furniture
from member4_chatbot.ingest.sinhala import (
    ZWJ,
    assess,
    common_word_rate,
    fm_to_unicode,
    invalid_sign_rate,
    sinhala_share,
)

READABLE_SI = (
    "මෙම නිර්දේශය යොදා ගත හැකි දිස්ත්‍රික්ක සහ ප්‍රදේශ. වී වගාව සඳහා යූරියා පොහොර යෙදිය යුතුය. "
    "කුඹුරු සඳහා ජලය අවශ්‍ය වේ. පස සකස් කිරීම සහ වල් පාලනය කළ යුතු අතර, රෝග හා පළිබෝධ ගැන සැලකිලිමත් විය යුතුය."
)
# A scanner's Latin OCR layer of a Sinhala leaflet (real sample from rrdi_leaf_awd_si.pdf)
SCANNER_GARBAGE = "tslbOrD I OcolOd eoigtol O@elo 666rloJ@to esrOeo qtl6 8@co gor€, 66€rd co8. E@o dOtocOo ce@eod"


def test_sinhala_share():
    assert sinhala_share("වී වගාව") == 1.0
    assert sinhala_share("paddy field") == 0.0
    assert 0.4 < sinhala_share("යූරියා urea") < 0.7
    assert sinhala_share("123 !!") == 0.0


def test_readable_sinhala_passes():
    a = assess(READABLE_SI, "si")
    assert a["ok"], a
    assert common_word_rate(READABLE_SI) > 0.1
    assert invalid_sign_rate(READABLE_SI) < 0.02


def test_scanner_garbage_fails_and_mapping_is_rejected():
    assert not assess(SCANNER_GARBAGE * 3, "si")["ok"]
    mapped = fm_to_unicode(SCANNER_GARBAGE * 3)
    assert sinhala_share(mapped) > 0.5  # the map does produce Sinhala letters...
    assert not assess(mapped, "si")["ok"]  # ...but the readability check rejects them
    assert invalid_sign_rate(mapped) > 0.1


def test_fm_map_moves_prefix_vowel():
    # FM fonts type the kombuva (ෙ) before the consonant; Unicode stores it after
    assert fm_to_unicode("fl") == "කෙ"
    assert fm_to_unicode("fld") == "කො"
    assert fm_to_unicode(",xld") == "ලංකා"


def test_no_text_layer():
    assert assess("", "si")["reason"] == "no text layer"


def test_english_text():
    assert assess("The urea fertilizer should be applied to the paddy field after two weeks. " * 3, "en")["ok"]


def test_normalise_keeps_zwj_and_nfc():
    word = "ප්" + ZWJ + "රමාණය"
    out = normalise(word + "​﻿")
    assert ZWJ in out
    assert "​" not in out and "﻿" not in out
    assert out == unicodedata.normalize("NFC", out)


def test_pdf_running_header_removed():
    body = "Variety name : Bg {i}\nAverage yield : {i}.0 t/ha\nMaturity : 9{i} days\nRecommendation : General"
    pages = [(i, f"RRDI Bathalagoda\n{body.format(i=i)}\nPage {i}") for i in range(1, 6)]
    cleaned = clean_pdf_pages(pages)
    for i, text in cleaned:
        assert "RRDI Bathalagoda" not in text and f"Page {i}" not in text
        # body lines that differ only in numbers are content, not headers
        assert f"Average yield : {i}.0 t/ha" in text and f"Maturity : 9{i} days" in text


def test_site_furniture_found_and_tables_kept():
    pages = [f"Address : Bathalagoda\n[TABLE]\nA | B\n[/TABLE]\nPage {i} text" for i in range(8)]
    furniture = site_furniture(pages)
    cleaned = clean_lines(pages[0], furniture)
    assert "Bathalagoda" not in cleaned
    assert "[TABLE]" in cleaned and "Page 0 text" in cleaned


def _words(n, w="word"):
    return " ".join([w] * n)


def test_chunks_respect_max_and_keep_table_whole():
    table_rows = "\n".join(f"Row {i} | {i} | {i * 2} | {i * 3}" for i in range(80))  # > 300 words
    text = f"## Heading one\n{_words(120)}.\n\n{_words(120)}.\n\n[TABLE]\n{table_rows}\n[/TABLE]\n\n## Heading two\n{_words(200)}."
    chunks = chunk_pages([(1, text)])
    table_chunks = [c for c in chunks if "Row 0 |" in c.text]
    assert len(table_chunks) == 1 and "Row 79 |" in table_chunks[0].text  # never split
    for c in chunks:
        if c is not table_chunks[0]:
            assert c.words <= MAX_WORDS
    assert any(c.text.startswith("Heading two") for c in chunks)  # a heading starts a passage


def test_long_paragraph_split_at_sentences():
    para = " ".join(f"Sentence number {i} has exactly seven words." for i in range(100))
    chunks = chunk_pages([(1, para)])
    assert len(chunks) >= 2
    assert all(c.words <= MAX_WORDS for c in chunks)
    assert all(c.text.rstrip().endswith(".") for c in chunks)


def test_repeated_paragraph_kept_once():
    text = f"## A\n{_words(60, 'alpha')}\n\n## B\n{_words(60, 'beta')}\n\n## A\n{_words(60, 'alpha')}"
    chunks = chunk_pages([(1, text)])
    assert sum(c.text.count("alpha") for c in chunks) == 60


def test_short_tail_merged():
    text = f"{_words(250)}.\n\n{_words(20, 'tail')}."
    chunks = chunk_pages([(1, text)])
    assert len(chunks) == 1 and "tail" in chunks[0].text
