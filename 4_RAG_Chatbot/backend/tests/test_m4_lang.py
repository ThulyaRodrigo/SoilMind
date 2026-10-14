"""Language detection (Phase 5, task 7)."""

import pytest

from member4_chatbot.lang import detect_language


@pytest.mark.parametrize("text, lang", [
    ("How much urea should I apply?", "en"),
    ("මගේ කුඹුරට යූරියා කොච්චර දාන්න ඕනද?", "si"),
    ("Bg 300 අස්වනු නෙලීමට දින කීයක් යනවද?", "si"),  # variety names in Latin letters inside Sinhala
    ("ටී.එස්.පී. vs TSP", "si"),
    ("mage kumburata pohora kochchara danna oneda", "si_latn"),
    ("ayubowan", "si_latn"),
    ("Is Bg 300 resistant to blast?", "en"),
    ("1234 ?!", "unknown"),
    ("", "unknown"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang
