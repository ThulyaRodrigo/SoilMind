"""Speech pre-warm at startup (Phase 7): the states /api/health reports, with fake speech models (the real ones are
exercised in test_m4_voice.py and by scripts/demo_check.py)."""

import numpy as np
import pytest

from member4_chatbot.voice import warmup


class FakeStt:
    model_id = "fake/whisper"

    def __init__(self):
        self.heard = []

    def transcribe_array(self, audio, language):
        self.heard.append((len(audio), language))
        return "මගේ කුඹුර අද කොහොමද?"


class FakeTts:
    def __init__(self, fail=False):
        self.fail, self.spoken = fail, []

    def engine_for(self, lang):
        if self.fail:
            raise RuntimeError("no voice")

    def speak_sentences(self, sentences):
        self.spoken += [s.text for s in sentences]
        return np.zeros(22050, np.float32), []

    def status(self):
        return {"loaded": {"sinhalavits": "SinhalaVITS-M2", "mms": "MMS-eng"}}


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(warmup, "_state", {"state": "not_started", "detail": "", "timings_ms": {}})
    monkeypatch.setattr(warmup, "_thread", None)


def test_warmup_loads_speaks_and_transcribes(fresh, monkeypatch):
    from member4_chatbot.voice import stt, tts

    s, t = FakeStt(), FakeTts()
    monkeypatch.setattr(stt, "get_stt", lambda: s)
    monkeypatch.setattr(tts, "get_tts", lambda: t)
    st = warmup.run()
    assert st["state"] == "ready" and "fake/whisper" in st["detail"]
    assert set(st["timings_ms"]) == {"load_stt", "load_tts_si", "load_tts_en", "warm_tts_si", "warm_tts_en",
                                     "warm_stt_si"}
    assert s.heard == [(16000, "si")]   # one second of 22.05 kHz audio, resampled to the 16 kHz the STT takes
    assert any("කුඹුර" in x for x in t.spoken) and any("field" in x for x in t.spoken)


def test_warmup_failure_is_reported_not_raised(fresh, monkeypatch):
    from member4_chatbot.voice import stt, tts

    monkeypatch.setattr(stt, "get_stt", lambda: FakeStt())
    monkeypatch.setattr(tts, "get_tts", lambda: FakeTts(fail=True))
    st = warmup.run()
    assert st["state"] == "failed" and "no voice" in st["detail"]


def test_background_start_runs_once(fresh, monkeypatch):
    calls = []
    monkeypatch.setattr(warmup, "run", lambda: calls.append(1) or warmup.state())
    warmup.start_background()
    warmup.start_background()
    warmup.wait(5)
    assert calls == [1]


# ------------------------------------------------------------------ speaking an answer part by part (Phase 7)
def test_speak_parts_first_part_is_one_sentence_and_nothing_is_lost():
    from member4_chatbot.voice.tts_text import clean_markup, expand_sinhala_abbreviations, speak_parts, split_sentences

    text = ("ඔබේ කුඹුරේ ගිනිකොන කොටස: නයිට්‍රජන් අඩුයි. A3-05 80.6 mg/kg. අක්කරයකට කි.ග්‍රෑ. 20 යොදන්න.\n"
            "ඔබේ කුඹුරේ සෞඛ්‍ය ලකුණු 100න් 25.3. අවසන් කියවීම: 2026-10-20 00:38.")
    sentences = split_sentences(expand_sinhala_abbreviations(clean_markup(text)))  # the split /tts itself uses
    parts, n = speak_parts(text)
    assert n == len(sentences) >= 4
    assert parts[0] == sentences[0]                                   # sound starts after one sentence
    assert len(parts) == 1 + -(-(n - 1) // 2)                         # then two sentences per part
    assert " ".join(parts) == " ".join(sentences)                     # nothing lost or reordered
    joined = " ".join(parts).replace(" ", "")
    for word in ("ගිනිකොන", "80.6", "25.3", "2026-10-20"):
        assert word in joined
    assert "කි.ග්‍රෑ" not in joined          # the abbreviation is expanded, so its dots never split a sentence
    assert speak_parts("") == ([], 0)


def test_tts_parts_route(client):
    r = client.post("/api/m4/tts/parts", json={"text": "How is my field? It is fine. Apply urea. Then wait.", "lang": "en"})
    assert r.status_code == 200
    body = r.json()
    assert body["parts"][0] == "How is my field?" and body["sentences"] == 4 and len(body["parts"]) == 3
    assert client.post("/api/m4/tts/parts", json={"text": "...", "lang": "en"}).status_code == 422
