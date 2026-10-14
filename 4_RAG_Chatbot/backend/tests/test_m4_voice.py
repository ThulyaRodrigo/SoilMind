"""Phase 6b voice: Sinhala number words, TTS text normalisation, WER/CER scoring, the STT term dictionary, audio
conversion, and the /stt, /tts, /voice-info routes (with the real local speech models when they are downloaded)."""

import io
import json
import re
import subprocess

import numpy as np
import pytest
import soundfile as sf

from member4_chatbot.voice import scoring
from member4_chatbot.voice.audio import AudioError, MAX_SECONDS, ffmpeg_exe, load_audio, to_wav_bytes
from member4_chatbot.voice.corrections import correct, load_rules
from member4_chatbot.voice.sinhala_numbers import (any_number_to_words, date_to_words, decimal_to_words,
                                                   number_to_words, numbers_to_words, time_to_words)
from member4_chatbot.voice.tts_text import en_numbers_to_words, prepare, sentence_lang, split_sentences

SI_LETTER = re.compile(r"[඀-෿]")


# ---- Sinhala numbers --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n, words", [
    (0, "බිංදුව"), (1, "එක"), (9, "නවය"), (10, "දහය"), (11, "එකොළහ"), (15, "පහළොව"), (19, "දහනවය"),
    (20, "විස්ස"), (25, "විසිපහ"), (30, "තිහ"), (45, "හතළිස්පහ"), (55, "පනස්පහ"), (70, "හැත්තෑව"), (99, "අනූනවය"),
    (100, "සියය"), (101, "එකසිය එක"), (120, "එකසිය විස්ස"), (200, "දෙසියය"), (285, "දෙසිය අසූපහ"),
    (500, "පන්සියය"), (1000, "දහස"), (1005, "එක්දහස් පහ"), (2000, "දෙදහස"), (2026, "දෙදහස් විසිහය"),
    (10000, "දසදහස"), (25000, "විසිපන්දහස"), (100000, "ලක්ෂ එක"), (150000, "ලක්ෂ එකයි පනස්දහස"),
])
def test_number_to_words(n, words):
    assert number_to_words(n) == words


def test_decimals_fractions_dates_times():
    assert decimal_to_words("5.5") == "පහ දශම පහ"
    assert decimal_to_words("0.42") == "බිංදුව දශම හතර දෙක"
    assert any_number_to_words("3 1/2") == "තුනහමාර"
    assert any_number_to_words("1/2") == "භාගය"
    assert any_number_to_words("5½") == "පහහමාර"
    assert date_to_words(2026, 10, 15) == "දෙදහස් විසිහය ඔක්තෝබර් පහළොව"
    assert time_to_words(14, 30) == "පැය දාහතර මිනිත්තු තිහ"
    assert time_to_words(4, 0) == "පැය හතර"


def test_numbers_in_a_sentence_keep_suffixes_and_handle_ranges():
    assert numbers_to_words("යූරියා කිලෝ 45ක්") == "යූරියා කිලෝ හතළිස්පහක්"
    assert numbers_to_words("දින 12-14") == "දින දොළහත් දාහතරත් අතර"
    assert numbers_to_words("මාස 4-4 1/2") == "මාස හතරත් හතරහමාරත් අතර"
    assert numbers_to_words("තෙතමනය 47.2%") == "තෙතමනය සියයට හතළිස්හත දශම දෙක"
    assert numbers_to_words(">10") == "දහයට වැඩි"
    assert numbers_to_words("1,500") == "එක්දහස් පන්සියය"
    assert numbers_to_words("ලකුණු 100න් 55.5") == "ලකුණු සියයෙන් පනස්පහ දශම පහ"
    assert numbers_to_words("2026-11-13 14:00") == "දෙදහස් විසිහය නොවැම්බර් දහතුන පැය දාහතර"
    assert not re.search(r"\d", numbers_to_words("1 2 3 10 99 100 1000 2026 0.5 3 1/2 20-30 5%"))


def test_english_numbers():
    assert en_numbers_to_words("45") == "forty-five"
    assert en_numbers_to_words("4-4 1/2 months") == "four to four and a half months"
    assert en_numbers_to_words("30%") == "thirty percent"
    assert "November" in en_numbers_to_words("2026-11-13")


# ---- TTS text normalisation ------------------------------------------------------------------------------
def test_units_terms_codes_sinhala():
    text = " ".join(s.text for s in prepare(
        "දුර්වලම ස්ථානය A3-06 හි නයිට්‍රජන් (87.7 mg/kg). pH 5.9. ටී.එස්.පී. කි.ග්‍රෑ. 55 යොදන්න. Bg 300 හොඳයි."))
    assert "ඒ තුන බිංදුව හය" in text  # node id spelled
    assert "කිලෝග්‍රෑමයකට මිලිග්‍රෑම් අසූහත දශම හත" in text  # unit read with the number
    assert "පී එච් පහ දශම නවය" in text
    assert "ටී එස් පී කිලෝග්‍රෑම් පනස්පහ" in text  # dotted abbreviations expanded, not split as sentences
    assert "බී ජී තුන්සියය" in text
    assert not re.search(r"[A-Za-z0-9]", text)


def test_terms_are_replaced_in_one_pass():
    # "TSP" -> "T S P" must not then have its "P" read as "phosphorus"
    s = prepare("Apply TSP and MOP; N is low.")[0]
    assert s.lang == "en"
    assert "T S P" in s.text and "M O P" in s.text and "nitrogen is low" in s.text
    assert "phosphorus" not in s.text and "potassium" not in s.text


def test_markup_and_citations_are_stripped():
    s = prepare("**Use** urea [2] (see https://doa.gov.lk/rrdi) now (p. 3).")
    assert len(s) == 1
    assert not re.search(r"https?|\[|\*|doa", s[0].text)


def test_sentences_tables_and_language_per_sentence():
    text = ("මේ සඳහා කෘෂිකර්ම දෙපාර්තමේන්තු පාඨය ඇත්තේ ඉංග්‍රීසියෙනි:\n"
            "Age Group | Time * | Urea | T.S.P\nFor 3 Month | Basic | | 55\n"
            "ඔබේ කුඹුර සඳහා යූරියා යොදන්න. 2026-11-13 සිට කරන්න.")
    ss = prepare(text)
    assert [s.lang for s in ss] == ["si", "en", "en", "si", "si"]
    assert ss[1].text == "Age Group, Time, urea, T S P."
    assert ss[2].text == "For three Month, Basic, fifty-five."
    assert split_sentences("එක. දෙක? තුන") == ["එක.", "දෙක?", "තුන"]
    assert sentence_lang("pH 5.5 නම් හුණු ඕනද?") == "si"


def test_sinhala_sentences_of_real_answers_have_no_leftovers():
    answers = [
        "ඔබේ කුඹුරේ (Plot A-3) සෞඛ්‍ය ලකුණු 100න් 55.5: අවධානයෙන් සිටින්න. අවසන් කියවීම: 2026-10-19 04:45.",
        "පාංශු විශ්ලේෂක අගයයන් | අක්කර එකක් සදහා\n5 -10 මි. ග්‍රෑම්/ කි. ග්‍රෑම් | 12 | |",
        "කුඹුරේ ගිනිකොන කොටසට ඩොලමයිට් මිටි 5½ක් පමණ (කි.ග්‍රෑ. 285) යොදන්න.",
        "කණගාටුයි, ඔබේ ප්‍රශ්නය යොමු කළා (යොමු අංකය ESC-C5F12D).",
        "රාත්‍රී කාලයේ අඩු උෂ්ණත්වය (17-20C)",
        "ඔබේ ප්‍රශ්නය යොමු කළා (යොමු අංකය ESC-3F9A1C).",  # "1C" is part of the code, not 1 °C
    ]
    for a in answers:
        for s in prepare(a):
            assert s.lang == "si"
            assert s.leftovers == [], (a, s.text)
            assert SI_LETTER.search(s.text)


# ---- scoring ---------------------------------------------------------------------------------------------
def test_scoring_normalisation():
    assert scoring.normalize("ටී.එස්.පී. මූලික පොහොර") == "ටීඑස්පී මූලික පොහොර"
    assert scoring.normalize("Bg 300") == scoring.normalize("බීජී 300") == "බීජී තුන්සියය"
    assert scoring.normalize("ප්‍රභේද") == scoring.normalize("ප්රභේද")
    assert scoring.cer(["යූරියා කිලෝ 45ක්"], ["යූරියා කිලෝ හතළිස්පහක්"]) == 0.0
    assert scoring.wer(["එක දෙක තුන හතර"], ["එක දෙක තුන පහ"]) == pytest.approx(0.25)
    assert scoring.cer(["අබ"], [""]) == 1.0  # an empty transcript is all deletions
    assert scoring.word_substitutions("යූරියා කීයක් දාන්න", "යුරියා කීයක් දාන්න") == [("යූරියා", "යුරියා")]


# ---- STT term dictionary -----------------------------------------------------------------------------------
def _rules(tmp_path, entries):
    import yaml

    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"corrections": entries}, allow_unicode=True), encoding="utf-8")
    load_rules.cache_clear()
    return load_rules(str(p))


def test_corrections_whole_words_only_and_logged(tmp_path, monkeypatch):
    import member4_chatbot.voice.corrections as C

    monkeypatch.setattr(C, "LOG_PATH", tmp_path / "log.jsonl")
    rules = _rules(tmp_path, [{"heard": "යුරියා", "write": "යූරියා"}, {"heard": "ඩොලමයිට", "write": "ඩොලමයිට්"}])
    text, changes = correct("යුරියා දාන්න, යුරියාවලට නෙවෙයි. ඩොලමයිට දාන්න", rules, context={"model": "t"})
    assert text == "යූරියා දාන්න, යුරියාවලට නෙවෙයි. ඩොලමයිට් දාන්න"  # not inside a longer word
    assert {c["heard"] for c in changes} == {"යුරියා", "ඩොලමයිට"}
    logged = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(logged) == 1 and logged[0]["model"] == "t" and len(logged[0]["changes"]) == 2
    assert correct("නිවැරදි වාක්‍යයක්", rules, log_to_file=False) == ("නිවැරදි වාක්‍යයක්", [])


def test_shipped_dictionary_loads_and_entries_are_safe():
    load_rules.cache_clear()
    rules = load_rules()
    import yaml

    from member4_chatbot.voice.corrections import CORRECTIONS_YAML

    entries = yaml.safe_load(CORRECTIONS_YAML.read_text(encoding="utf-8"))["corrections"] or []
    assert len(rules) == len(entries)
    for e in entries:
        assert e["heard"] != e["write"] and e.get("evidence"), e
        # a dictionary entry must never rewrite a correct farming term
        assert correct(e["write"], rules, log_to_file=False)[0] == e["write"]


# ---- audio conversion ------------------------------------------------------------------------------------
def _tone(seconds: float, sr: int = 44100, lead: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    return np.concatenate([np.zeros(int(lead * sr), np.float32), x, np.zeros(int(lead * sr), np.float32)])


def _encode(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    wav = to_wav_bytes(audio, sr)
    if fmt == "wav":
        return wav
    codec = {"webm": "libopus", "ogg": "libopus"}[fmt]
    p = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-i", "pipe:0", "-c:a", codec, "-f", fmt, "pipe:1"],
                       input=wav, capture_output=True, check=True)
    return p.stdout


@pytest.mark.parametrize("fmt", ["webm", "ogg", "wav"])
def test_browser_audio_to_16k_mono_with_silence_trimmed(fmt):
    data = _encode(_tone(2.0, lead=1.0), 44100, fmt)
    audio, info = load_audio(data)
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert info["input_seconds"] == pytest.approx(4.0, abs=0.1)
    assert info["trimmed"] and info["seconds"] == pytest.approx(2.0, abs=0.15)  # 1 s of silence cut at each end


def test_audio_capped_at_30_seconds_and_bad_input_rejected():
    audio, info = load_audio(_encode(_tone(35.0, sr=16000), 16000, "wav"))
    assert audio.size / 16000 == pytest.approx(MAX_SECONDS, abs=0.05) and info["capped"]
    with pytest.raises(AudioError):
        load_audio(b"this is not audio at all")
    with pytest.raises(AudioError):
        load_audio(b"")
    silence = to_wav_bytes(np.zeros(16000, np.float32), 16000)
    audio, info = load_audio(silence)  # all silence: kept untrimmed rather than emptied
    assert audio.size == 16000 and not info["trimmed"]


# ---- API routes with the real models ----------------------------------------------------------------------
def _models_present() -> bool:
    try:
        from huggingface_hub import snapshot_download

        from member4_chatbot.voice.stt import resolve_model
        from shared.config import get_settings

        snapshot_download(resolve_model(get_settings().stt_model), local_files_only=True)
        snapshot_download("dialoglk/SinhalaVITS-TTS-M2", local_files_only=True)
        snapshot_download("facebook/mms-tts-eng", local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


needs_voice = pytest.mark.skipif(not _models_present(), reason="speech models not downloaded")


@needs_voice
def test_tts_route_returns_wav_and_caches(client):
    text = "ඔබේ කුඹුරට අක්කරයකට යූරියා කි.ග්‍රෑ. 20ක් යොදන්න. Apply it in 2 weeks."
    r = client.post("/api/m4/tts", json={"text": text, "lang": "si"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    audio, sr = sf.read(io.BytesIO(r.content))
    assert sr == 22050 and len(audio) / sr > 2.0 and np.abs(audio).max() > 0.05
    assert r.headers["x-tts-sentences"] == "2" and r.headers["x-tts-failures"] == "0"
    assert "SinhalaVITS" in r.headers["x-tts-engines"] and "mms-tts-eng" in r.headers["x-tts-engines"]
    again = client.post("/api/m4/tts", json={"text": text})
    assert again.headers["x-tts-cache"] == "hit" and again.content == r.content
    assert client.post("/api/m4/tts", json={"text": "… ! ?"}).status_code == 422
    assert client.post("/api/m4/tts", json={"text": ""}).status_code == 422


@needs_voice
def test_stt_route_round_trip_with_our_tts(client):
    """Sinhala spoken by our TTS, sent as browser-style Opus/webm, must come back as close Sinhala text."""
    sentence = "යූරියා පොහොර කොච්චර දාන්න ඕනද"
    wav = client.post("/api/m4/tts", json={"text": sentence}).content
    audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
    webm = _encode(audio, sr, "webm")
    r = client.post("/api/m4/stt", files={"audio": ("q.webm", webm, "audio/webm")}, data={"lang": "si"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"text", "language", "duration", "model", "raw_text", "corrections"}
    assert body["language"] == "si" and 0.5 < body["duration"] <= 30
    assert SI_LETTER.search(body["text"])
    assert scoring.cer([sentence], [body["text"]]) < 0.35, body["text"]


@needs_voice
def test_stt_route_rejects_bad_uploads(client):
    assert client.post("/api/m4/stt", files={"audio": ("x.webm", b"not audio", "audio/webm")}).status_code == 422
    assert client.post("/api/m4/stt", files={"audio": ("x.wav", to_wav_bytes(np.zeros(800, np.float32), 16000))},
                       data={"lang": "fr"}).status_code == 422


def test_voice_info_and_openapi(client):
    r = client.get("/api/m4/voice-info")
    assert r.status_code == 200
    body = r.json()
    assert body["stt_model"] and "voice" in body["tts"] and isinstance(body["metrics"], dict)
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/m4/stt" in paths and "/api/m4/tts" in paths and "/api/m4/voice-info" in paths


# ---- recording prompts and test sets ----------------------------------------------------------------------
def test_record_prompts_split_and_sources():
    import pandas as pd

    from member4_chatbot.paths import VOICE_EVAL_DIR
    from member4_chatbot.pipeline.voice_prompts import build

    saved = pd.read_csv(VOICE_EVAL_DIR / "record_prompts.csv", keep_default_na=False)
    rebuilt = build()
    assert saved.prompt_id.tolist() == rebuilt.prompt_id.tolist() and saved.split.tolist() == rebuilt.split.tolist()
    assert 200 <= len(saved) <= 300 and saved.prompt_id.is_unique and saved.text.is_unique
    assert set(saved.split) == {"finetune", "test"}
    assert 0.25 <= (saved.split == "test").mean() <= 0.35
    for _, g in saved.groupby("category"):  # every category has held-out phrases
        assert (g.split == "test").any()
    assert all(SI_LETTER.search(t) for t in saved.text)
    assert (saved.checked_by_human == "no").all()
    for term in ("යූරියා", "ටී.එස්.පී", "එම්.ඕ.පී", "ඩොලමයිට්", "නයිට්‍රජන්", "යල", "මහ"):
        assert saved.text.str.contains(term, regex=False).any(), term


def test_synthetic_set_keeps_the_prompt_splits():
    import pandas as pd

    from member4_chatbot.paths import VOICE_EVAL_DIR

    path = VOICE_EVAL_DIR / "synthetic_test.csv"
    if not path.exists():
        pytest.skip("synthetic set not built")
    syn = pd.read_csv(path, keep_default_na=False)
    prompts = pd.read_csv(VOICE_EVAL_DIR / "record_prompts.csv", keep_default_na=False).set_index("prompt_id")
    assert (syn.split == prompts.loc[syn.prompt_id, "split"].to_numpy()).all()
    assert syn.groupby("prompt_id").voice.nunique().eq(2).all()
