"""Voice evaluation: STT word / character error rates and latency, TTS failures, latency and intelligibility.

Test sets (all manifests in eval/voice/):
- `openslr30`: 200 utterances of real read Sinhala speech, 12 speakers (OpenSLR-30, Google, CC BY-SA 4.0). General
  Sinhala, not farming. Built by `pipeline.voice_data`.
- `synthetic`: the 249 recording prompts (record_prompts.csv) spoken by our TTS voices, 2 voices per prompt (4
  SinhalaVITS speakers + Piper). SYNTHETIC SPEECH: it measures farming vocabulary and numbers, not real farmers'
  voices, accents or field noise. It stands in until the team records the prompts (`farm`, below).
- `farm`: the team's recordings of the prompts (eval/voice/recordings/manifest.csv, made by
  `pipeline.voice_recordings`). Empty until the team records.

Every set has a `split` column for the prompts: `finetune` phrases may be used to build the term dictionary or to
fine-tune; scores "on farming phrases" are reported on `test` phrases only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from member4_chatbot.paths import MEMBER_DIR, VOICE_DATA_DIR, VOICE_EVAL_DIR
from member4_chatbot.voice import scoring
from member4_chatbot.voice.corrections import correct, load_rules

RESULTS_DIR = VOICE_EVAL_DIR / "results"
METRICS_JSON = RESULTS_DIR / "metrics.json"
SYNTH_DIR = VOICE_DATA_DIR / "synthetic"
SETS = {
    "openslr30": VOICE_EVAL_DIR / "openslr30_test.csv",
    "synthetic": VOICE_EVAL_DIR / "synthetic_test.csv",
    "farm": VOICE_EVAL_DIR / "recordings" / "manifest.csv",
}
SYNTH_VOICES = ["F1", "M1", "F2", "M2", "piper"]
SEED = 20261019


def load_set(name: str) -> pd.DataFrame:
    df = pd.read_csv(SETS[name], keep_default_na=False)
    if "split" not in df.columns:
        df["split"] = "test"
    return df


# ---- metrics.json (one file, updated step by step) ---------------------------------------------------------
def read_metrics() -> dict:
    return json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {}


def update_metrics(section: str, key: str, value) -> None:
    m = read_metrics()
    m.setdefault(section, {})[key] = value
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


# ---- STT ---------------------------------------------------------------------------------------------------
def run_stt(model: str, set_name: str, device: str = "cpu", limit: int | None = None, splits=("finetune", "test"),
            hardware: str = "") -> pd.DataFrame:
    """Transcribe a test set; one row per clip with the raw transcript, the dictionary transcript and timing."""
    from member4_chatbot.voice.audio import load_audio
    from member4_chatbot.voice.stt import WhisperStt

    df = load_set(set_name)
    df = df[df.split.isin(splits)]
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=SEED).sort_index()
    stt = WhisperStt(model, device)
    rules = load_rules()
    stt.transcribe_array(np.zeros(16000, np.float32))  # warm-up, not timed
    rows = []
    for r in df.itertuples():
        audio, info = load_audio((MEMBER_DIR / r.file).read_bytes())
        t0 = time.perf_counter()
        raw = stt.transcribe_array(audio, "si")
        ms = (time.perf_counter() - t0) * 1000
        fixed, changes = correct(raw, rules, log_to_file=False)
        rows.append({"id": getattr(r, "utt_id", None) or r.clip_id, "split": r.split, "ref": r.text, "hyp_raw": raw,
                     "hyp_dict": fixed, "corrections": json.dumps(changes, ensure_ascii=False),
                     "audio_s": info["seconds"], "stt_ms": round(ms, 1)})
    out = pd.DataFrame(rows)
    out["model"], out["device"], out["set"] = stt.model_id, device, set_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_n{limit}" if limit else ""  # a sample run (e.g. a timing run) does not overwrite the full-set results
    out.to_csv(RESULTS_DIR / f"stt_{model}_{set_name}_{device}{tag}.csv", index=False, encoding="utf-8")
    update_metrics("stt", f"{model}|{set_name}|{device}{tag}", summarize_stt(out, hardware))
    return out


def summarize_stt(df: pd.DataFrame, hardware: str = "") -> dict:
    def part(d: pd.DataFrame) -> dict:
        if d.empty:
            return {}
        return {"n": len(d),
                "wer": round(scoring.wer(list(d.ref), list(d.hyp_raw)), 4),
                "cer": round(scoring.cer(list(d.ref), list(d.hyp_raw)), 4),
                "wer_dict": round(scoring.wer(list(d.ref), list(d.hyp_dict)), 4),
                "cer_dict": round(scoring.cer(list(d.ref), list(d.hyp_dict)), 4),
                "clips_corrected": int((d.hyp_raw != d.hyp_dict).sum())}
    lat = df.stt_ms.to_numpy()
    return {"model": df.model.iloc[0], "device": df.device.iloc[0], "hardware": hardware,
            "all": part(df), "finetune_split": part(df[df.split == "finetune"]), "test_split": part(df[df.split == "test"]),
            "latency_ms_median": round(float(np.median(lat)), 1), "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
            "audio_s_median": round(float(df.audio_s.median()), 2),
            "real_time_factor": round(float(lat.sum() / 1000 / df.audio_s.sum()), 3)}


def rescore_dictionary(model: str, set_name: str, device: str = "cpu") -> dict:
    """Re-apply the current dictionary to saved transcripts (no model run) and update metrics.json."""
    path = RESULTS_DIR / f"stt_{model}_{set_name}_{device}.csv"
    df = pd.read_csv(path, keep_default_na=False)
    rules = load_rules.__wrapped__()  # read the YAML again
    fixed = [correct(h, rules, log_to_file=False) for h in df.hyp_raw]
    df["hyp_dict"] = [f for f, _ in fixed]
    df["corrections"] = [json.dumps(c, ensure_ascii=False) for _, c in fixed]
    df.to_csv(path, index=False, encoding="utf-8")
    old = read_metrics().get("stt", {}).get(f"{model}|{set_name}|{device}", {})
    summary = summarize_stt(df, old.get("hardware", ""))
    update_metrics("stt", f"{model}|{set_name}|{device}", summary)
    return summary


def dictionary_candidates(result_csvs: list[Path], min_count: int = 2) -> pd.DataFrame:
    """Word substitutions (reference word -> what the model wrote) on FINETUNE-split clips only, most frequent first.
    A person then picks the safe ones (farming terms, never common words) into data/voice/stt_corrections.yaml."""
    pairs = []
    for p in result_csvs:
        df = pd.read_csv(p, keep_default_na=False)
        df = df[df.split == "finetune"]
        for r in df.itertuples():
            for ref_w, hyp_w in scoring.word_substitutions(r.ref, r.hyp_raw):
                pairs.append({"write": ref_w, "heard": hyp_w, "model": r.model, "set": r.set})
    if not pairs:
        return pd.DataFrame(columns=["write", "heard", "count", "models"])
    c = pd.DataFrame(pairs)
    out = (c.groupby(["write", "heard"]).agg(count=("model", "size"), models=("model", lambda s: ", ".join(sorted(set(s)))))
           .reset_index().sort_values("count", ascending=False))
    return out[out["count"] >= min_count]


# ---- synthetic farming set ----------------------------------------------------------------------------------
def synthesize_testset() -> pd.DataFrame:
    """Each recording prompt spoken by 2 of the 5 TTS voices (fixed rotation), as 16 kHz WAV."""
    import soundfile as sf

    from member4_chatbot.voice.audio import resample
    from member4_chatbot.voice.tts import OUT_RATE, TtsService
    from member4_chatbot.voice.tts_text import prepare

    prompts = pd.read_csv(VOICE_EVAL_DIR / "record_prompts.csv", keep_default_na=False)
    services = {v: TtsService(voice=v if v != "piper" else "F1", engine="piper" if v == "piper" else "sinhalavits",
                              cache=False) for v in SYNTH_VOICES}
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(prompts.itertuples()):
        for voice in (SYNTH_VOICES[i % 5], SYNTH_VOICES[(i + 2) % 5]):
            audio, spoken = services[voice].speak_sentences(prepare(p.text))
            clip = f"{p.prompt_id}_{voice}"
            sf.write(SYNTH_DIR / f"{clip}.wav", resample(audio, OUT_RATE, 16000), 16000, subtype="PCM_16")
            rows.append({"clip_id": clip, "prompt_id": p.prompt_id, "voice": voice, "text": p.text,
                         "category": p.category, "split": p.split, "file": f"data/voice/synthetic/{clip}.wav",
                         "duration_s": round(audio.size / OUT_RATE, 2),
                         "tts_leftovers": "; ".join(x for s in spoken for x in s.leftovers)})
    df = pd.DataFrame(rows)
    df.to_csv(SETS["synthetic"], index=False, encoding="utf-8")
    return df


# ---- TTS ----------------------------------------------------------------------------------------------------
def tts_texts() -> pd.DataFrame:
    """What the chatbot really says: every answer of the 60-question evaluation run, plus the fixed texts."""
    from member4_chatbot.chat import text as T

    res = pd.read_csv(MEMBER_DIR / "eval" / "results" / "results.csv", keep_default_na=False)
    rows = [{"id": r.id, "lang": r.lang, "kind": r.path, "text": r.answer} for r in res.itertuples() if r.answer.strip()]
    for name in ("GREETING", "REFUSE_OUT_OF_SCOPE", "REFUSE_NOT_FOUND"):
        for lang, txt in getattr(T, name).items():
            rows.append({"id": f"{name}_{lang}", "lang": lang, "kind": "fixed", "text": txt.replace("{ref}", "ESC-3F9A1C")})
    return pd.DataFrame(rows)


def run_tts_eval(engines=("sinhalavits", "piper"), voice: str = "M2", roundtrip_model: str | None = "small",
                 hardware: str = "") -> pd.DataFrame:
    """Per sentence: the engine, leftovers the voice cannot read, synthesis time, and (Sinhala) the CER of our STT
    on the synthesised audio against the normalised text (an automatic intelligibility check, not a human score)."""
    from member4_chatbot.voice.audio import resample
    from member4_chatbot.voice.tts import OUT_RATE, TtsService
    from member4_chatbot.voice.tts_text import prepare

    stt = None
    if roundtrip_model:
        from member4_chatbot.voice.stt import WhisperStt

        stt = WhisperStt(roundtrip_model, "cpu")
    texts = tts_texts()
    rows = []
    for engine in engines:
        svc = TtsService(voice=voice, engine=engine, cache=False)
        svc.speak_sentences(prepare("ආයුබෝවන්. Hello."))  # load + warm up, not timed
        for t in texts.itertuples():
            for k, s in enumerate(prepare(t.text)):
                audio, spoken = svc.speak_sentences([s])
                sp = spoken[0]
                row = {"engine": engine if s.lang == "si" else "mms", "text_id": t.id, "answer_lang": t.lang,
                       "kind": t.kind, "n": k, "lang": s.lang, "source": s.source, "spoken": s.text,
                       "leftovers": "; ".join(sp.leftovers), "failed": bool(sp.leftovers), "audio_s": sp.seconds,
                       "synth_ms": sp.synth_ms, "chars": len(s.text)}
                if stt is not None and s.lang == "si" and audio.size:
                    hyp = stt.transcribe_array(resample(audio, OUT_RATE, 16000)[: 30 * 16000], "si")
                    row["roundtrip_hyp"] = hyp
                    row["roundtrip_cer"] = round(scoring.cer([s.text], [hyp]), 4)
                rows.append(row)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["engine", "text_id", "n"])  # English sentences are spoken by MMS in both runs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "tts_sentences.csv", index=False, encoding="utf-8")
    summary = {}
    for engine, d in df.groupby("engine"):
        summary[engine] = {
            "sentences": len(d), "failed_sentences": int(d.failed.sum()),
            "failure_rate": round(float(d.failed.mean()), 4),
            "latency_ms_per_sentence_median": round(float(d.synth_ms.median()), 1),
            "latency_ms_per_sentence_p95": round(float(np.percentile(d.synth_ms, 95)), 1),
            "real_time_factor": round(float(d.synth_ms.sum() / 1000 / max(d.audio_s.sum(), 1e-6)), 3),
            "roundtrip_cer": (round(scoring.cer(list(d.spoken), list(d.roundtrip_hyp)), 4)
                              if "roundtrip_hyp" in d and d.roundtrip_hyp.notna().any() else None),
        }
    summary["_info"] = {"voice": voice, "hardware": hardware, "texts": len(texts),
                        "roundtrip_stt": roundtrip_model,
                        "note": "roundtrip_cer = CER of our STT on the TTS audio vs the text the voice was given; "
                                "an automatic intelligibility proxy, not a listening score"}
    update_metrics("tts", "cpu", summary)
    return df


def listening_sheet(n: int = 20, voice: str = "M2") -> Path:
    """n Sinhala answers spoken by SinhalaVITS and Piper, plus a spreadsheet for a Sinhala speaker's 1-5 scores."""
    import openpyxl
    from openpyxl.worksheet.datavalidation import DataValidation

    from member4_chatbot.voice.tts import TtsService

    texts = tts_texts()
    texts = texts[texts.lang == "si"].drop_duplicates("text").sample(n=n, random_state=SEED)
    out_dir = VOICE_EVAL_DIR / "listening"
    out_dir.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "scores"
    ws.append(["item", "audio_file", "engine", "text", "naturalness_1_5", "numbers_correct_yes_no", "rater", "notes"])
    rng = np.random.default_rng(SEED)
    item = 0
    for engine in ("sinhalavits", "piper"):
        svc = TtsService(voice=voice, engine=engine, cache=False)
        for t in texts.itertuples():
            item += 1
            res = svc.speak(t.text)
            name = f"L{item:02d}_{rng.integers(1000, 9999)}.wav"  # the file name does not reveal the engine
            (out_dir / name).write_bytes(res.wav)
            ws.append([item, name, engine, t.text, None, None, None, None])
    dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5")
    ws.add_data_validation(dv)
    dv.add(f"E2:E{item + 1}")
    yn = DataValidation(type="list", formula1='"yes,no"')
    ws.add_data_validation(yn)
    yn.add(f"F2:F{item + 1}")
    how = wb.create_sheet("how to score")
    for line in ["Listen to each file and give a score from 1 to 5:",
                 "5 = natural, like a person; 4 = clear, slightly robotic; 3 = understandable with effort;",
                 "2 = many words unclear; 1 = cannot understand.",
                 "numbers_correct: are all numbers and units said correctly? (yes / no)",
                 "Hide column C (engine) before scoring, so you do not know which voice is which.",
                 "Then run notebook 07 to read the scores."]:
        how.append([line])
    path = out_dir / "listening_scores.xlsx"
    wb.save(path)
    return path
