"""Text-to-speech: answer text -> one WAV file, spoken sentence by sentence. Every model runs locally.

| Language | Engine | Model | Licence |
|---|---|---|---|
| Sinhala (primary) | SinhalaVITS (Coqui VITS, 22.05 kHz) | dialoglk/SinhalaVITS-TTS-{F1,F2,M1,M2}, Dialog Axiata + Dialog-UoM lab | MPL-2.0 |
| Sinhala (fallback) | Piper (VITS, ONNX, eSpeak-ng phonemes) | chan4lk/piper-tts-sinhala, trained on OpenSLR-30 | MIT (data CC BY-SA 4.0) |
| English | MMS-TTS (VITS, 16 kHz) | facebook/mms-tts-eng | CC BY-NC 4.0 |

The brief names `facebook/mms-tts-sin` as the fallback, but Meta's MMS-TTS has no Sinhala voice (that model
does not exist), so Piper's Sinhala voice is used instead.

How a request is spoken:
1. `tts_text.prepare` cleans the text and splits it into sentences, each with its language (a Sinhala answer can
   quote an English passage), and writes numbers, units and codes as words.
2. Each sentence is synthesised by the voice of its language, then all are resampled to 22.05 kHz and joined with
   a short pause.
3. The WAV is cached under data/voice/tts_cache/ by a hash of (text, voice, engine, terms map version), so the same
   answer is synthesised once.

VITS samples noise at inference; the seed is fixed per sentence, so the same text always gives the same audio.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from member4_chatbot.paths import VOICE_DATA_DIR
from member4_chatbot.voice.audio import resample, to_wav_bytes
from member4_chatbot.voice.tts_text import Sentence, prepare, terms_version

log = logging.getLogger("soilmind.m4.tts")

OUT_RATE = 22050
PAUSE_S = 0.25
SEED = 20261019
CACHE_DIR = VOICE_DATA_DIR / "tts_cache"
SINHALAVITS_REPO = "dialoglk/SinhalaVITS-TTS-{voice}"
PIPER_REPO, PIPER_MODEL = "chan4lk/piper-tts-sinhala", "si_LK-sinhala-medium.onnx"
MMS_ENG = "facebook/mms-tts-eng"


class TtsUnavailable(RuntimeError):
    """No engine could be loaded for a language."""


def _coqui_import_shim() -> None:
    """coqui-tts refuses to import without torchaudio and (for torch >= 2.9) torchcodec. Neither is used for VITS
    *inference* (only for loading training audio and speaker-encoder resampling), and no torchaudio build exists
    for our torch 2.14. When they are missing, register empty placeholder modules that raise if anything ever
    touches them, so a hidden use fails loudly instead of silently."""
    import importlib.machinery
    import importlib.util
    import sys
    import types

    for name in ("torchaudio", "torchcodec"):
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            continue
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__version__ = "0+soilmind-placeholder"

        def _missing(attr: str, _name: str = name):
            if attr.startswith("__"):
                raise AttributeError(attr)
            raise ImportError(f"{_name}.{attr} is not installed (placeholder; SoilMind only runs VITS inference)")

        mod.__getattr__ = _missing
        sys.modules[name] = mod


def _seed(text: str) -> None:
    import torch

    torch.manual_seed(SEED + int(hashlib.md5(text.encode()).hexdigest()[:6], 16))


# ---- engines --------------------------------------------------------------------------------------------
class SinhalaVits:
    """dialoglk/SinhalaVITS-TTS-<voice>: Sinhala text -> romanized Sinhala -> Coqui VITS."""

    engine = "sinhalavits"

    def __init__(self, voice: str = "M2", device: str = "cpu"):
        from huggingface_hub import snapshot_download

        _coqui_import_shim()
        from TTS.utils.synthesizer import Synthesizer

        self.voice = voice.upper()
        folder = Path(snapshot_download(SINHALAVITS_REPO.format(voice=self.voice)))
        ckpt = next(folder.glob("*.pth"))
        config = next(folder.glob("*_config.json"))
        self.name = f"{SINHALAVITS_REPO.format(voice=self.voice)} ({ckpt.stem})"
        self.synth = Synthesizer(tts_checkpoint=str(ckpt), tts_config_path=str(config), use_cuda=device == "cuda")
        self.sample_rate = self.synth.output_sample_rate
        chars = json.loads(config.read_text(encoding="utf-8"))["characters"]
        self.vocab = set(chars["characters"]) | set(chars["punctuations"])

    def romanize(self, text: str) -> tuple[str, list[str]]:
        """Sinhala -> the model's alphabet; returns the characters it cannot read (dropped)."""
        from member4_chatbot.voice.romanizer import sinhala_to_roman

        roman = sinhala_to_roman(text).lower()
        roman = roman.replace("w", "v").replace("q", "k").replace("x", "ks").replace("z", "s")
        roman = re.sub(r"[\"“”]", "", roman).replace("’", "'").replace("—", ",").replace("–", ",")
        dropped = sorted({c for c in roman if c not in self.vocab and c.isalnum()})  # punctuation is only a pause
        return "".join(c for c in roman if c in self.vocab or c.isspace()), dropped

    def synthesize(self, text: str) -> tuple[np.ndarray, list[str]]:
        roman, dropped = self.romanize(text)
        _seed(roman)
        wav = np.asarray(self.synth.tts(roman, split_sentences=False), dtype=np.float32)
        return wav, dropped


class PiperSinhala:
    engine = "piper"

    def __init__(self):
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        model = hf_hub_download(PIPER_REPO, PIPER_MODEL)
        config = hf_hub_download(PIPER_REPO, PIPER_MODEL + ".json")
        self.voice = PiperVoice.load(model, config_path=config)
        self.sample_rate = self.voice.config.sample_rate
        self.name = f"{PIPER_REPO} ({PIPER_MODEL})"

    # eSpeak-ng writes the prenasalised consonants (ඹ ඳ ඬ ඟ) as ᵐb ⁿd ..., which this voice's phoneme map lacks
    # (Piper would silently drop the nasal); spelling them as nasal + stop keeps the sound.
    PRENASAL = {"ඹ": "ම්බ", "ඳ": "න්ද", "ඬ": "ණ්ඩ", "ඟ": "ඞ්ග"}

    def synthesize(self, text: str) -> tuple[np.ndarray, list[str]]:
        for k, v in self.PRENASAL.items():
            text = text.replace(k, v)
        chunks = [c.audio_float_array for c in self.voice.synthesize(text)]
        return (np.concatenate(chunks) if chunks else np.zeros(0, np.float32)), []


class MmsEnglish:
    engine = "mms"

    def __init__(self, device: str = "cpu"):
        from transformers import AutoTokenizer, VitsModel

        self.tok = AutoTokenizer.from_pretrained(MMS_ENG)
        self.model = VitsModel.from_pretrained(MMS_ENG).to(device).eval()
        self.device = device
        self.sample_rate = self.model.config.sampling_rate
        self.name = MMS_ENG
        self.vocab = set(self.tok.get_vocab())

    def synthesize(self, text: str) -> tuple[np.ndarray, list[str]]:
        import torch

        text = text.lower()
        dropped = sorted({c for c in text if c not in self.vocab and c.isalnum()})
        text = "".join(c if c in self.vocab else " " for c in text).strip()
        _seed(text)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            wav = self.model(**inputs).waveform[0].cpu().numpy()
        return wav.astype(np.float32), dropped


# ---- the service ----------------------------------------------------------------------------------------------
@dataclass
class SpokenSentence:
    lang: str
    engine: str
    text: str
    seconds: float
    synth_ms: float
    leftovers: list[str] = field(default_factory=list)


@dataclass
class TtsResult:
    wav: bytes
    seconds: float
    cached: bool
    key: str
    engines: list[str]
    sentences: list[SpokenSentence]
    total_ms: float

    @property
    def failures(self) -> int:
        return sum(bool(s.leftovers) for s in self.sentences)


class TtsService:
    def __init__(self, voice: str = "M2", engine: str = "auto", device: str = "cpu", cache: bool = True):
        self.voice, self.engine_choice, self.device, self.cache = voice.upper(), engine, device, cache
        self._engines: dict[str, object] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()
        self._load_lock = threading.RLock()  # the startup warm-up thread and a request may load together

    # engines load lazily, once
    def _load(self, name: str):
        with self._load_lock:
            return self._load_locked(name)

    def _load_locked(self, name: str):
        if name in self._engines:
            return self._engines[name]
        if name in self._errors:
            return None
        try:
            t0 = time.perf_counter()
            eng = {"sinhalavits": lambda: SinhalaVits(self.voice, self.device), "piper": PiperSinhala,
                   "mms": lambda: MmsEnglish(self.device)}[name]()
            self._engines[name] = eng
            log.info("tts: %s loaded in %.1f s", eng.name, time.perf_counter() - t0)
            return eng
        except Exception as exc:  # noqa: BLE001 - fall back to the next engine
            self._errors[name] = f"{type(exc).__name__}: {exc}"
            log.warning("tts: %s could not be loaded (%s)", name, exc)
            return None

    def engine_for(self, lang: str):
        if lang == "en":
            order = ["mms"]
        elif self.engine_choice in ("sinhalavits", "piper"):
            order = [self.engine_choice]
        else:
            order = ["sinhalavits", "piper"]
        for name in order:
            eng = self._load(name)
            if eng is not None:
                return eng
        raise TtsUnavailable(f"no TTS engine for '{lang}': {self._errors}")

    def status(self) -> dict:
        return {"voice": self.voice, "engine_choice": self.engine_choice, "device": self.device,
                "loaded": {k: v.name for k, v in self._engines.items()}, "errors": dict(self._errors)}

    def cache_key(self, text: str) -> str:
        si_engine = self.engine_for("si").engine
        raw = json.dumps([text, self.voice if si_engine == "sinhalavits" else "", si_engine, terms_version(), SEED])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def speak_sentences(self, sentences: list[Sentence]) -> tuple[np.ndarray, list[SpokenSentence]]:
        parts, spoken = [], []
        pause = np.zeros(int(PAUSE_S * OUT_RATE), np.float32)
        for s in sentences:
            eng = self.engine_for(s.lang)
            t0 = time.perf_counter()
            with self._lock:  # the models are not thread-safe
                wav, dropped = eng.synthesize(s.text)
            ms = (time.perf_counter() - t0) * 1000
            wav = resample(wav, eng.sample_rate, OUT_RATE)
            peak = float(np.abs(wav).max()) if wav.size else 0.0
            if peak > 0.99:
                wav = wav * (0.95 / peak)
            leftovers = list(s.leftovers) + [f"dropped:{c}" for c in dropped]
            if wav.size == 0:
                leftovers.append("empty_audio")
            parts += [wav, pause]
            spoken.append(SpokenSentence(s.lang, eng.engine, s.text, round(wav.size / OUT_RATE, 2), round(ms, 1),
                                         leftovers))
        audio = np.concatenate(parts[:-1]) if parts else np.zeros(0, np.float32)
        return audio, spoken

    def speak(self, text: str) -> TtsResult:
        t0 = time.perf_counter()
        key = self.cache_key(text)
        path, meta_path = CACHE_DIR / f"{key}.wav", CACHE_DIR / f"{key}.json"
        if self.cache and path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return TtsResult(path.read_bytes(), meta["seconds"], True, key, meta["engines"],
                             [SpokenSentence(**s) for s in meta["sentences"]], (time.perf_counter() - t0) * 1000)
        sentences = prepare(text)
        audio, spoken = self.speak_sentences(sentences)
        wav = to_wav_bytes(audio, OUT_RATE)
        engines = sorted({self._engines[s.engine].name for s in spoken})
        seconds = round(audio.size / OUT_RATE, 2)
        if self.cache and spoken:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(wav)
            meta_path.write_text(json.dumps({"seconds": seconds, "engines": engines,
                                             "sentences": [s.__dict__ for s in spoken]}, ensure_ascii=False),
                                 encoding="utf-8")
        return TtsResult(wav, seconds, False, key, engines, spoken, (time.perf_counter() - t0) * 1000)


@lru_cache(maxsize=1)
def get_tts() -> TtsService:
    from shared.config import get_settings

    s = get_settings()
    device = s.stt_device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return TtsService(voice=s.tts_voice, engine=s.tts_engine, device=device)
