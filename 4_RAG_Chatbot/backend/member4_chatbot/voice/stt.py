"""Speech-to-text: a farmer's recorded question -> text, with pretrained Whisper models fine-tuned for Sinhala.

Models (choose with STT_MODEL in .env; all run locally through `transformers`):

| STT_MODEL | Model | Size | Use |
|---|---|---|---|
| `small` (default) | seniruk/whisper-small-si | 242 M parameters, 0.97 GB | CPU laptops |
| `small-sinhaspeech` | SinhaSpeech/whisper-small-sinhala-run11 | 242 M, 0.97 GB | the other small model, compared in notebook 07 |
| `large` | kasunw/whisper-large-v3-sinhala | 1.55 B, 6.2 GB | GPU only; NOT recommended: less accurate and slower than `small` in notebook 07 |
| `finetuned` | artifacts/stt/finetuned (ours, after the team records phrases; notebook 06) | as its base | |
| any other value | a Hugging Face id or local folder of a Whisper model | | |

The brief names `seniruk/whisper-medium-si-v2`; that repository is not public on Hugging Face (checked
2026-10-19), so the small and large models above are used. The default small model was chosen on the
general-Sinhala test set (OpenSLR-30, notebook 07): CER 4.5% vs 5.0%, WER 24.5% vs 32.3%.

Steps: audio bytes -> ffmpeg (16 kHz mono, silence trimmed, at most 30 s) -> Whisper (greedy decoding, language
forced) -> NFC -> farming-term dictionary (corrections.py).
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from member4_chatbot.paths import STT_DIR
from member4_chatbot.voice import audio as A
from member4_chatbot.voice.corrections import correct

log = logging.getLogger("soilmind.m4.stt")

MODELS = {
    "small": "seniruk/whisper-small-si",
    "small-sinhaspeech": "SinhaSpeech/whisper-small-sinhala-run11",
    "large": "kasunw/whisper-large-v3-sinhala",
    "finetuned": str(STT_DIR / "finetuned"),
}
LANGS = {"si": "sinhala", "en": "english"}


@dataclass
class Transcript:
    text: str
    raw_text: str
    language: str
    duration: float
    model: str
    corrections: list[dict] = field(default_factory=list)
    audio: dict = field(default_factory=dict)
    latency_ms: float = 0.0


def resolve_model(name: str) -> str:
    return MODELS.get(name, name)


class WhisperStt:
    def __init__(self, model: str = "small", device: str = "cpu"):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.model_id = resolve_model(model)
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        t0 = time.perf_counter()
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id, dtype=dtype).to(device).eval()
        if not hasattr(self.model.generation_config, "lang_to_id"):
            # some fine-tunes (e.g. kasunw/whisper-large-v3-sinhala) ship no generation_config.json; the language
            # tokens are those of the OpenAI base model of the same size
            from transformers import GenerationConfig

            base = {1280: "openai/whisper-large-v3" if self.model.config.num_mel_bins == 128 else "openai/whisper-large-v2",
                    1024: "openai/whisper-medium", 768: "openai/whisper-small", 512: "openai/whisper-base",
                    384: "openai/whisper-tiny"}[self.model.config.d_model]
            self.model.generation_config = GenerationConfig.from_pretrained(base)
        self.dtype = dtype
        self._lock = threading.Lock()
        log.info("stt: %s loaded on %s in %.1f s", self.model_id, device, time.perf_counter() - t0)

    def transcribe_array(self, audio: np.ndarray, language: str = "si", num_beams: int = 1) -> str:
        import torch

        inputs = self.processor(audio, sampling_rate=A.SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
        feats = inputs.input_features.to(self.device, dtype=self.dtype)
        with self._lock, torch.inference_mode():
            ids = self.model.generate(feats, attention_mask=inputs.attention_mask.to(self.device),
                                      language=LANGS.get(language, language), task="transcribe",
                                      num_beams=num_beams, max_new_tokens=220)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return unicodedata.normalize("NFC", text).strip()

    def transcribe(self, data: bytes, language: str = "si", use_dictionary: bool = True) -> Transcript:
        t0 = time.perf_counter()
        audio, info = A.load_audio(data)
        raw = self.transcribe_array(audio, language)
        text, changes = correct(raw, context={"model": self.model_id}) if use_dictionary else (raw, [])
        return Transcript(text=text, raw_text=raw, language=language, duration=info["seconds"],
                          model=self.model_id, corrections=changes, audio=info,
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1))


_build_lock = threading.Lock()


def get_stt() -> WhisperStt:
    """The configured model, loaded once. Locked: the startup warm-up thread and a request may ask together."""
    with _build_lock:
        return _build_stt()


@lru_cache(maxsize=1)
def _build_stt() -> WhisperStt:
    from shared.config import get_settings

    s = get_settings()
    device = s.stt_device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperStt(s.stt_model, device)


get_stt.cache_info = _build_stt.cache_info  # type: ignore[attr-defined]  (used by /voice-info and the tests)
get_stt.cache_clear = _build_stt.cache_clear  # type: ignore[attr-defined]
