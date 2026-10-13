"""Audio in and out: browser recordings (webm / ogg / wav) to 16 kHz mono float32, and float arrays to WAV bytes.

ffmpeg does the decoding. The `ffmpeg` on PATH is used if there is one; otherwise the static ffmpeg build that
ships with the `imageio-ffmpeg` pip package, so nothing has to be installed by hand.

The conversion trims silence at both ends (ffmpeg `silenceremove`, -40 dB) and caps the result at 30 s, the
length of one Whisper window. If trimming leaves almost nothing (a very quiet recording), the untrimmed audio is
kept rather than sending silence to the model.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from functools import lru_cache

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
MAX_SECONDS = 30.0
SILENCE_DB = -40
MIN_KEPT_SECONDS = 0.3  # below this after trimming, fall back to the untrimmed audio


class AudioError(ValueError):
    """The upload is not audio ffmpeg can read, or it is empty."""


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`") from exc


def _run_ffmpeg(data: bytes, filters: str | None) -> np.ndarray:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-vn", "-ac", "1",
           "-ar", str(SAMPLE_RATE)]
    if filters:
        cmd += ["-af", filters]
    cmd += ["-t", str(MAX_SECONDS), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, input=data, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg could not read the audio: {proc.stderr.decode(errors='replace').strip()[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def trim_filter(db: int = SILENCE_DB) -> str:
    """Trim leading silence, reverse, trim again (= trailing silence), reverse back."""
    one_end = f"silenceremove=start_periods=1:start_duration=0.05:start_threshold={db}dB:detection=peak"
    return f"{one_end},areverse,{one_end},areverse"


def load_audio(data: bytes, trim: bool = True) -> tuple[np.ndarray, dict]:
    """Any ffmpeg-readable bytes -> (16 kHz mono float32 at most 30 s, info)."""
    if not data:
        raise AudioError("empty upload")
    raw = _run_ffmpeg(data, None)
    if raw.size == 0:
        raise AudioError("no audio samples in the upload")
    info = {"input_seconds": round(raw.size / SAMPLE_RATE, 2), "trimmed": False}
    audio = raw
    if trim:
        trimmed = _run_ffmpeg(data, trim_filter())
        if trimmed.size >= MIN_KEPT_SECONDS * SAMPLE_RATE:
            audio, info["trimmed"] = trimmed, True
    info["seconds"] = round(audio.size / SAMPLE_RATE, 2)
    info["capped"] = info["input_seconds"] >= MAX_SECONDS - 0.01
    return audio, info


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    import soxr

    return soxr.resample(audio.astype(np.float32), sr_in, sr_out)
