"""Load and pre-warm the speech models when the server starts (Phase 7), so the first voice question is fast.

Loading is not enough: the first inference of each model is several times slower than the next (lazy kernels,
memory allocation, tokenizer caches). So after loading, a fixed Sinhala question is spoken by the Sinhala voice,
an English sentence by the English voice, and the Sinhala audio is transcribed by the speech-to-text model.
Nothing is cached from this (the TTS cache is bypassed), and nothing is sent anywhere: every model is local.

It runs in a background thread started by router.on_startup(), so the API and the dashboard are usable while it
works (about 1-2 minutes on a CPU laptop, most of it reading the Whisper weights). /api/health shows the state:
loading -> warming -> ready (or failed, with the error).
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("soilmind.m4.warmup")

# A real farmer question from the chat suggestions (the same words the demo uses), and one English sentence.
WARM_SI = "මගේ කුඹුර අද කොහොමද?"
WARM_EN = "Your field health score is good."

_state: dict = {"state": "not_started", "detail": "", "timings_ms": {}}
_lock = threading.Lock()
_thread: threading.Thread | None = None


def state() -> dict:
    with _lock:
        return {**_state, "timings_ms": dict(_state["timings_ms"])}


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def _timed(name: str, fn):
    t0 = time.perf_counter()
    out = fn()
    with _lock:
        _state["timings_ms"][name] = round((time.perf_counter() - t0) * 1000)
    return out


def run() -> dict:
    """Load + warm STT and TTS (blocking). Returns the final state."""
    from member4_chatbot.voice.audio import SAMPLE_RATE, resample
    from member4_chatbot.voice.stt import get_stt
    from member4_chatbot.voice.tts import OUT_RATE, get_tts
    from member4_chatbot.voice.tts_text import prepare

    try:
        _set(state="loading", detail="loading speech models")
        stt = _timed("load_stt", get_stt)
        tts = get_tts()
        _timed("load_tts_si", lambda: tts.engine_for("si"))
        _timed("load_tts_en", lambda: tts.engine_for("en"))
        _set(state="warming", detail="first inference of each speech model")
        audio_si, _ = _timed("warm_tts_si", lambda: tts.speak_sentences(prepare(WARM_SI)))
        _timed("warm_tts_en", lambda: tts.speak_sentences(prepare(WARM_EN)))
        clip = resample(audio_si, OUT_RATE, SAMPLE_RATE)
        heard = _timed("warm_stt_si", lambda: stt.transcribe_array(clip, "si"))
        engines = ", ".join(sorted(tts.status()["loaded"].values()))
        _set(state="ready", detail=f"STT {stt.model_id}; TTS {engines}; warm-up heard '{heard}'")
        log.info("m4: speech models warm (%s)", state()["timings_ms"])
    except Exception as exc:  # noqa: BLE001 - voice failing must not take the chat down
        _set(state="failed", detail=f"{type(exc).__name__}: {exc}")
        log.warning("m4: speech warm-up failed (%s); /stt and /tts will try again on first use", exc)
    return state()


def start_background() -> None:
    """Start run() once in a daemon thread (repeated calls do nothing)."""
    global _thread
    with _lock:
        if _thread is not None:
            return
        _state["state"] = "queued"
        _thread = threading.Thread(target=run, name="m4-voice-warmup", daemon=True)
    _thread.start()


def wait(timeout_s: float | None = None) -> dict:
    if _thread is not None:
        _thread.join(timeout_s)
    return state()
