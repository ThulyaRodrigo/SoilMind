"""The fake-microphone audio for the Playwright voice test (frontend/e2e/voice.spec.ts).

    python -m member4_chatbot.pipeline.voice_e2e_fixture

A Sinhala farming question ("how is my field today?") spoken by our Piper Sinhala voice, with 0.6 s of silence
before and after, written as 16-bit 48 kHz mono WAV (what Chromium's fake audio capture expects) to
frontend/e2e/fixtures/si_question.wav. Piper is used, not SinhalaVITS, so the STT under test hears a different
voice from the one that reads answers aloud.
"""

import numpy as np
import soundfile as sf

from member4_chatbot.paths import MEMBER_DIR
from member4_chatbot.voice.audio import resample
from member4_chatbot.voice.tts import TtsService
from member4_chatbot.voice.tts_text import prepare

QUESTION = "මගේ කුඹුර අද කොහොමද?"
OUT = MEMBER_DIR.parent.parent / "frontend" / "e2e" / "fixtures" / "si_question.wav"

if __name__ == "__main__":
    svc = TtsService(engine="piper", cache=False)
    audio, _ = svc.speak_sentences(prepare(QUESTION))
    audio = resample(audio, 22050, 48000)
    pad = np.zeros(int(0.6 * 48000), np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(OUT, np.concatenate([pad, audio, pad]), 48000, subtype="PCM_16")
    print(f"{OUT} ({(audio.size + 2 * pad.size) / 48000:.1f} s)")
