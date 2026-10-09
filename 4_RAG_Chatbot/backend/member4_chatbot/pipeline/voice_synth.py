"""Phase 6b step 3: the synthetic farming test set. Every recording prompt (eval/voice/record_prompts.csv) is
spoken by 2 of our 5 TTS voices (SinhalaVITS F1, M1, F2, M2 and Piper), as 16 kHz WAV in data/voice/synthetic/,
with the manifest eval/voice/synthetic_test.csv.

    python -m member4_chatbot.pipeline.voice_synth

This is SYNTHETIC speech: clean, read by a machine. It tests whether the speech models know the farming words and
numbers; it says nothing about real farmers' voices, accents or field noise. The team's own recordings
(pipeline.voice_recordings) replace it for the final numbers. About 15 minutes on CPU.
"""

from member4_chatbot.voice.evaluate import synthesize_testset

if __name__ == "__main__":
    df = synthesize_testset()
    print(f"{len(df)} clips, {df.duration_s.sum() / 60:.1f} min; voices: {df.voice.value_counts().to_dict()}")
    left = df[df.tts_leftovers != ""]
    print(f"clips with TTS leftovers: {len(left)}")
