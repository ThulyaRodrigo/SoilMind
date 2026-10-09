"""Phase 6b, after the team records: import the recordings made with eval/voice/recorder.html.

    python -m member4_chatbot.pipeline.voice_recordings          # import ZIPs from eval/voice/recordings/incoming/
    python -m member4_chatbot.pipeline.voice_recordings --pack   # + colab_pack.zip for notebook 06 on Colab

Each ZIP holds <speaker>_<prompt>.webm files and a manifest.csv. Every recording is converted to 16 kHz mono WAV
(silence trimmed, at most 30 s; the same conversion as the live /stt route) in eval/voice/recordings/wav/, and
eval/voice/recordings/manifest.csv lists them with the prompt text, category and split from record_prompts.csv.
When a speaker wrote "I said instead", that text is the reference. Re-importing the same file replaces it.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import soundfile as sf

from member4_chatbot.paths import MEMBER_DIR, VOICE_DATA_DIR, VOICE_EVAL_DIR
from member4_chatbot.voice.audio import SAMPLE_RATE, AudioError, load_audio

REC_DIR = VOICE_EVAL_DIR / "recordings"
INCOMING, WAV_DIR, MANIFEST = REC_DIR / "incoming", REC_DIR / "wav", REC_DIR / "manifest.csv"
COLUMNS = ["clip_id", "speaker", "prompt_id", "text", "written_text", "category", "split", "file", "duration_s",
           "gender", "age_group", "place", "recorded_at"]


def import_zip(path: Path, prompts: pd.DataFrame) -> list[dict]:
    rows = []
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if "manifest.csv" not in names:
            print(f"  {path.name}: no manifest.csv, skipped")
            return rows
        meta = pd.read_csv(io.BytesIO(z.read("manifest.csv")), encoding="utf-8-sig", keep_default_na=False, dtype=str)
        for m in meta.itertuples():
            if m.file not in names or m.prompt_id not in prompts.index or "/" in m.file or "\\" in m.file:
                print(f"  {path.name}: {m.file} skipped (missing file or unknown prompt)")
                continue
            try:
                audio, info = load_audio(z.read(m.file))
            except AudioError as exc:
                print(f"  {path.name}: {m.file} skipped ({exc})")
                continue
            p = prompts.loc[m.prompt_id]
            clip = f"{m.speaker}_{m.prompt_id}"
            WAV_DIR.mkdir(parents=True, exist_ok=True)
            sf.write(WAV_DIR / f"{clip}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
            rows.append({"clip_id": clip, "speaker": m.speaker, "prompt_id": m.prompt_id,
                         "text": m.said_instead.strip() or p.text, "written_text": p.text, "category": p.category,
                         "split": p.split, "file": f"eval/voice/recordings/wav/{clip}.wav",
                         "duration_s": info["seconds"], "gender": m.gender, "age_group": m.age_group,
                         "place": m.place, "recorded_at": m.recorded_at})
    return rows


def pack() -> Path:
    """Everything notebook 06 needs on Colab: the recordings, the prompts, the term dictionary and the voice code."""
    out = REC_DIR / "colab_pack.zip"
    files = [MANIFEST, VOICE_EVAL_DIR / "record_prompts.csv", VOICE_DATA_DIR / "stt_corrections.yaml",
             *sorted(WAV_DIR.glob("*.wav")), *sorted((MEMBER_DIR / "member4_chatbot" / "voice").glob("*.py")),
             MEMBER_DIR / "member4_chatbot" / "__init__.py", MEMBER_DIR / "member4_chatbot" / "paths.py",
             MEMBER_DIR / "member4_chatbot" / "lang.py", MEMBER_DIR / "member4_chatbot" / "ingest" / "__init__.py",
             MEMBER_DIR / "member4_chatbot" / "ingest" / "sinhala.py"]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.exists():
                z.write(f, f.relative_to(MEMBER_DIR).as_posix())
    return out


def main(argv: list[str]) -> None:
    prompts = pd.read_csv(VOICE_EVAL_DIR / "record_prompts.csv", keep_default_na=False).set_index("prompt_id")
    old = pd.read_csv(MANIFEST, keep_default_na=False, dtype=str) if MANIFEST.exists() else pd.DataFrame(columns=COLUMNS)
    INCOMING.mkdir(parents=True, exist_ok=True)
    new = []
    for z in sorted(INCOMING.glob("*.zip")):
        rows = import_zip(z, prompts)
        print(f"{z.name}: {len(rows)} recordings")
        new += rows
    df = pd.concat([old, pd.DataFrame(new, columns=COLUMNS)]).drop_duplicates("clip_id", keep="last")
    df = df.sort_values("clip_id")
    df.to_csv(MANIFEST, index=False, encoding="utf-8")
    print(f"{len(df)} recordings from {df.speaker.nunique()} speakers -> {MANIFEST}")
    if len(df):
        print(df.groupby("split").size().to_string())
    if "--pack" in argv:
        print(f"colab pack: {pack()}")


if __name__ == "__main__":
    main(sys.argv[1:])
