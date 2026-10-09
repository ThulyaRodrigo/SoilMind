"""Phase 6b step 1: download the general-Sinhala speech test set (OpenSLR-30) and take a fixed sample.

    python -m member4_chatbot.pipeline.voice_data            # 200 utterances (default)
    python -m member4_chatbot.pipeline.voice_data 400        # a bigger sample

OpenSLR-30 is Google's Sinhala multi-speaker read-speech corpus (CC BY-SA 4.0, https://www.openslr.org/30/).
It is NOT the OpenSLR-52 ASR corpus the Whisper-Sinhala models were trained on, so it is a fair held-out test of
general Sinhala. The sample is drawn with a fixed seed, spread evenly over speakers, and converted to 16 kHz mono
WAV in data/voice/openslr30/ with a manifest eval/voice/openslr30_test.csv. The 700 MB archive is kept in
data/voice/external/ (delete it afterwards if disk space matters; re-running downloads it again).
"""

from __future__ import annotations

import io
import re
import sys
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import soundfile as sf
import soxr

from member4_chatbot.paths import VOICE_DATA_DIR, VOICE_EVAL_DIR

SEED = 20261019
BASE = "https://www.openslr.org/resources/30/"
EXTERNAL = VOICE_DATA_DIR / "external"
OUT_DIR = VOICE_DATA_DIR / "openslr30"
MANIFEST = VOICE_EVAL_DIR / "openslr30_test.csv"


def _download(name: str) -> None:
    dest = EXTERNAL / name
    if dest.exists() and dest.stat().st_size > 0:
        return
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    print(f"downloading {BASE + name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(BASE + name, tmp)
    tmp.replace(dest)


def read_lines(path) -> pd.DataFrame:
    """si_lk.lines.txt is in Festival format: ( sin_2241_0329430812 "text" )."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r'\(\s*(\S+)\s+"(.*)"\s*\)\s*$', line.strip())
        if m:
            utt = m.group(1)
            rows.append({"utt_id": utt, "speaker": "_".join(utt.split("_")[:2]), "text": m.group(2).strip()})
    return pd.DataFrame(rows)


def sample(lines: pd.DataFrame, n: int) -> pd.DataFrame:
    """n utterances spread evenly over speakers (round robin over a seeded shuffle)."""
    rng = np.random.default_rng(SEED)
    by_spk = {s: g.sample(frac=1, random_state=int(rng.integers(1 << 31))).to_dict("records")
              for s, g in lines.groupby("speaker", sort=True)}
    picked: list[dict] = []
    while len(picked) < n and any(by_spk.values()):
        for s in sorted(by_spk):
            if by_spk[s] and len(picked) < n:
                picked.append(by_spk[s].pop())
    return pd.DataFrame(picked)


def main(n: int = 200) -> None:
    for name in ("si_lk.lines.txt", "LICENSE.txt", "si_lk.tar.gz"):
        _download(name)
    lines = read_lines(EXTERNAL / "si_lk.lines.txt")
    picked = sample(lines, n).set_index("utt_id")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    durations = {}
    with tarfile.open(EXTERNAL / "si_lk.tar.gz", "r:gz") as tar:
        for member in tar:  # read the chosen members only; nothing is extracted to disk as is
            stem = member.name.rsplit("/", 1)[-1].removesuffix(".wav")
            if not member.isfile() or stem not in picked.index:
                continue
            audio, sr = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32", always_2d=True)
            audio = soxr.resample(audio.mean(axis=1), sr, 16000)
            sf.write(OUT_DIR / f"{stem}.wav", audio, 16000, subtype="PCM_16")
            durations[stem] = round(len(audio) / 16000, 2)
    picked = picked.loc[list(durations)].reset_index()
    picked["file"] = [f"data/voice/openslr30/{u}.wav" for u in picked.utt_id]
    picked["duration_s"] = [durations[u] for u in picked.utt_id]
    VOICE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    picked.sort_values("utt_id").to_csv(MANIFEST, index=False, encoding="utf-8")
    print(f"{len(picked)} utterances from {picked.speaker.nunique()} speakers "
          f"({picked.duration_s.sum() / 60:.1f} min) -> {MANIFEST}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
