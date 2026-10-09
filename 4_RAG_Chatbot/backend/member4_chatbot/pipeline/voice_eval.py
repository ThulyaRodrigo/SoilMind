"""Phase 6b step 4: evaluate speech-to-text and text-to-speech. Results go to eval/voice/results/ and are shown in
notebook 07_voice_evaluation.ipynb.

    python -m member4_chatbot.pipeline.voice_eval stt small openslr30            # one model on one test set (CPU)
    python -m member4_chatbot.pipeline.voice_eval stt large synthetic --device cuda
    python -m member4_chatbot.pipeline.voice_eval stt small farm                 # after the team records
    python -m member4_chatbot.pipeline.voice_eval stt small openslr30 --limit 40 # a sample (e.g. slow CPU runs)
    python -m member4_chatbot.pipeline.voice_eval candidates                     # misheard words, finetune split only
    python -m member4_chatbot.pipeline.voice_eval rescore                        # re-apply the term dictionary
    python -m member4_chatbot.pipeline.voice_eval tts                            # TTS failures, latency, round trip
    python -m member4_chatbot.pipeline.voice_eval listening                      # audio + sheet for the 1-5 scores

`--hardware` stores a note of the machine with the metrics (default: the CPU / GPU name).
"""

from __future__ import annotations

import argparse
import platform

from member4_chatbot.voice import evaluate as E


def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def hardware(device: str) -> str:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        return f"GPU {torch.cuda.get_device_name(0)} (CPU {cpu_name()}), torch {torch.__version__}"
    return f"CPU {cpu_name()}, {torch.get_num_threads()} threads, torch {torch.__version__}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stt", "candidates", "rescore", "tts", "listening"])
    ap.add_argument("model", nargs="?", default="small")
    ap.add_argument("set", nargs="?", default="openslr30", choices=list(E.SETS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hardware", default=None)
    a = ap.parse_args()
    if a.what == "stt":
        df = E.run_stt(a.model, a.set, a.device, a.limit, hardware=a.hardware or hardware(a.device))
        print(E.read_metrics()["stt"][f"{a.model}|{a.set}|{a.device}{f'_n{a.limit}' if a.limit else ''}"])
        print(f"{len(df)} clips")
    elif a.what == "candidates":
        files = sorted(E.RESULTS_DIR.glob("stt_*_synthetic_*.csv")) + sorted(E.RESULTS_DIR.glob("stt_*_farm_*.csv"))
        out = E.dictionary_candidates(files)
        out.to_csv(E.RESULTS_DIR / "dictionary_candidates.csv", index=False, encoding="utf-8")
        print(f"{len(out)} candidate pairs from {len(files)} result files -> dictionary_candidates.csv")
    elif a.what == "rescore":
        for key in list(E.read_metrics().get("stt", {})):
            model, set_name, device = key.split("|")
            if "_n" in device:
                continue
            if (E.RESULTS_DIR / f"stt_{model}_{set_name}_{device}.csv").exists():
                s = E.rescore_dictionary(model, set_name, device)
                print(key, s["test_split"] or s["all"])
    elif a.what == "tts":
        df = E.run_tts_eval(hardware=a.hardware or hardware("cpu"))
        print(E.read_metrics()["tts"]["cpu"])
        print(f"{len(df)} sentences")
    else:
        print(E.listening_sheet())


if __name__ == "__main__":
    main()
