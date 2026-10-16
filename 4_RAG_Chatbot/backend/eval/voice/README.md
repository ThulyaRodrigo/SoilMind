# Voice evaluation and the team's recordings (Member 4, Phase 6b)

## What is here
| File | What |
|---|---|
| `record_prompts.csv` | 249 Sinhala farming phrases to record, each with a fixed `split` (`finetune` 70% / `test` 30%). Built by `python -m member4_chatbot.pipeline.voice_prompts` from `sources/term_phrases.yaml`, the 30 Sinhala questions of `../qa_testset.csv` and Sinhala training questions. **Not human-checked yet.** |
| `sources/term_phrases.yaml` | the 100 short term-focused phrases (urea, TSP, MOP, dolomite, nitrogen, kanna names, numbers with units, ...), drafted by the AI coding assistant |
| `recorder.html` | the recording page (works offline in the browser, saves ZIP batches) |
| `recordings/` | `incoming/` (put the ZIPs here), `wav/` + `manifest.csv` (made by the import step), `colab_pack.zip` |
| `openslr30_test.csv` | 200 utterances of OpenSLR-30 (real read Sinhala, 12 speakers), general-Sinhala test set |
| `synthetic_test.csv` | the prompts spoken by our TTS voices (stand-in farming test set until the team records) |
| `results/` | per-clip transcripts, `metrics.json`, TTS sentence results, dictionary candidates |
| `listening/` | 40 WAV files + `listening_scores.xlsx` for the 1-5 listening score |

## How the team records (about 1 hour per speaker)
1. **Check the wording first.** A Sinhala speaker reads `record_prompts.csv` and fixes unnatural phrases in
   `sources/term_phrases.yaml` (or the other sources), then rebuilds:
   `python -m member4_chatbot.pipeline.voice_prompts`. Set `checked_by_human` in your notes.
2. **Start the page** (any computer with a microphone; a phone works too if it is on the same Wi-Fi and you use
   the laptop's address):
   ```bash
   cd backend/member4-chatbot/eval/voice
   python -m http.server 8765
   ```
   open http://localhost:8765/recorder.html, enter a speaker code (S01, S02, ... no names), tick consent.
3. **Record**: space to start / stop, play it back, → to save and go to the next phrase. If you say a phrase
   differently from the text, type what you said in "I said instead" (that becomes the reference text).
4. Press **Download batch** every ~50 phrases. Put every ZIP in `recordings/incoming/`.
5. **Import**: `python -m member4_chatbot.pipeline.voice_recordings` (converts to 16 kHz WAV, builds
   `recordings/manifest.csv`).

Aim for **at least 3 speakers** (women and men, different ages), all 249 phrases each: about 750 clips. A few
recorded outside (place = field) make the test more realistic. Notebook 06 needs at least 150 `finetune` and 40
`test` recordings to start.

## Then
```bash
python -m member4_chatbot.pipeline.voice_eval stt small farm        # base model on the real recordings
python -m member4_chatbot.pipeline.voice_eval candidates             # misheard words, finetune split only
#   edit ../../data/voice/stt_corrections.yaml (rules at its top), then
python -m member4_chatbot.pipeline.voice_eval rescore
python -m member4_chatbot.pipeline.voice_recordings --pack           # for Colab
#   run notebooks/06_sinhala_stt_finetune.ipynb on a Colab T4, copy artifacts/stt/finetuned/ back,
#   set STT_MODEL=finetuned in .env if it wins, and run notebook 07 again
```

## Listening score
Open `listening/listening_scores.xlsx`, hide column C (which engine), play each file and fill in the 1-5 score
and whether the numbers were said correctly. Notebook 07 reads the sheet.
