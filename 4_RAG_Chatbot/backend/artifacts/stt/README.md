# artifacts/stt/

Our fine-tuned Sinhala speech-to-text model goes here as `finetuned/` (Hugging Face Whisper format, with
`metrics.json`), saved by `notebooks/06_sinhala_stt_finetune.ipynb`.

**Empty for now:** fine-tuning needs the team's recordings of `eval/voice/record_prompts.csv` (see
`eval/voice/README.md`). Until then the app uses the pretrained Whisper-Sinhala model named by `STT_MODEL` in `.env`
(default `small` = `seniruk/whisper-small-si`), downloaded to the Hugging Face cache on first use.

After notebook 06 has run and its last cell says the fine-tuned model beats the base model on the held-out farming
phrases, set `STT_MODEL=finetuned` and restart the API.
