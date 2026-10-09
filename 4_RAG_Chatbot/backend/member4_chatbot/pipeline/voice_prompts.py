"""Phase 6b step 2: the recording script for our STT fine-tuning, eval/voice/record_prompts.csv.

    python -m member4_chatbot.pipeline.voice_prompts

About 250 short Sinhala farming phrases and questions for the team to read aloud (2-3 speakers each, see
eval/voice/README.md). Sources, all Sinhala script:
- `qa_testset`: the 30 Sinhala questions of the chatbot evaluation set (eval/qa_testset.csv)
- `term_phrase`: short phrases about urea, TSP, MOP, dolomite, nitrogen, seasons (kanna), varieties, pests,
  water, numbers with units (eval/voice/sources/term_phrases.yaml)
- `training_ai_drafted`, `training_template`: Sinhala questions of the chatbot training data that contain a
  farming term (machine-translated rows are left out: their Sinhala is often wrong, see Phase 5)

NONE of the text is human-checked yet (checked_by_human = no): it was written by the AI coding assistant or
generated from templates. A Sinhala speaker should fix the wording before the team records.

Each phrase gets a split, fixed by seed and stratified by category:
- `finetune` (about 70%): recordings used to fine-tune Whisper (notebook 06)
- `test` (about 30%): held out; WER/CER before vs after fine-tuning and with vs without the term dictionary
  are measured on these only
Recordings of one phrase never appear in both.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import yaml

from member4_chatbot.paths import EVAL_DIR, TRAINING_DIR, VOICE_EVAL_DIR

SEED = 20261019
OUT = VOICE_EVAL_DIR / "record_prompts.csv"
TERM_YAML = VOICE_EVAL_DIR / "sources" / "term_phrases.yaml"
TEST_SHARE = 0.3
N_TRAINING_AI, N_TRAINING_TEMPLATE = 75, 45
FARM_WORDS = ["යූරියා", "ටී.එස්.පී", "එම්.ඕ.පී", "ඩොලමයිට්", "හුණු", "සින්ක්", "පොහොර", "නයිට්‍රජන්",
              "පොස්පරස්", "පොටෑසියම්", "යල", "මහ", "කන්න", "කුඹුර", "වී", "ගොයම", "තවාන", "පැල", "වල්",
              "කීඩෑවා", "මැස්සා", "පාළුව", "අංගමාරය", "වතුර", "ජලය", "අස්වැන්න", "පස", "බිත්තර"]


def _category_of_intent(intent: str) -> str:
    return {"fertilizer_amount": "fertilizer", "fertilizer_timing": "fertilizer", "liming_ph": "soil",
            "deficiency_symptom": "fertilizer", "crop_choice": "varieties", "crop_establishment": "establishment",
            "pest_disease": "pests_diseases", "weeds": "weeds", "water_management": "water",
            "my_field_status": "my_field", "greeting_help": "general", "out_of_scope": "general"}.get(intent, "general")


def _speakable(text: str) -> bool:
    """Sinhala script, no English words other than variety codes, short enough to read in one breath."""
    latin = re.findall(r"[A-Za-z]+", text)
    return (len(text) <= 110 and all(w in {"Bg", "Bw", "Ld", "At", "H", "pH"} for w in latin)
            and any(w in text for w in FARM_WORDS))


def build() -> pd.DataFrame:
    rows = []
    qa = pd.read_csv(EVAL_DIR / "qa_testset.csv", keep_default_na=False)
    for r in qa[qa.lang == "si"].itertuples():
        rows.append({"text": r.question, "gloss_en": "", "category": _category_of_intent(r.expected_intent),
                     "source": "qa_testset", "source_id": r.id})
    for category, items in yaml.safe_load(TERM_YAML.read_text(encoding="utf-8")).items():
        for i, item in enumerate(items):
            text, gloss = (x.strip() for x in item.split("|", 1))
            rows.append({"text": text, "gloss_en": gloss, "category": category, "source": "term_phrase",
                         "source_id": f"{category}-{i + 1:02d}"})
    train = pd.read_csv(TRAINING_DIR / "intents.csv", keep_default_na=False)
    train = train[(train.lang == "si") & train.text.map(_speakable)].drop_duplicates("text")
    for source, n in (("ai_drafted", N_TRAINING_AI), ("auto_template", N_TRAINING_TEMPLATE)):
        pick = train[train.source == source].sample(n=n, random_state=SEED).sort_values("id")
        for r in pick.itertuples():
            rows.append({"text": r.text, "gloss_en": "", "category": _category_of_intent(r.intent),
                         "source": f"training_{'template' if source == 'auto_template' else source}",
                         "source_id": r.id})
    df = pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)
    df.insert(0, "prompt_id", [f"P{i + 1:03d}" for i in range(len(df))])

    rng = np.random.default_rng(SEED)
    df["split"] = "finetune"
    for _, idx in df.groupby("category").groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        df.loc[idx[: max(1, round(len(idx) * TEST_SHARE))], "split"] = "test"
    df["checked_by_human"] = "no"
    return df


def main() -> None:
    df = build()
    VOICE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"{len(df)} prompts -> {OUT}")
    print(df.groupby(["category", "split"]).size().unstack(fill_value=0).to_string())
    print(df.source.value_counts().to_string())


if __name__ == "__main__":
    main()
