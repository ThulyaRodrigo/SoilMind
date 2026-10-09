"""Step 4: copy the CSV files the notebooks need into notebooks/data/.

    python -m member4_chatbot.pipeline.export_notebook_data

The notebooks read only these CSVs (like Member 2's), so they also run in Google Colab with the `data/` folder
uploaded beside them:
- intents.csv, qa_pairs.csv, out_of_scope.csv (data/training/)
- passages.csv (data/processed/)
- dataset_stats.json and extraction_log.csv, for notebook 00
"""

import shutil

from member4_chatbot.paths import NOTEBOOKS_DIR, PROCESSED_DIR, TRAINING_DIR

FILES = [TRAINING_DIR / "intents.csv", TRAINING_DIR / "qa_pairs.csv", TRAINING_DIR / "out_of_scope.csv",
         TRAINING_DIR / "dataset_stats.json", PROCESSED_DIR / "passages.csv", PROCESSED_DIR / "extraction_log.csv"]


def main() -> None:
    out = NOTEBOOKS_DIR / "data"
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy2(f, out / f.name)
        print(f"copied {f.name}")


if __name__ == "__main__":
    main()
