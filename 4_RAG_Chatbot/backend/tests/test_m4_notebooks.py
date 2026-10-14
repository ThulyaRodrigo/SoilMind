"""Run Member 4's notebooks top to bottom with RETRAIN = False; any cell error fails the test.

Notebook 06 (STT fine-tuning) runs with RUN_FINETUNE = False: it only checks the team's recordings.
Notebook 01 ends by asserting the intent model beats the TF-IDF baseline, and notebook 02 by asserting the
fine-tuned retriever beats BM25 and the pretrained model on Sinhala hit@3, so those checks run here too.
The saved notebooks are not modified (executed in memory).
"""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from member4_chatbot.paths import INTENT_DIR, NOTEBOOKS_DIR, RETRIEVER_DIR

NOTEBOOKS = sorted(NOTEBOOKS_DIR.glob("[0-9][0-9]_*.ipynb"))

pytestmark = pytest.mark.skipif(
    not (INTENT_DIR / "metrics.json").exists() or not (RETRIEVER_DIR / "metrics.json").exists()
    or not (NOTEBOOKS_DIR / "data" / "intents.csv").exists(),
    reason="models not trained / notebook data not exported")


def test_all_notebooks_present():
    assert [p.name for p in NOTEBOOKS] == ["00_dataset_exploration.ipynb", "01_intent_classifier.ipynb",
                                          "02_retriever.ipynb", "03_extractive_answers.ipynb",
                                          "04_end_to_end_evaluation.ipynb", "05_sentence_reranker.ipynb",
                                          "06_sinhala_stt_finetune.ipynb", "07_voice_evaluation.ipynb"]


@pytest.mark.slow  # executes the notebook; run with `pytest -m slow`
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs_without_errors(path: Path):
    nb = nbformat.read(path, as_version=4)
    flags = [c for c in nb.cells if c.cell_type == "code" and c.source.lstrip().startswith(("RETRAIN", "RUN_EVAL", "RUN_FINETUNE"))]
    if path.stem[:2] in ("01", "02", "04", "05", "06", "07"):
        assert flags, "notebook has no RETRAIN / RUN_EVAL / RUN_FINETUNE flag cell"
    for c in flags:  # evaluate the saved models / show the saved evaluation; never retrain in a test
        c.source = (c.source.replace("RETRAIN = True", "RETRAIN = False").replace("RUN_EVAL = True", "RUN_EVAL = False")
                    .replace("RUN_FINETUNE = True", "RUN_FINETUNE = False"))
        assert "= False" in c.source
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS_DIR)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors
