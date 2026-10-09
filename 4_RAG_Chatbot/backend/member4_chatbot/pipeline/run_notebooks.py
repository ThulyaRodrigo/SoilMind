"""Step 5: official training. Runs notebooks 00, 01, 02 in order with RETRAIN = True and saves them with outputs.

    python -m member4_chatbot.pipeline.run_notebooks            # all three
    python -m member4_chatbot.pipeline.run_notebooks 02         # only the retriever

Exports the notebook data first. The saved notebooks keep RETRAIN = True and their training outputs, so they
show how the models in artifacts/ were made. CPU time on the build machine: about 25-40 minutes in total.
"""

import sys
import time

import nbformat
from nbclient import NotebookClient

from member4_chatbot.paths import NOTEBOOKS_DIR
from member4_chatbot.pipeline import export_notebook_data


def main(only: list[str]) -> None:
    export_notebook_data.main()
    for path in sorted(NOTEBOOKS_DIR.glob("[0-9][0-9]_*.ipynb")):
        if only and path.stem[:2] not in only:
            continue
        t0 = time.time()
        print(f"running {path.name} ...", flush=True)
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS_DIR)}}).execute()
        nbformat.write(nb, path)
        print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
