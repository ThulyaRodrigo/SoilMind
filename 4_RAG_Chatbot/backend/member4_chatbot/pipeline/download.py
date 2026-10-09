"""Step 1: download the knowledge-base documents.  python -m member4_chatbot.pipeline.download [--force]"""

import sys

from member4_chatbot.collect.download import main

if __name__ == "__main__":
    main("--force" in sys.argv)
