"""Post-correction of transcripts: farming words the speech model commonly mishears -> the right spelling.

The dictionary is data/voice/stt_corrections.yaml. Each entry says what the model wrote (`heard`), the right word
(`write`), and the evidence (which model, which recordings). Only whole words are replaced, and every replacement
is logged (logger `soilmind.m4.stt` and data/voice/logs/stt_corrections.jsonl) and returned to the caller, so the
farmer sees the corrected text and an officer can audit what was changed.

Python's `\\b` does not work inside Sinhala words (vowel signs are not "word" characters), so word edges are
defined here as "not a Sinhala letter, Latin letter, digit or ZWJ".
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from functools import lru_cache

import yaml

from member4_chatbot.paths import VOICE_DATA_DIR

log = logging.getLogger("soilmind.m4.stt")

CORRECTIONS_YAML = VOICE_DATA_DIR / "stt_corrections.yaml"
LOG_PATH = VOICE_DATA_DIR / "logs" / "stt_corrections.jsonl"
_WORD = r"\u0D80-\u0DFF\u200d\w"
_log_lock = threading.Lock()


@lru_cache(maxsize=4)
def load_rules(path: str = str(CORRECTIONS_YAML)) -> list[tuple[re.Pattern, str, str]]:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    entries = yaml.safe_load(p.read_text(encoding="utf-8")).get("corrections") or []
    rules = []
    # longest first, so a phrase wins over a word inside it
    for e in sorted(entries, key=lambda e: -len(e["heard"])):
        pattern = re.compile(rf"(?<![{_WORD}]){re.escape(e['heard'])}(?![{_WORD}])")
        rules.append((pattern, e["write"], e["heard"]))
    return rules


def correct(text: str, rules: list | None = None, log_to_file: bool = True, context: dict | None = None
            ) -> tuple[str, list[dict]]:
    """Apply the dictionary. Returns (corrected text, [{heard, write, count}])."""
    rules = load_rules() if rules is None else rules
    changes = []
    for pattern, write, heard in rules:
        text, n = pattern.subn(write, text)
        if n:
            changes.append({"heard": heard, "write": write, "count": n})
    if changes:
        log.info("stt corrections: %s", "; ".join(f"{c['heard']} -> {c['write']} x{c['count']}" for c in changes))
        if log_to_file:
            row = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "changes": changes,
                   "text": text, **(context or {})}
            with _log_lock:
                LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with LOG_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return text, changes
