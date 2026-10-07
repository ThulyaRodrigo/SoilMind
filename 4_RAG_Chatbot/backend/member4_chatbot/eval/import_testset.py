"""Import the team-reviewed evaluation set: eval/qa_testset.xlsx -> eval/qa_testset.csv.

    python -m member4_chatbot.eval.import_testset

The spreadsheet is the only file the team reviews (sheet "questions"). This script validates it, drops rows
marked "remove", and writes the CSV that run_eval.py and notebook 04 read. It prints how many rows a person has
checked; the evaluation report is labelled "unchecked" until every row has been checked.
"""

from __future__ import annotations

import sys

import pandas as pd

from member4_chatbot.paths import EVAL_DIR

XLSX = EVAL_DIR / "qa_testset.xlsx"
CSV = EVAL_DIR / "qa_testset.csv"
REQUIRED = ["id", "lang", "question", "expected_behaviour", "expected_intent", "expected_answer", "key_facts",
            "expected_source_doc", "expected_passage_id", "expected_source_url", "live_checks", "review_status",
            "reviewer", "review_notes"]
BEHAVIOURS = {"answer", "live", "refuse"}
STATUSES = {"needs human check", "checked - ok", "checked - edited", "remove"}
LIVE_CHECKS = {"health_score", "avg_n", "avg_p", "avg_k", "avg_ph", "avg_moisture", "plan_urea", "plan_dolomite",
               "plan_tsp", "plan_mop"}


def load(path=XLSX) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(path, sheet_name="questions", dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    problems = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"columns missing in {path.name}: {missing}")
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df["review_status"] = df.review_status.str.lower()
    for r in df.itertuples():
        if r.lang not in ("en", "si"):
            problems.append(f"{r.id}: lang must be en or si, got {r.lang!r}")
        if r.expected_behaviour not in BEHAVIOURS:
            problems.append(f"{r.id}: expected_behaviour must be one of {sorted(BEHAVIOURS)}")
        if r.review_status not in STATUSES:
            problems.append(f"{r.id}: review_status must be one of {sorted(STATUSES)}")
        if r.expected_behaviour == "answer" and not r.key_facts:
            problems.append(f"{r.id}: an 'answer' row needs key_facts")
        if r.expected_behaviour == "live":
            bad = {c.strip() for c in r.live_checks.split(";") if c.strip()} - LIVE_CHECKS
            if bad or not r.live_checks:
                problems.append(f"{r.id}: live_checks must be ';'-separated from {sorted(LIVE_CHECKS)}")
        if not r.question:
            problems.append(f"{r.id}: empty question")
    if df.id.duplicated().any():
        problems.append(f"duplicate ids: {sorted(df.id[df.id.duplicated()])}")
    return df, problems


def main(argv: list[str] | None = None) -> pd.DataFrame:
    df, problems = load()
    if problems:
        print("The spreadsheet has problems; fix them and run again:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    kept = df[df.review_status != "remove"].reset_index(drop=True)
    kept.to_csv(CSV, index=False, encoding="utf-8")
    checked = kept.review_status.str.startswith("checked").sum()
    print(f"{len(kept)} questions -> {CSV.name} ({len(df) - len(kept)} removed); human-checked: {checked} of {len(kept)}")
    print(kept.groupby(["lang", "expected_behaviour"]).size().unstack(fill_value=0).to_string())
    counts = kept.groupby("lang").size().to_dict()
    if counts.get("en", 0) < 30 or counts.get("si", 0) < 30:
        print("warning: fewer than 30 English or 30 Sinhala questions")
    for b, n in (("refuse", 10), ("live", 10)):
        if (kept.expected_behaviour == b).sum() < n:
            print(f"warning: fewer than {n} '{b}' questions")
    return kept


if __name__ == "__main__":
    main(sys.argv[1:])
