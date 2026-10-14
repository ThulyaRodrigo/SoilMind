"""The training dataset: split leakage, sizes, and that template answers really come from their passage."""

import re

import pandas as pd
import pytest

from member4_chatbot.dataset.build import assign_groups, assign_splits
from member4_chatbot.dataset.paraphrase import paraphrase
from member4_chatbot.dataset.templates import parse_fert_table
from member4_chatbot.paths import PASSAGES_CSV, RAW_DOCS_DIR, TRAINING_DIR

pytestmark = pytest.mark.skipif(not (TRAINING_DIR / "qa_pairs.csv").exists(), reason="dataset not built")


@pytest.fixture(scope="module")
def qa():
    return pd.read_csv(TRAINING_DIR / "qa_pairs.csv", keep_default_na=False)


@pytest.fixture(scope="module")
def intents():
    return pd.read_csv(TRAINING_DIR / "intents.csv", keep_default_na=False)


@pytest.fixture(scope="module")
def passages():
    return pd.read_csv(PASSAGES_CSV, keep_default_na=False)


def test_no_passage_in_two_splits(qa):
    assert (qa.groupby("passage_id").split.nunique() == 1).all()


def test_twin_passages_in_same_split(qa):
    split_of = qa.drop_duplicates("passage_id").set_index("passage_id").split
    for r in qa[qa.also_correct != ""].itertuples():
        for twin in r.also_correct.split(";"):
            if twin in split_of:
                assert split_of[twin] == r.split, (r.passage_id, twin)


def test_groups_never_split(intents):
    assert (intents.groupby("group_id").split.nunique() == 1).all()


def test_every_passage_id_exists(qa, passages):
    ids = set(passages.id)
    assert set(qa.passage_id) <= ids
    for also in qa.also_correct[qa.also_correct != ""]:
        assert set(also.split(";")) <= ids


def test_size_targets(intents, qa):
    assert len(intents) >= 1500
    assert (intents.lang == "si").mean() >= 0.40
    assert intents.intent.value_counts().min() >= 60
    assert set(intents.split) == {"train", "val", "test"}
    assert {"en", "si"} <= set(qa[qa.split == "test"].lang)


def test_out_of_scope_set():
    oos = pd.read_csv(TRAINING_DIR / "out_of_scope.csv", keep_default_na=False)
    assert len(oos) >= 100
    assert set(oos.split) == {"train", "val", "test"}


def test_template_numbers_appear_in_their_passage(qa, passages):
    """Every number in a fertilizer / soil-report / variety template answer (except kg/acre conversions) is in
    the passage the question points to: the pairs are checkable."""
    text = dict(zip(passages.id, passages.text))
    rows = qa[(qa.source == "auto_template") & qa.template.str.match(r"fert_split|fert_total|soil_report|variety_(days|yield|year)")]
    assert len(rows) > 100
    for r in rows.itertuples():
        answer = re.sub(r"(about|පමණ)?\s*\(?\s*(අක්කරයකට කිලෝ\s*)?[\d.]+\s*(kg per acre|ක් පමණ\))", "", r.answer)
        for num in re.findall(r"\d+(?:\.\d+)?", answer):
            n = num[:-2] if num.endswith(".0") else num
            assert n in text[r.passage_id], (r.question, r.answer, n)


def test_fert_tables_parse_and_languages_agree():
    for zone in ("irrigated_izdz", "rainfed_izdz", "irrigated_wz", "rainfed_wz"):
        en = parse_fert_table((RAW_DOCS_DIR / f"rrdi_fert_{zone}_en.txt").read_text(encoding="utf-8"))
        si = parse_fert_table((RAW_DOCS_DIR / f"rrdi_fert_{zone}_si.txt").read_text(encoding="utf-8"))
        assert en == si
        for age, rows in en.items():
            # the splits add up to the published total
            assert sum(v.get("urea", 0) for k, v in rows.items() if k != "total") == rows["total"]["urea"]


def test_split_assignment_unit():
    rows = [{"passage_id": f"p{i % 10}", "also_correct": "p1" if i % 10 == 0 else "", "origin_id": str(i),
             "intent": "a" if i % 2 else "b"} for i in range(200)]
    rows += [{"passage_id": "", "also_correct": "", "origin_id": f"q{i // 2}", "intent": "c"} for i in range(60)]
    assign_groups(rows)
    assign_splits(rows, seed=1)
    df = pd.DataFrame(rows)
    assert (df[df.passage_id != ""].groupby("passage_id").split.nunique() == 1).all()
    assert df[df.passage_id.isin(["p0", "p1"])].split.nunique() == 1  # twins together
    assert (df.groupby("group_id").split.nunique() == 1).all()
    assert set(df[df.intent == "c"].split) == {"train", "val", "test"}


def test_paraphrase_keeps_numbers_and_names():
    import random

    rng = random.Random(0)
    q = "How much urea should I apply at 4 weeks for Bg 300 per acre?"
    for _ in range(20):
        p = paraphrase(q, rng)
        if p:
            assert "4" in p and "Bg 300" in p and "urea" in p.lower()
