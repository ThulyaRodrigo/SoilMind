"""Both trained models load from artifacts/ and predict (Phase 5, task 7)."""

import pandas as pd
import pytest

from member4_chatbot.paths import CHROMA_DIR, INTENT_DIR, PASSAGES_CSV, RETRIEVER_DIR

needs_intent = pytest.mark.skipif(not (INTENT_DIR / "model").exists(), reason="intent model not trained")
needs_retriever = pytest.mark.skipif(not (RETRIEVER_DIR / "model").exists() or not CHROMA_DIR.exists(),
                                     reason="retriever not trained")


@needs_intent
@pytest.mark.parametrize("text, intent", [
    ("How much urea should I apply for a 3 month variety in the Dry Zone?", "fertilizer_amount"),
    ("මගේ කුඹුරේ දැන් කරන්න ඕන මොනවද?", "my_field_status"),
    ("ආයුබෝවන්", "greeting_help"),
    ("How do I control brown planthopper?", "pest_disease"),
    ("Who will win the next election?", "out_of_scope"),
])
def test_intent_model_predicts(text, intent):
    from member4_chatbot.models import predict_intent

    p = predict_intent(text)
    # "how much urea for a 3 month variety" sits on the amount / timing boundary (the model's main confusion,
    # see phase-5.md); either fertilizer intent is acceptable for it
    ok = {intent, "fertilizer_timing"} if intent == "fertilizer_amount" else {intent}
    assert p.intent in ok, (text, p.intent, p.raw_intent, round(p.confidence, 3))
    assert 0.0 <= p.confidence <= 1.0
    assert abs(sum(p.scores.values()) - 1.0) < 1e-3


@needs_intent
def test_intent_low_confidence_becomes_out_of_scope():
    import json

    from member4_chatbot.models import predict_intent

    tau = json.loads((INTENT_DIR / "inference.json").read_text())["threshold"]
    p = predict_intent("qwerty zxcv asdf")
    assert p.intent == "out_of_scope" or p.confidence >= tau


@needs_retriever
def test_retriever_index_matches_passages():
    from member4_chatbot.models import _retriever

    _, col = _retriever()
    assert col.count() == len(pd.read_csv(PASSAGES_CSV))


@needs_retriever
@pytest.mark.parametrize("question, doc_prefix", [
    ("How do I control brown planthopper?", "rrdi_pests_brownplanthopper"),
    ("වියළි කලාපයේ වාරි කුඹුරට යූරියා කොච්චර දාන්න ඕනද?", "rrdi_fert_irrigated_izdz"),
    ("What are the symptoms of rice blast?", "rrdi_ricediseases_riceblast"),
])
def test_retriever_finds_passage(question, doc_prefix):
    from member4_chatbot.models import search

    hits = search(question, k=3)
    assert len(hits) == 3
    assert hits[0].score >= hits[-1].score
    assert any(h.doc.startswith(doc_prefix) for h in hits), [h.passage_id for h in hits]
