"""Phase 6 chat: extraction rules, number safety, bilingual templates, every pipeline path, and the API routes.

The live Member 2 / 3 data is replaced by a fake with fixed values, so the answers are deterministic; the intent
model and retriever are the real trained ones.
"""

import asyncio
import re
import string

import numpy as np
import pytest

from member4_chatbot.chat import text as T
from member4_chatbot.chat.extractive import extract, split_units, unsupported_numbers
from member4_chatbot.chat.store import Escalations, Sessions
from member4_chatbot.paths import ANSWER_DIR, INTENT_DIR, RETRIEVER_DIR

needs_models = pytest.mark.skipif(
    not (INTENT_DIR / "model").exists() or not (RETRIEVER_DIR / "model").exists()
    or not (ANSWER_DIR / "config.json").exists(), reason="models / answer config not built")

KPIS = {
    "last_reading_at": "2026-10-18T08:30:00Z",
    "kpis": {"health": {"score": 35.2, "grade": "act", "limiting": "n"},
             "averages": {"n": 123.1, "p": 7.05, "k": 118.3, "ph": 5.88, "moisture": 47.2},
             "worst": {"n": {"node_id": "A3-06", "value": 88.9}}},
}
ALERTS = {"alerts": [{"severity": "crit", "title_farmer": {"en": "Nitrogen is low", "si": "නයිට්‍රජන් අඩුයි"},
                      "title_officer": {"en": "N below threshold", "si": "N අඩුයි"}}]}
RECS = {"plot_area_acres": 4.549, "actions": [
    {"id": "dolomite-1", "product": "dolomite", "kg_per_acre": 505.86, "kg_total": 285.1, "priority": "crit",
     "window_start": "2026-10-19", "window_end": "2026-10-19", "date": "2026-10-19",
     "title": {"en": "Spread lime", "si": "හුණු යොදන්න"},
     "instruction": {"en": "Spread 285 kg of dolomite on the south-east part.", "si": "ගිනිකොන කොටසට ඩොලමයිට් කි.ග්‍රෑ. 285 යොදන්න."},
     "reason": {"en": "pH 5.49 is low.", "si": "pH 5.49 අඩුයි."}, "farmer_text": {"en": "Soil is sour.", "si": "පස ඇඹුල්."},
     "zone": {"label": {"en": "south-east part", "si": "ගිනිකොන කොටස"}}},
    {"id": "urea-1", "product": "urea", "kg_per_acre": 20.23, "kg_total": 92.0, "priority": "crit",
     "window_start": "2026-11-13", "window_end": "2026-11-16", "date": "2026-11-13",
     "title": {"en": "Urea dressing 1", "si": "යූරියා 1"},
     "instruction": {"en": "Spread 92 kg of urea on the whole field.", "si": "මුළු කුඹුරටම යූරියා කි.ග්‍රෑ. 92 යොදන්න."},
     "reason": {"en": "Feeds nitrogen.", "si": "නයිට්‍රජන් ලබා දෙයි."}, "farmer_text": {"en": "Feeds the paddy.", "si": "වී පෝෂණය."},
     "zone": {"label": {"en": "whole field", "si": "මුළු කුඹුර"}}},
]}


class FakeLive:
    def __init__(self, kpis=KPIS, fail=False):
        self._kpis, self.fail = kpis, fail

    async def _get(self, v):
        if self.fail:
            from member4_chatbot.chat.live import LiveError
            raise LiveError("down")
        return v

    async def kpis(self): return await self._get(self._kpis)
    async def alerts(self): return await self._get(ALERTS)
    async def recommendations(self): return await self._get(RECS)
    async def field(self): return await self._get({"name": "Plot A-3", "field_id": "A-3"})
    async def season(self): return await self._get({"age_class": "3_5_month"})


# ------------------------------------------------------------------ extraction and safety (no models)
def test_split_units_sentences_rows_and_title():
    units = split_units("Brown Plant Hopper\nIt sucks sap. It spreads virus diseases.\nA | B | C\nok", "Brown Plant Hopper")
    assert [u.kind for u in units] == ["sentence", "sentence", "row"]
    assert units[2].cells == 3


def _fake_encode(target_words):
    def encode(texts):
        v = np.array([[1.0 if any(w in t for w in target_words) else 0.0, 1.0] for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return encode


def test_extract_whole_table_when_a_row_wins():
    text = "Intro sentence about urea here.\nAge | Time | Urea\nFor 3 Month | Basic | 0\n4 Weeks | 75 |\nTotal | 225 |"
    q = np.array([1.0, 0.0])
    ex = extract("p", text, "", q, _fake_encode(["75"]), max_sentences=1, margin=0.0)
    assert ex.lines[0] == "Age | Time | Urea" and "4 Weeks | 75 |" in ex.lines and "Total | 225 |" in ex.lines
    assert "Intro sentence about urea here." not in ex.lines


def test_extract_list_after_lead_in():
    text = ("If the disease spread fast, following fungicides could be applied\nTebuconazole 10 ml in 16 l\n"
            "Carbendazim 11 g in 16 l\nSomething unrelated and long enough here.")
    ex = extract("p", text, "", np.array([1.0, 0.0]), _fake_encode(["following"]), max_sentences=1, margin=0.0)
    assert ex.lines[:3] == ["If the disease spread fast, following fungicides could be applied",
                            "Tebuconazole 10 ml in 16 l", "Carbendazim 11 g in 16 l"]


def test_unsupported_numbers():
    passage = "Apply 75 kg/ha urea at 4 weeks; area 4.549 acres; 20.23 kg per acre."
    assert unsupported_numbers("Use 75 kg at 4 weeks", passage) == set()
    assert unsupported_numbers("Area 4.55 acres, 20 kg per acre", passage) == set()  # roundings of source numbers
    assert unsupported_numbers("Use 80 kg", passage) == {"80"}
    assert unsupported_numbers("pH 5.9", passage) == {"5.9"}


def test_templates_fill_in_both_languages():
    values = dict(plot="Plot A-3", score="35.2", grade="x", var="n", node="A3-06", value="88.9", unit=" mg/kg", n="1",
                  p="2", k="3", ph="5.9", moisture="47", time="t", titles="a", area="4.55", instruction="i",
                  start="s", end="e", per_acre="20", reason="r", product="urea", ref="ESC-1")
    for name in ("GREETING", "REFUSE_OUT_OF_SCOPE", "REFUSE_NOT_FOUND", "OTHER_LANGUAGE_NOTE", "LIVE_UNAVAILABLE",
                 "NO_RECENT_READINGS", "FIELD_STATUS", "ALERTS_LINE", "NEXT_ACTION", "NEXT_ACTION_ONE_DAY", "NO_ACTION"):
        tmpl = getattr(T, name)
        assert set(tmpl) == {"en", "si"}, name
        en_fields = {f for _, f, _, _ in string.Formatter().parse(tmpl["en"]) if f}
        si_fields = {f for _, f, _, _ in string.Formatter().parse(tmpl["si"]) if f}
        assert en_fields == si_fields, name  # both languages use the same API values
        out = tmpl["si"].format(**values)
        assert "{" not in out and re.search(r"[඀-෿]", out), name
        # templates carry no agronomic numbers of their own (only the health score's 0-100 scale)
        assert not re.search(r"\d", re.sub(r"\{\w+\}|100", "", tmpl["en"])), name


def test_escalations_memory_fallback():
    store = Escalations(db_getter=None)
    ref = asyncio.run(store.add({"question": "q", "reason": "out_of_scope"}))
    items = asyncio.run(store.recent())
    assert ref.startswith("ESC-") and items[0]["ref"] == ref


def test_sessions_keep_last_three():
    s = Sessions()
    for i in range(5):
        s.add("a", {"question": str(i)})
    assert [t["question"] for t in s.history("a")] == ["2", "3", "4"]


# ------------------------------------------------------------------ pipeline paths (real models, fake live data)
@pytest.fixture(scope="module")
def pipe():
    from member4_chatbot.chat.pipeline import ChatPipeline

    return ChatPipeline(FakeLive(), Escalations(), Sessions())


def ask(p, q, session="t", **kw):
    return asyncio.run(p.answer(session, q, **kw))


@needs_models
def test_path_greeting(pipe):
    r = ask(pipe, "Hello", session="g")
    assert r["path"] == "greeting" and r["answer"] == T.GREETING["en"]


@needs_models
@pytest.mark.parametrize("q, lang", [("Who will win the next election?", "en"), ("ඊයේ ක්‍රිකට් මැච් එක දිනුවේ කවුද?", "si")])
def test_path_refused_out_of_scope(pipe, q, lang):
    r = ask(pipe, q, session=f"oos-{lang}")
    assert r["path"] == "refused" and r["escalated"] and r["escalation_ref"].startswith("ESC-")
    assert r["answer_lang"] == lang and r["escalation_ref"] in r["answer"]


@needs_models
def test_path_refused_low_retrieval(pipe):
    from dataclasses import replace

    from member4_chatbot.chat.pipeline import ChatPipeline

    strict = ChatPipeline(FakeLive(), Escalations(), Sessions(), replace(pipe.config, retrieval_threshold=0.99))
    r = ask(strict, "How do I control brown planthopper?", session="low")
    assert r["path"] == "refused" and r["retrieval_score"] < 0.99 and "forwarded" in r["answer"]


@needs_models
def test_path_extractive_with_citations(pipe):
    r = ask(pipe, "Which fungicides can I use for blast?", session="ex")
    assert r["path"] == "extractive" and r["intent"] == "pest_disease"
    assert r["citations"] and r["citations"][0]["url"].startswith("https://doa.gov.lk")
    cited = pipe.passages.loc[r["citations"][0]["passage_id"], "text"]
    for line in r["answer_lines"]:
        assert line in cited  # copied, never generated
    assert unsupported_numbers(r["answer"], cited) == set()


@needs_models
@pytest.mark.parametrize("q, lang", [("How is my field today?", "en"), ("මගේ කුඹුර අද කොහොමද?", "si")])
def test_path_template_my_field(pipe, q, lang):
    r = ask(pipe, q, session=f"field-{lang}")
    assert r["path"] == "template" and r["answer_lang"] == lang
    assert "35.2" in r["answer"] and "123.1" in r["answer"] and "A3-06" in r["answer"]
    assert {v["key"] for v in r["live_values"]} >= {"n", "ph", "health"}


@needs_models
def test_template_uses_the_product_asked_about(pipe):
    r = ask(pipe, "When should I put the next urea on my field?", session="urea")
    assert "92 kg of urea" in r["answer"] and "13 Nov 2026" in r["answer"]


@needs_models
def test_path_extractive_plus_template(pipe):
    r = ask(pipe, "How much urea should I apply for a 3 month variety in the Dry Zone?", session="fert")
    assert r["path"] == "extractive+template"
    assert any("92 kg of urea" in line for line in r["answer_lines"])
    assert r["live_values"] and r["live_values"][0]["key"] == "plan:urea-1"


@needs_models
def test_no_recent_readings(pipe):
    from member4_chatbot.chat.pipeline import ChatPipeline

    stale = {"last_reading_at": "2026-10-18T08:30:00Z", "kpis": {"health": None, "averages": {}, "worst": {}}}
    p = ChatPipeline(FakeLive(kpis=stale), Escalations(), Sessions(), pipe.config)
    r = ask(p, "How is my field today?", session="stale")
    assert r["path"] == "template" and "no recent readings" in r["answer"]


@needs_models
def test_live_down_refuses_and_escalates(pipe):
    from member4_chatbot.chat.pipeline import ChatPipeline

    p = ChatPipeline(FakeLive(fail=True), Escalations(), Sessions(), pipe.config)
    r = ask(p, "How is my field today?", session="down")
    assert r["path"] == "refused" and r["escalated"] and T.LIVE_UNAVAILABLE["en"] in r["answer"]


@needs_models
def test_follow_up_uses_previous_question(pipe):
    ask(pipe, "How much urea for a 3 month variety in the irrigated Dry Zone?", session="fu")
    r = ask(pipe, "and the MOP?", session="fu")
    assert r["follow_up_of"] and r["path"] != "refused"


@needs_models
def test_cross_language_note(pipe):
    r = ask(pipe, "දුඹුරු පැළ කීඩෑවා පඳුරකට කීයක් හිටියොත් බෙහෙත් ගහන්න ඕනද?", session="x")
    if r["path"].startswith("extractive") and r["citations"][0]["lang"] == "en":
        assert r["note"] == T.OTHER_LANGUAGE_NOTE["si"]


# ------------------------------------------------------------------ API routes
@needs_models
def test_api_chat_sources_escalations_model_info(client):
    from main import app
    from member4_chatbot.chat.pipeline import ChatPipeline
    from member4_chatbot.router import get_pipeline

    fake = ChatPipeline(FakeLive(), Escalations(), Sessions())
    app.dependency_overrides[get_pipeline] = lambda: fake
    try:
        r = client.post("/api/m4/chat", json={"session_id": "api", "message": "How is my field today?", "persona": "officer"})
        assert r.status_code == 200 and r.json()["path"] == "template"
        assert client.post("/api/m4/chat", json={"session_id": "api", "message": ""}).status_code == 422
        r = client.post("/api/m4/chat", json={"session_id": "api", "message": "Who will win the election?"})
        assert r.json()["escalated"]
    finally:
        app.dependency_overrides.clear()
    s = client.get("/api/m4/sources").json()
    assert s["passages_total"] > 500 and s["waiting_for_ocr"] >= 1 and len(s["documents"]) > 200
    e = client.get("/api/m4/escalations").json()
    assert e["count"] == len(e["items"])
    m = client.get("/api/m4/model-info").json()
    assert [c["name"] for c in m["models"]] == ["intent", "retriever", "reranker", "answer"]
    assert "NOT human-checked" in m["data_warning"] and m["answer_config"]["retrieval_threshold"] > 0
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/m4/chat", "/api/m4/sources", "/api/m4/escalations", "/api/m4/model-info"} <= set(paths)


# ------------------------------------------------------------------ the evaluation set
def test_eval_set_import():
    from member4_chatbot.eval.import_testset import XLSX, load

    df, problems = load(XLSX)
    assert not problems
    assert len(df) == 60 and (df.lang == "en").sum() == 30 and (df.lang == "si").sum() == 30
    assert (df.expected_behaviour == "refuse").sum() == 10 and (df.expected_behaviour == "live").sum() == 10


def test_eval_set_import_rejects_bad_rows(tmp_path):
    import pandas as pd

    from member4_chatbot.eval.import_testset import XLSX, load

    df = pd.read_excel(XLSX, sheet_name="questions", dtype=str).fillna("")
    df.loc[0, "expected_behaviour"] = "maybe"
    df.loc[1, "lang"] = "ta"
    bad = tmp_path / "bad.xlsx"
    df.to_excel(bad, sheet_name="questions", index=False)
    _, problems = load(bad)
    assert any("expected_behaviour" in p for p in problems) and any("lang" in p for p in problems)
