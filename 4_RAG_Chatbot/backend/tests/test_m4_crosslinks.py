"""Phase 7 cross-links: the questions the twin's "Ask SoilMind" and the Advisory page's "Why?" pre-fill, and the
answers they get.

The pre-filled questions come from POST /api/m4/prefill (chat/text.py `prefill`), the same function tested here, so
the wording the user sees is the wording tested. The intent model, retriever and reranker are the real trained ones;
the live Member 2 / 3 data is a fake with a south-east N and pH problem, so the answers are deterministic.
"""

import asyncio

import pytest

from member4_chatbot.chat import text as T
from member4_chatbot.chat.store import Escalations, Sessions
from member4_chatbot.paths import ANSWER_DIR, INTENT_DIR, RETRIEVER_DIR

needs_models = pytest.mark.skipif(
    not (INTENT_DIR / "model").exists() or not (RETRIEVER_DIR / "model").exists()
    or not (ANSWER_DIR / "config.json").exists(), reason="models / answer config not built")

SE = ["A3-05", "A3-06"]
KPIS = {"last_reading_at": "2026-10-20T04:00:00Z",
        "kpis": {"health": {"score": 25.3, "grade": "act", "limiting": "ph"},
                 "averages": {"n": 116.8, "p": 7.23, "k": 118.1, "ph": 5.78, "moisture": 47.0},
                 "worst": {"ph": {"node_id": "A3-05", "value": 5.19}, "n": {"node_id": "A3-06", "value": 75.7}}}}


def _alert(var, title, body, nodes=SE, zone="south-east", sev="crit", kind="low_now"):
    return {"id": f"{kind}-{var}", "kind": kind, "severity": sev, "variable": var, "nodes": nodes, "zone": zone,
            "zone_label": {"en": "South-east", "si": "ගිනිකොන"},
            "title_officer": {"en": title, "si": title + " (si)"}, "title_farmer": {"en": title, "si": title + " (si)"},
            "body_officer": {"en": body, "si": body}}


ALERTS = {"alerts": [
    _alert("n", "N below the deficiency level in the south-east zone",
           "A3-05 80.6 mg/kg, A3-06 75.7 mg/kg (deficiency level 107.1 mg/kg). About 30% of the field."),
    _alert("ph", "pH below the deficiency level in the south-east zone",
           "A3-05 5.19, A3-06 5.20 (deficiency level 5.50). About 23% of the field."),
    _alert("p", "P in the north-west zone forecast to drop below the deficiency level in 40 days",
           "A3-01 5.6 mg/kg now.", nodes=["A3-01"], zone="north-west", sev="watch", kind="forecast_low"),
]}
SOURCES = [{"title": "RRDI Soil Science Division, soil test kit", "url": "https://doa.gov.lk/rrdi_technology_soilscience_soiltestkit/"}]
RECS = {"plot_area_acres": 4.549, "actions": [
    {"id": "dolomite-se", "product": "dolomite", "kg_per_acre": 505.86, "kg_total": 717.0, "priority": "crit",
     "window_start": "2026-10-20", "window_end": "2026-10-23", "date": "2026-10-20",
     "title": {"en": "Spread lime (dolomite)", "si": "ඩොලමයිට් (හුණු) යොදන්න"},
     "instruction": {"en": "Spread about 14½ bags (717 kg) of dolomite on the south-east part of the field.",
                     "si": "කුඹුරේ ගිනිකොන කොටසට ඩොලමයිට් කි.ග්‍රෑ. 717 යොදන්න."},
     "reason": {"en": "pH 5.18 is below 5.5.", "si": "pH 5.18 අඩුයි."},
     "farmer_text": {"en": "Your soil is sour.", "si": "ඔබේ පස ආම්ලිකයි."},
     "zone": {"id": "z1", "nodes": SE, "whole_field": False, "direction": "south-east", "area_acres": 1.42,
              "area_share": 0.31, "label": {"en": "south-east part", "si": "ගිනිකොන කොටස"}},
     "sources": SOURCES,
     "expected_effect": {"target": "ph", "change_median": 0.45, "unit": "pH", "at_date": "2026-11-18"}},
    {"id": "urea-1", "product": "urea", "kg_per_acre": 20.23, "kg_total": 92.0, "priority": "crit",
     "window_start": "2026-11-13", "window_end": "2026-11-16", "date": "2026-11-13",
     "title": {"en": "Urea, split 1 of 4", "si": "යූරියා 1"},
     "instruction": {"en": "Spread about 2 bags (92 kg) of urea on the whole field.", "si": "මුළු කුඹුරටම යූරියා කි.ග්‍රෑ. 92."},
     "reason": {"en": "RRDI split 1 of 4; N below 107.1 mg/kg at A3-05, A3-06.", "si": "යෙදීම් 4 න් 1."},
     "farmer_text": {"en": "Feeds the paddy.", "si": "වී පෝෂණය."},
     "zone": {"id": "z0", "nodes": ["A3-01", "A3-02", "A3-03", "A3-04", *SE], "whole_field": True, "direction": None,
              "area_acres": 4.549, "area_share": 1.0, "label": {"en": "whole field", "si": "මුළු කුඹුර"}},
     "sources": SOURCES, "expected_effect": None},
    {"id": "tsp-1", "product": "TSP", "kg_per_acre": 12.0, "kg_total": 55.0, "priority": "good",
     "window_start": "2026-10-27", "window_end": "2026-10-30", "date": "2026-10-27",
     "title": {"en": "TSP (basal)", "si": "TSP"}, "instruction": {"en": "Spread 55 kg of TSP.", "si": "TSP කි.ග්‍රෑ. 55."},
     "reason": {"en": "P 6.9 mg/kg, RRDI class 5-10.", "si": "P 6.9."}, "farmer_text": {"en": "Roots.", "si": "මුල්."},
     "zone": {"id": "z0", "nodes": [], "whole_field": True, "direction": None, "area_acres": 4.549, "area_share": 1.0,
              "label": {"en": "whole field", "si": "මුළු කුඹුර"}}, "sources": SOURCES, "expected_effect": None},
    {"id": "mop-1", "product": "MOP", "kg_per_acre": 4.0, "kg_total": 18.0, "priority": "good",
     "window_start": "2026-11-27", "window_end": "2026-11-27", "date": "2026-11-27",
     "title": {"en": "MOP", "si": "MOP"}, "instruction": {"en": "Spread 18 kg of MOP.", "si": "MOP කි.ග්‍රෑ. 18."},
     "reason": {"en": "K class 80-160.", "si": "K."}, "farmer_text": {"en": "Strong stems.", "si": "කඳ."},
     "zone": {"id": "z0", "nodes": [], "whole_field": True, "direction": None, "area_acres": 4.549, "area_share": 1.0,
              "label": {"en": "whole field", "si": "මුළු කුඹුර"}}, "sources": SOURCES, "expected_effect": None},
]}


class FakeLive:
    async def kpis(self): return KPIS
    async def alerts(self): return ALERTS
    async def recommendations(self): return RECS
    async def field(self): return {"name": "Plot A-3", "field_id": "A-3"}
    async def season(self): return {"age_class": "3_5_month"}


@pytest.fixture(scope="module")
def pipe():
    from member4_chatbot.chat.pipeline import ChatPipeline

    return ChatPipeline(FakeLive(), Escalations(), Sessions())


def ask(p, q, session="x", **kw):
    return asyncio.run(p.answer(session, q, **kw))


def zone_ctx(zone="south-east", variable="n", nodes=SE, side="low"):
    return {"kind": "zone", "source": "twin", "zone": zone, "variable": variable, "nodes": nodes, "side": side}


def action_ctx(a):
    return {"kind": "recommendation", "source": "advisory", "action_id": a["id"], "product": a["product"]}


# ------------------------------------------------------------------ the pre-filled questions themselves
def test_prefill_wording_has_no_numbers_and_names_the_zone():
    q = T.prefill(zone_ctx(), "en")
    assert q == "Why is nitrogen low in the south-east part of my field?"
    assert "ගිනිකොන" in T.prefill(zone_ctx(), "si")
    assert T.prefill(None, "en") == T.PREFILL_FIELD["en"]
    assert T.prefill(action_ctx(RECS["actions"][0]), "en") == "Why should I apply dolomite on my field now?"
    for z in T.ZONE_NAME:
        for lang in ("en", "si"):
            for v in ("n", "p", "k", "ph", "health", None):
                assert not any(ch.isdigit() for ch in T.prefill(zone_ctx(z, v), lang))


DEMO_ZONE = [(zone_ctx(variable=v), lang) for v in ("n", "ph", "health") for lang in ("en", "si")]


@needs_models
@pytest.mark.parametrize("ctx, lang", DEMO_ZONE)
def test_twin_question_alone_is_routed_to_the_field_answer(pipe, ctx, lang):
    """Even if the context were lost, our intent model sends the pre-filled demo question to the live answer."""
    r = ask(pipe, T.prefill(ctx, lang), session=f"alone-{ctx['variable']}-{lang}", ui_lang=lang)
    assert r["intent"] == "my_field_status" and r["path"] == "template", (r["intent"], r["confidence"])
    assert r["answer_lang"] == lang
    assert r["answer_lines"][0].startswith(T.ZONE_ALERT[lang].split("{zone}")[0] + "South-east"
                                           if lang == "en" else T.ZONE_ALERT[lang].split("{zone}")[0])
    assert "80.6" in r["answer"] and "107.1" in r["answer"]  # the zone's values and Member 2's level


@needs_models
def test_every_zone_question_routes_to_the_field_answer(pipe):
    """All compass zones and layers, both languages: the share routed to the live answer by the intent model alone
    (reported), and with the context every one is a zone answer."""
    alone, total = 0, 0
    for z in T.ZONE_NAME:
        for v in ("n", "p", "k", "ph", "health"):
            for lang in ("en", "si"):
                q = T.prefill(zone_ctx(z, v, nodes=["A3-01"]), lang)
                r = ask(pipe, q, session=f"all-{z}-{v}-{lang}", ui_lang=lang, context=zone_ctx(z, v, nodes=["A3-01"]))
                assert r["path"] == "template" and r["map_focus"], (q, r["path"])
                total += 1
                alone += ask(pipe, q, session=f"alone2-{z}-{v}-{lang}", ui_lang=lang)["intent"] == "my_field_status"
    print(f"\nzone questions routed to the field answer by the intent model alone: {alone}/{total}")
    assert alone / total >= 0.9


@needs_models
@pytest.mark.parametrize("lang", ["en", "si"])
def test_twin_zone_answer_explains_the_zone_and_what_to_do(pipe, lang):
    ctx = zone_ctx()
    r = ask(pipe, T.prefill(ctx, lang), session=f"zone-{lang}", ui_lang=lang, context=ctx, persona="officer")
    lines = r["answer_lines"]
    assert r["path"] == "template" and r["context"]["kind"] == "zone"
    assert "75.7" in lines[0] and "107.1" in lines[0]                      # N in the zone vs the deficiency level
    assert any("5.19" in line and "5.50" in line for line in lines[:2])     # and its pH problem
    assert any("92" in line for line in lines) and any("717" in line for line in lines)  # urea AND lime advised
    assert "25.3" in lines[-1]                                              # then the whole-field summary
    zones = [m for m in r["map_focus"] if m["kind"] == "zone"]
    assert zones[0]["nodes"] == SE and zones[0]["variable"] == "n" and zones[0]["zone"] == "south-east"
    assert {v["key"] for v in r["live_values"]} >= {"plan:urea-1", "plan:dolomite-se"}


@needs_models
def test_text_naming_a_zone_or_sensor_gets_the_zone_answer(pipe):
    r = ask(pipe, "How is the south-east part of my field doing, and what should I do?", session="txt")
    assert r["path"] == "template" and r["answer_lines"][0].startswith("South-east part of your field")
    r = ask(pipe, "What is the status of A3-06 in my field?", session="node")
    if r["intent"] == "my_field_status":
        assert "A3-06" in r["answer_lines"][0] and r["map_focus"][0]["nodes"] == SE


@needs_models
def test_zone_without_alert_says_so(pipe):
    ctx = zone_ctx("north-east", "k", nodes=["A3-02"])
    r = ask(pipe, T.prefill(ctx, "en"), session="ne", context=ctx)
    assert r["answer_lines"][0] == T.ZONE_NO_ALERT["en"].format(zone="north-east")
    assert r["map_focus"][0]["nodes"] == ["A3-02"]


# ------------------------------------------------------------------ "Why?" on the Advisory page
@needs_models
@pytest.mark.parametrize("lang", ["en", "si"])
@pytest.mark.parametrize("action", RECS["actions"], ids=[a["id"] for a in RECS["actions"]])
def test_advisory_why_explains_that_recommendation(pipe, action, lang):
    ctx = action_ctx(action)
    r = ask(pipe, T.prefill(ctx, lang), session=f"why-{action['id']}-{lang}", ui_lang=lang, context=ctx,
            persona="officer")
    assert r["path"] == "template" and r["answer_lang"] == lang
    assert action["instruction"][lang] in r["answer_lines"][0]
    assert any(line.startswith(T.ACTION_WHY[lang].split("{")[0]) and action["reason"][lang] in line
               for line in r["answer_lines"])
    assert r["citations"][0]["url"] == SOURCES[0]["url"]                    # the recommendation's own source
    assert r["live_values"][0]["key"] == f"plan:{action['id']}"
    if not action["zone"]["whole_field"]:
        assert r["map_focus"][0]["nodes"] == SE and r["map_focus"][0]["variable"] == "ph"
    if action["expected_effect"]:
        assert any("+0.45" in line for line in r["answer_lines"])


@needs_models
def test_why_for_an_action_no_longer_in_the_advice(pipe):
    r = ask(pipe, "Why should I apply urea on my field now?", session="gone",
            context={"kind": "recommendation", "action_id": "urea-old", "product": "urea"})
    assert "92 kg" in r["answer_lines"][0]                                  # the next urea action instead
    r = ask(pipe, "Why?", session="gone2", context={"kind": "recommendation", "action_id": "x", "product": None})
    assert r["answer_lines"] == [T.ACTION_GONE["en"]]


@needs_models
def test_farmer_why_leads_with_the_plain_words(pipe):
    ctx = action_ctx(RECS["actions"][0])
    r = ask(pipe, T.prefill(ctx, "en"), session="farmer-why", context=ctx, persona="farmer")
    assert "Your soil is sour." in r["answer_lines"][1]


# ------------------------------------------------------------------ graceful degradation
@needs_models
def test_reranker_failure_falls_back_to_retriever_sentences(pipe, monkeypatch):
    from member4_chatbot.chat import pipeline as P

    def broken(*a, **k):
        raise RuntimeError("reranker down")
    monkeypatch.setattr(P, "choose", broken)
    r = ask(pipe, "Which fungicides can I use for blast?", session="deg")
    assert r["path"] == "extractive" and r["degraded"] == ["reranker"] and r["selection"] == "retriever_cosine"
    assert r["note"].startswith(T.DEGRADED_NOTE["en"])
    cited = pipe.passages.loc[r["citations"][0]["passage_id"], "text"]
    assert all(line in cited for line in r["answer_lines"])                 # still copied from the passage


@needs_models
def test_intent_failure_answers_from_retrieval(pipe, monkeypatch):
    from member4_chatbot import models

    def broken(*a, **k):
        raise RuntimeError("intent model down")
    monkeypatch.setattr(models, "predict_intent", broken)
    r = ask(pipe, "Which fungicides can I use for blast?", session="deg2")
    assert "intent" in r["degraded"] and r["path"].startswith("extractive") and r["citations"]


# ------------------------------------------------------------------ the route
def test_prefill_route(client):
    r = client.post("/api/m4/prefill", json={"context": zone_ctx(), "lang": "en"})
    assert r.status_code == 200
    assert r.json() == {"question": T.prefill(zone_ctx(), "en"), "context": {**zone_ctx(), "action_id": None,
                                                                             "product": None},
                        "label": {"en": "south-east · nitrogen · A3-05, A3-06", "si": "ගිනිකොන · නයිට්‍රජන් · A3-05, A3-06"}}
    r = client.post("/api/m4/prefill", json={"lang": "si"})
    assert r.json()["question"] == T.PREFILL_FIELD["si"]
    assert client.post("/api/m4/prefill", json={"context": {"kind": "nope"}}).status_code == 422
