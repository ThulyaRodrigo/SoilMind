"""Sentence reranker (labels, inputs, selection) and field context (plot profile, question override, ranking)."""

import asyncio

import pytest

from member4_chatbot.chat.extractive import split_units
from member4_chatbot.chat.profile import (Profile, match_score, passage_tags, plot_profile, question_profile,
                                          row_age)
from member4_chatbot.chat.rerank import choose, contains_fact, group_age, key_facts, table_context, unit_inputs
from member4_chatbot.paths import RERANKER_DIR

# shaped like the RRDI tables: header and group rows have one cell more than the week rows
TABLE = ("Fertilizer\nAge | Time | Urea | TSP\nFor 3 Month | Basic | | 55\n2 Weeks | 50 |\n6 Weeks | 65 |\n"
         "For 3 1/2 Month | Basic | | 55\n2 Weeks | 50 |\n6 Weeks | 70 |")


# ------------------------------------------------------------------ profile
def test_plot_profile_from_member2_api():
    recs = {"rules": [{"source_url": None},
                      {"source_url": "https://doa.gov.lk/rrdi_fertilizerrecomendation_soilreport_irrigated_izdz/"}]}
    assert plot_profile(recs, {"age_class": "3_5_month"}) == Profile("izdz", "irrigated", "3.5")
    assert plot_profile({}, None) == Profile()


@pytest.mark.parametrize("q, expected", [
    ("How much urea for rainfed paddy in the Wet Zone?", Profile("wz", "rainfed", None)),
    ("I farm irrigated paddy in Polonnaruwa, 4 month variety", Profile("izdz", "irrigated", "4")),
    ("Urea at 6 weeks for a 3 1/2 month variety", Profile(None, None, "3.5")),
    ("වැසි දියෙන් වගා කරන තෙත් කලාපයේ කුඹුරට යූරියා කීයද?", Profile("wz", "rainfed", None)),
    ("මාස 3 1/2 වී වලට යූරියා කොච්චරද?", Profile(None, None, "3.5")),
    ("ගම්පහ කුඹුරට පොහොර", Profile("wz", None, None)),
    ("How much urea?", Profile()),
])
def test_question_profile(q, expected):
    assert question_profile(q) == expected


def test_question_overrides_plot():
    plot = Profile("izdz", "irrigated", "3.5")
    assert plot.merged_with(question_profile("rainfed Wet Zone, 4 month")) == Profile("wz", "rainfed", "4")
    assert plot.merged_with(question_profile("how much urea for my field?")) == plot


def test_passage_tags_and_match():
    assert passage_tags("rrdi_fert_rainfed_wz_si") == Profile("wz", "rainfed", None)
    assert passage_tags("naicc_cal_paddy_izdz_irrigated_en") == Profile("izdz", "irrigated", None)
    p = Profile("izdz", "irrigated", None)
    assert match_score(p, passage_tags("rrdi_fert_irrigated_izdz_en")) == 2
    assert match_score(p, passage_tags("rrdi_fert_rainfed_wz_en")) == -2
    assert match_score(p, passage_tags("rrdi_pests_thrips")) == 0


class _Ranked:
    def __init__(self, doc): self.doc, self.passage_id = doc, doc


def test_profile_boost_reorders_ranking():
    from member4_chatbot.chat.retrieval import PassageRanker

    class R(PassageRanker):
        def __init__(self): pass
        def rank(self, q, query, mode, k): return [_Ranked(d) for d in ("rrdi_fert_rainfed_izdz_en", "rrdi_fert_irrigated_wz_en", "rrdi_fert_irrigated_izdz_en")][:k]

    r = R()
    prof = Profile("izdz", "irrigated", None)
    assert [h.doc for h in r.rank_profiled(None, "", "dense", 3, prof, 0)][0] == "rrdi_fert_rainfed_izdz_en"
    assert [h.doc for h in r.rank_profiled(None, "", "dense", 3, prof, 3)][0] == "rrdi_fert_irrigated_izdz_en"


def test_row_age():
    assert row_age("For 3 1/2 Month | Basic |") == "3.5"
    assert row_age("මාස 4 සඳහා | මූලික |") == "4"
    assert row_age("2 Weeks | 50 |") is None


# ------------------------------------------------------------------ reranker labels, inputs, selection
def test_key_facts_and_labels():
    assert key_facts("RRDI: 65 kg/ha at 6 weeks for Bg 300", ["6 Weeks | 65 | | ... Bg 300"]) == {"65", "6", "Bg 300"}
    assert contains_fact("6 Weeks | 65 |", "65") and not contains_fact("650 kg", "65")
    assert not contains_fact("6.5 t/ha", "6")


def test_table_context_and_inputs():
    units = split_units(TABLE, "Fertilizer")
    i = next(k for k, u in enumerate(units) if u.text.startswith("6 Weeks | 70"))
    ctx = [units[c].text for c in table_context(units, i)]
    assert ctx == ["Age | Time | Urea | TSP", "For 3 1/2 Month | Basic | | 55"]
    assert group_age(units, i) == "3.5"
    assert unit_inputs(units, "Fertilizer")[i] == "Fertilizer :: Age | Time | Urea | TSP ; For 3 1/2 Month | Basic | | 55 ; 6 Weeks | 70 |"


def _scorer(best_words):
    def score(q, texts):
        return [1.0 if any(w in t.split("::")[-1].split(";")[-1] for w in best_words) else 0.1 for t in texts]
    return score


def test_choose_picks_row_with_context_and_age_bonus():
    p = [{"id": "p1", "text": TABLE, "title": "Fertilizer", "lang": "en"}]
    c = choose("urea at 6 weeks?", p, _scorer(["6 Weeks"]), answer_lang="en", max_units=1, margin=0.0, lang_delta=0.1,
               age="3.5", age_bonus=0.3)
    assert c.selected == ["6 Weeks | 70 |"]  # both 6-week rows score 1.0; the plot's age class wins
    assert c.lines == ["Age | Time | Urea | TSP", "For 3 1/2 Month | Basic | | 55", "6 Weeks | 70 |"]
    c = choose("urea at 6 weeks?", p, _scorer(["6 Weeks"]), answer_lang="en", max_units=1, margin=0.0, lang_delta=0.1,
               age="3", age_bonus=0.3)
    assert c.selected == ["6 Weeks | 65 |"]


def test_choose_prefers_answer_language_within_delta():
    en = {"id": "en", "text": "Apply urea at two weeks after planting.", "title": "", "lang": "en"}
    si = {"id": "si", "text": "සති දෙකේදී යූරියා යොදන්න කියා නිර්දේශ කෙරේ.", "title": "", "lang": "si"}
    scores = {"Apply urea at two weeks after planting.": 0.9, "සති දෙකේදී යූරියා යොදන්න කියා නිර්දේශ කෙරේ.": 0.85}
    sc = lambda q, ts: [scores[t] for t in ts]
    assert choose("q", [en, si], sc, answer_lang="si", max_units=2, margin=0.1, lang_delta=0.1).passage_id == "si"
    assert choose("q", [en, si], sc, answer_lang="si", max_units=2, margin=0.1, lang_delta=0.01).passage_id == "en"


# ------------------------------------------------------------------ the trained reranker in the pipeline
needs_reranker = pytest.mark.skipif(not (RERANKER_DIR / "model").exists(), reason="reranker not trained")


@needs_reranker
def test_reranker_loads_and_ranks():
    import json

    from member4_chatbot.chat.rerank import Reranker

    cfg = json.loads((RERANKER_DIR / "config.json").read_text())
    r = Reranker(RERANKER_DIR / "model", cfg["max_length"])
    s = r("How much urea at 4 weeks?", ["4 Weeks | 75 | | 25 |", "Paddy bug sucks the developing grains."])
    assert len(s) == 2 and all(0 <= x <= 1 for x in s) and s[0] > s[1]


@needs_reranker
def test_pipeline_uses_reranker_and_profile():
    from test_m4_chat import FakeLive

    from member4_chatbot.chat.pipeline import ChatPipeline
    from member4_chatbot.chat.store import Escalations, Sessions

    class LiveWithProfile(FakeLive):
        async def season(self): return {"age_class": "3_5_month"}
        async def recommendations(self):
            r = dict(await super().recommendations())
            r["rules"] = [{"source_url": "https://doa.gov.lk/rrdi_fertilizerrecomendation_irrigated_izdz/"}]
            return r

    p = ChatPipeline(LiveWithProfile(), Escalations(), Sessions())
    r = asyncio.run(p.answer("rr", "How much urea should I apply at 2 weeks for a 3 month variety?", "officer"))
    assert r["path"].startswith("extractive"), (r["intent"], r["path"])
    # plot profile (Member 2) fills zone and water; the question's "3 month" wins over the plot's 3 1/2 month
    assert r["selection"] == p.rerank_config["picker"] and r["field_profile"] == {"zone": "izdz", "water": "irrigated", "age": "3"}
    assert r["full_passage"] and r["highlights"]
    assert all(h in r["full_passage"] for h in r["highlights"])
    r = asyncio.run(p.answer("rr2", "How much urea for rainfed Wet Zone paddy?", "officer"))
    assert r["field_profile"]["zone"] == "wz" and r["field_profile"]["water"] == "rainfed"  # the question wins
