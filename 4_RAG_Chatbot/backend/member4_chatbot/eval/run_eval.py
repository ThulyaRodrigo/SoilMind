"""End-to-end evaluation of the chatbot on the human-reviewed set (eval/qa_testset.csv).

    python -m member4_chatbot.eval.import_testset   # after the team edits eval/qa_testset.xlsx
    python -m member4_chatbot.eval.run_eval

Every question goes through the real API, `POST /api/m4/chat`, on the whole SoilMind app (FastAPI TestClient:
MongoDB, Member 2 and Member 3 included, all on this machine). Measures:
- intent accuracy (expected_intent)
- retrieval hit@3: the expected passage, or a passage of the same document (or its other-language twin) that
  contains all key facts, among the top 3 retrieved
- answer correctness: every key fact appears in the answer ('answer' rows); for 'live' rows, every live value
  named in live_checks appears, read from /api/m3/kpis and /api/m2/recommendations at run time
- citation correctness: the expected document (or its twin) is cited
- refusal precision / recall (refused = path "refused")
- hallucinated-number rate: a number in the answer that is in no cited passage and no live API value
- latency on CPU (server-side, and the client's wall time)
Writes eval/results/results.csv and eval/results/metrics.json. The report is labelled "unchecked" while any row
of the set still has review_status "needs human check".
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from datetime import UTC, datetime

import pandas as pd

from member4_chatbot.chat.extractive import numbers, unsupported_numbers
from member4_chatbot.chat.pipeline import fmt_date, fmt_time, num
from member4_chatbot.paths import EVAL_DIR, PASSAGES_CSV

RESULTS = EVAL_DIR / "results"
DASH = re.compile(r"\s*[–—-]\s*")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", DASH.sub("-", s)).strip().lower()


def twin(doc: str) -> str:
    return doc[:-3] if doc.endswith("_si") else (doc[:-3] + "_si" if doc.endswith("_en") else doc + "_si")


def live_expectations(client) -> tuple[dict[str, list[str]], str]:
    """Strings a correct live answer must contain, per check name, from the live APIs (and their raw JSON)."""
    k = client.get("/api/m3/kpis").json()
    recs = client.get("/api/m2/recommendations", params={"node": "plot"}).json()
    av, h = k["kpis"]["averages"], k["kpis"]["health"]
    exp = {"health_score": [num(h["score"])], "avg_n": [num(av["n"])], "avg_p": [num(av["p"], 2)],
           "avg_k": [num(av["k"])], "avg_ph": [num(av["ph"], 2)], "avg_moisture": [num(av["moisture"])]}
    for product, key in (("urea", "plan_urea"), ("dolomite", "plan_dolomite"), ("TSP", "plan_tsp"), ("MOP", "plan_mop")):
        acts = sorted((a for a in recs["actions"] if a["product"] == product), key=lambda a: a.get("window_start") or a["date"])
        if acts:
            a = acts[0]
            start = a.get("window_start") or a["date"]
            exp[key] = [num(a["kg_per_acre"], 0), start]  # date checked in either language format below
    # dates / times the templates render from API timestamps (Colombo time, both languages) are not "new" numbers
    stamps = [k.get("last_reading_at")] if k.get("last_reading_at") else []
    rendered = [fmt_time(t, lang) for t in stamps for lang in ("en", "si")]
    rendered += [fmt_date(a.get(f) or a["date"], lang) for a in recs["actions"] for f in ("window_start", "window_end")
                 for lang in ("en", "si")]
    return exp, json.dumps([k, recs]) + " " + " ".join(rendered)


def main(argv: list[str] | None = None) -> dict:
    from fastapi.testclient import TestClient

    from member4_chatbot.paths import MEMBER_DIR

    if str(MEMBER_DIR.parent) not in sys.path:  # backend/, where main.py lives
        sys.path.insert(0, str(MEMBER_DIR.parent))
    from main import app  # the whole SoilMind API

    ts = pd.read_csv(EVAL_DIR / "qa_testset.csv", dtype=str, keep_default_na=False)
    passages = pd.read_csv(PASSAGES_CSV, keep_default_na=False).set_index("id")
    rows = []
    with TestClient(app) as client:
        live_exp, live_json = live_expectations(client)
        client.post("/api/m4/chat", json={"session_id": "warm-up", "message": "hello", "persona": "farmer"})
        for q in ts.itertuples():
            t0 = time.perf_counter()
            r = client.post("/api/m4/chat", json={"session_id": f"eval-{q.id}", "message": q.question,
                                                  "persona": "farmer", "ui_lang": q.lang})
            wall = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            a = r.json()
            ans = norm(a["answer"])
            facts = [f.strip() for f in q.key_facts.split(";") if f.strip()]
            ok_docs = {q.expected_source_doc, twin(q.expected_source_doc)}
            top3 = [d["passage_id"] for d in a["retrieved"][:3]]

            def good_passage(pid: str) -> bool:
                if pid == q.expected_passage_id:
                    return True
                if pid not in passages.index or passages.loc[pid, "doc"] not in ok_docs:
                    return False
                text = norm(passages.loc[pid, "text"])
                return all(norm(f) in text for f in facts)

            row = {"id": q.id, "lang": q.lang, "behaviour": q.expected_behaviour, "question": q.question,
                   "expected_intent": q.expected_intent, "intent": a["intent"], "confidence": a["confidence"],
                   "path": a["path"], "refused": a["path"] == "refused", "answer": a["answer"],
                   "retrieved_top3": ";".join(top3), "cited_docs": ";".join(c["doc"] for c in a["citations"]),
                   "retrieval_score": a["retrieval_score"], "latency_ms": a["latency_ms"], "wall_ms": round(wall, 1),
                   "review_status": q.review_status}
            row["intent_ok"] = a["intent"] == q.expected_intent
            if q.expected_behaviour == "answer":
                found = [norm(f) in ans for f in facts]
                row["fact_recall"] = sum(found) / len(found)
                row["answer_ok"] = all(found)
                row["hit3"] = any(good_passage(p) for p in top3)
                row["citation_ok"] = any(c["doc"] in ok_docs for c in a["citations"])
            elif q.expected_behaviour == "live":
                checks = [c.strip() for c in q.live_checks.split(";") if c.strip()]
                found = []
                for c in checks:
                    want = live_exp.get(c)
                    if not want:
                        found.append(False)
                        continue
                    vals = [norm(want[0])] + ([norm(fmt_date(want[1], "en")), norm(fmt_date(want[1], "si"))] if len(want) > 1 else [])
                    found.append(vals[0] in ans and (len(vals) == 1 or vals[1] in ans or vals[2] in ans))
                row["fact_recall"] = sum(found) / len(found) if found else 0.0
                row["answer_ok"] = bool(found) and all(found)
                row["citation_ok"] = bool(a["citations"])
            else:
                row["answer_ok"] = a["path"] == "refused"
            cited = " ".join(passages.loc[c["passage_id"], "text"] for c in a["citations"]
                             if c["passage_id"] and c["passage_id"] in passages.index)
            ref = a.get("escalation_ref") or ""
            extra = unsupported_numbers(a["answer"].replace(ref, ""), cited, live_json, q.question)
            row["unsupported_numbers"] = ";".join(sorted(extra))
            rows.append(row)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "results.csv", index=False, encoding="utf-8")
    metrics = summarise(df, ts)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics["overall"], indent=2))
    print("status:", metrics["status"])
    return metrics


def summarise(df: pd.DataFrame, ts: pd.DataFrame) -> dict:
    def block(d: pd.DataFrame) -> dict:
        ans, live, ref = d[d.behaviour == "answer"], d[d.behaviour == "live"], d[d.behaviour == "refuse"]
        tp = int((ref.refused).sum())
        fp = int((d.refused & (d.behaviour != "refuse")).sum())
        lat = d.latency_ms.astype(float)
        return {
            "n": int(len(d)),
            "intent_accuracy": round(float(d.intent_ok.mean()), 4),
            "retrieval_hit3": round(float(ans.hit3.mean()), 4) if len(ans) else None,
            "answer_correct_kb": round(float(ans.answer_ok.mean()), 4) if len(ans) else None,
            "key_fact_recall_kb": round(float(ans.fact_recall.mean()), 4) if len(ans) else None,
            "citation_correct_kb": round(float(ans.citation_ok.mean()), 4) if len(ans) else None,
            "answer_correct_live": round(float(live.answer_ok.mean()), 4) if len(live) else None,
            "refusal_precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "refusal_recall": round(tp / len(ref), 4) if len(ref) else None,
            "hallucinated_number_rate": round(float((d.unsupported_numbers != "").mean()), 4),
            "latency_ms_median": round(statistics.median(lat), 1),
            "latency_ms_p95": round(float(lat.quantile(0.95)), 1),
            "wall_ms_median": round(float(d.wall_ms.median()), 1),
        }

    checked = int(ts.review_status.str.startswith("checked").sum())
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": ("checked" if checked == len(ts) else f"UNCHECKED: {checked} of {len(ts)} questions human-checked"),
        "questions": int(len(ts)), "human_checked": checked,
        "overall": block(df), "en": block(df[df.lang == "en"]), "si": block(df[df.lang == "si"]),
        "by_behaviour": {b: int(n) for b, n in df.behaviour.value_counts().items()},
        "device": "cpu",
    }


if __name__ == "__main__":
    main(sys.argv[1:])
