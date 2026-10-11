"""The answer pipeline (Phase 6, extractive version: no text generator).

    question -> language -> intent model -> greeting | refusal | live template | retrieval -> extract -> answer

1. Language: Sinhala script -> answer in Sinhala; anything else -> English (Singlish follows the UI language).
2. Follow-ups ("and how much?"): a short message is joined to the previous question (last 3 turns kept).
3. Intent model (ours). `greeting_help` -> fixed bilingual reply. `out_of_scope` (including a top probability
   below the validation-chosen threshold) -> refusal + escalation.
4. `my_field_status` -> bilingual template filled from Member 3's KPIs / alerts and Member 2's advice.
5. Everything else -> retriever (ours). Best passage score below the validation-chosen threshold -> refusal +
   escalation. Otherwise the best 1-3 sentences of the best passage are copied (extractive.py); a passage in
   the answer's language is preferred when it scores almost as well. If only the other language has it, the
   sentences are shown as they are, after a short note.
6. Fertilizer / liming intents also get one "for your field" line from Member 2's advice (template, exact
   amounts and dates from the API).
7. Safety: every number in the document part must appear in the cited passage, every number in the field part
   comes from the API. Checked on every answer; a violation is logged and the answer falls back to the single
   best sentence.
8. Citations (title, page, URL, snippet) are always returned.
9. Phase 7 cross-links: a request may carry a `context` from the twin ("Ask SoilMind": a zone and layer) or the
   Advisory page ("Why?": one recommendation). The answer is then about exactly that, from the live APIs. A field
   question that names a zone or a sensor ("the south-east part", "A3-06") is answered about that zone too.
   `map_focus` lists the zones / sensors an answer talks about, for the "Show on the map" links.
10. Graceful degradation: if the sentence reranker fails, the retriever's own sentence scores pick the sentences
   of the best passage, with a note; if the intent model fails, the question goes to retrieval (with its
   refusal threshold). `degraded` names what was replaced.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from member4_chatbot import models
from member4_chatbot.chat import text as T
from member4_chatbot.chat.extractive import extract, numbers, unsupported_numbers
from member4_chatbot.chat.live import Live, LiveError
from member4_chatbot.chat.profile import Profile, plot_profile, question_profile
from member4_chatbot.chat.rerank import Reranker, choose
from member4_chatbot.chat.retrieval import PassageRanker
from member4_chatbot.chat.store import Escalations, Sessions
from member4_chatbot.lang import detect_language
from member4_chatbot.paths import ANSWER_DIR, INTENT_DIR, PASSAGES_CSV, RERANKER_DIR

log = logging.getLogger("soilmind.m4")
COLOMBO = ZoneInfo("Asia/Colombo")
LIVE_INTENTS_WITH_DOCS = {"fertilizer_amount", "fertilizer_timing", "liming_ph"}
PROFILE_INTENTS = {"fertilizer_amount", "fertilizer_timing", "liming_ph", "my_field_status"}
NODE_ID = re.compile(r"\bA\d-\d{2}\b", re.I)
FOLLOW_UP = re.compile(r"^(and|what about|how about|also|then|ඒ|එතකොට|එහෙනම්|ඒ වගේම|තව)\b", re.I)


@dataclass
class AnswerConfig:
    retrieval_threshold: float
    max_sentences: int = 3
    margin: float = 0.08
    same_language_delta: float = 0.05
    top_k: int = 5
    retrieval_mode: dict | None = None  # {"en": "dense" | "rrf", "si": ...}, chosen on validation
    source: str = "default"

    @classmethod
    def load(cls) -> "AnswerConfig":
        f = ANSWER_DIR / "config.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            return cls(**{k: d[k] for k in ("retrieval_threshold", "max_sentences", "margin",
                                            "same_language_delta", "top_k", "retrieval_mode")}, source=str(f.name))
        log.warning("artifacts/answer/config.json missing (run notebook 03); using an untuned default")
        return cls(retrieval_threshold=0.35, source="default (notebook 03 not run)")


def cosine_scorer(question: str, texts: list[str]) -> list[float]:
    """Our retriever's cosine similarity between the question and each sentence (a sentence scorer)."""
    if not texts:
        return []
    e = models.embed([question] + list(texts))
    return [float(x) for x in e[1:] @ e[0]]


def answer_language(q_lang: str, ui_lang: str | None) -> str:
    if q_lang == "si":
        return "si"
    if q_lang == "si_latn" and ui_lang == "si":
        return "si"
    return "en"


def is_follow_up(message: str) -> bool:
    return bool(FOLLOW_UP.match(message.strip())) or len(message.split()) <= 3


def fmt_date(iso: str, lang: str) -> str:
    d = date.fromisoformat(iso[:10])
    return d.strftime("%d %b %Y").lstrip("0") if lang == "en" else d.isoformat()


def fmt_time(iso: str, lang: str) -> str:
    t = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(COLOMBO)
    return t.strftime("%d %b %Y, %H:%M").lstrip("0") if lang == "en" else t.strftime("%Y-%m-%d %H:%M")


def num(v: float, digits: int = 1) -> str:
    s = f"{v:.{digits}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


class ChatPipeline:
    def __init__(self, live: Live | None, escalations: Escalations, sessions: Sessions | None = None,
                 config: AnswerConfig | None = None):
        self.live = live
        self.escalations = escalations
        self.sessions = sessions or Sessions()
        self.config = config or AnswerConfig.load()
        p = pd.read_csv(PASSAGES_CSV, keep_default_na=False)
        self.passages = p.set_index("id")
        self.intent_threshold = json.loads((INTENT_DIR / "inference.json").read_text())["threshold"]
        self._ranker: PassageRanker | None = None
        f = RERANKER_DIR / "config.json"
        self.rerank_config = json.loads(f.read_text(encoding="utf-8")) if f.exists() and (RERANKER_DIR / "model").exists() else None
        self._reranker: Reranker | None = None

    @property
    def ranker(self) -> PassageRanker:
        if self._ranker is None:
            _, collection = models._retriever()
            self._ranker = PassageRanker.from_chroma(self.passages.reset_index(), collection)
        return self._ranker

    @property
    def reranker(self):
        """The sentence scorer chosen on validation in notebook 05 (config "picker"): our fine-tuned cross-encoder,
        the open pretrained one, or our retriever's cosine similarity. None if notebook 05 has not been run."""
        if self.rerank_config and self._reranker is None:
            picker = self.rerank_config.get("picker", "fine_tuned_reranker")
            if picker == "retriever_cosine":
                self._reranker = cosine_scorer
            else:
                path = RERANKER_DIR / "model" if picker == "fine_tuned_reranker" else self.rerank_config["base_model"]
                self._reranker = Reranker(path, self.rerank_config.get("max_length", 256))
        return self._reranker

    def _rank(self, query: str, lang: str, profile: Profile | None, boost: float):
        q_emb = models.embed([query])[0]
        hits = self.ranker.rank_profiled(q_emb, query, self.mode(lang), self.config.top_k, profile, boost)
        return q_emb, hits, self.ranker.best_dense(q_emb)

    async def field_profile(self, message: str, intent: str) -> Profile | None:
        """Plot profile (Member 2's advice + season) overridden by what the question says; only for field and
        fertilizer questions."""
        if intent not in PROFILE_INTENTS:
            return None
        plot = Profile()
        if self.live is not None:
            try:
                plot = plot_profile(await self.live.recommendations(), await self.live.season())
            except LiveError as exc:
                log.warning("plot profile unavailable: %s", exc)
        return plot.merged_with(question_profile(message))

    def warm_up(self) -> None:
        """Load both models and the passage index now (first chat would otherwise wait ~30 s)."""
        models.predict_intent("hello")
        _ = self.ranker
        _ = self.reranker

    def mode(self, lang: str) -> str:
        return (self.config.retrieval_mode or {}).get(lang, "dense")

    # ------------------------------------------------------------------ helpers
    def citation(self, hit, snippet: str | None = None) -> dict:
        row = self.passages.loc[hit.passage_id]
        return {"passage_id": hit.passage_id, "title": row["title"] or row["doc_title"], "doc": row["doc"],
                "doc_title": row["doc_title"], "page": int(row["page"]), "url": row["url"], "lang": row["lang"],
                "score": round(hit.score, 4), "snippet": (snippet or row["text"])[:280]}

    async def refuse(self, kind: str, lang: str, record: dict) -> tuple[list[str], str]:
        ref = await self.escalations.add({**record, "reason": kind})
        msg = (T.REFUSE_OUT_OF_SCOPE if kind == "out_of_scope" else T.REFUSE_NOT_FOUND)[lang].format(ref=ref)
        return [msg], ref

    def focus_in(self, message: str) -> dict | None:
        """A zone or sensor the question names ("south-east part", "ගිනිකොන", "A3-06"), else None."""
        m = message.lower()
        nodes = sorted({x.upper() for x in NODE_ID.findall(message)})
        zone = next((z for z, words in T.ZONE_WORDS.items() if any(w in m for w in words)), None)
        return {"zone": zone, "nodes": nodes, "variable": None} if zone or nodes else None

    @staticmethod
    def zone_alerts(alerts: list[dict], focus: dict) -> list[dict]:
        """Alerts about the focused zone / sensors: the focused variable first, then critical first."""
        hit = [a for a in alerts if (focus.get("zone") and a.get("zone") == focus["zone"])
               or set(a.get("nodes") or []) & set(focus.get("nodes") or [])]
        var = focus.get("variable")
        return sorted(hit, key=lambda a: (var not in (None, "health") and a.get("variable") != var,
                                          a.get("severity") != "crit"))

    @staticmethod
    def alert_focus(a: dict) -> dict:
        """A map link for an alert's zone: its sensors and the layer of its variable."""
        var = T.VARIABLE.get(a.get("variable") or "", {})
        return {"kind": "zone", "nodes": list(a.get("nodes") or []), "zone": a.get("zone"),
                "variable": a.get("variable"),
                "label": {lg: " · ".join(x for x in (a["zone_label"][lg], var.get(lg, "")) if x) for lg in ("en", "si")}}

    async def field_status(self, lang: str, persona: str, product: str | None = None,
                           focus: dict | None = None, out: dict | None = None) -> tuple[list[str], list[dict]]:
        """The "my field" answer. With a `focus` (zone / sensors) it leads with that zone's alerts and the action
        for its problem. `out["map_focus"]` gets the zones and sensors the answer mentions."""
        out = out if out is not None else {}
        out.setdefault("map_focus", [])
        k = await self.live.kpis()
        kp = k["kpis"]
        if not kp.get("health") or not kp.get("averages"):  # every node offline: no live map, no health score
            when = fmt_time(k["last_reading_at"], lang) if k.get("last_reading_at") else "-"
            return [T.NO_RECENT_READINGS[lang].format(time=when)], []
        lim = kp["health"]["limiting"] or "n"
        worst = kp["worst"].get(lim) or {}
        av = kp["averages"]
        lines = [T.FIELD_STATUS[lang].format(
            plot=await self.plot_name(), score=num(kp["health"]["score"]), grade=T.GRADE.get(kp["health"]["grade"], {}).get(lang, kp["health"]["grade"]),
            var=T.VARIABLE[lim][lang], node=worst.get("node_id", "-"), value=num(worst.get("value", 0), 2),
            unit=(" " + T.UNIT[lim]) if T.UNIT[lim] and T.UNIT[lim] != "%" else T.UNIT[lim],
            n=num(av["n"]), p=num(av["p"], 2), k=num(av["k"]), ph=num(av["ph"], 2), moisture=num(av["moisture"]),
            time=fmt_time(k["last_reading_at"], lang))]
        values = [{"key": v, "label": T.VARIABLE[v], "value": av[v], "unit": T.UNIT[v], "node": "field mean",
                   "at": k["last_reading_at"]} for v in ("n", "p", "k", "ph", "moisture")]
        values.append({"key": "health", "label": {"en": "health score", "si": "සෞඛ්‍ය ලකුණු"},
                       "value": kp["health"]["score"], "unit": "/100", "node": "plot", "at": k["last_reading_at"],
                       "status": kp["health"]["grade"]})
        if worst.get("node_id"):
            out["map_focus"].append({"kind": "node", "nodes": [worst["node_id"]], "zone": None, "variable": lim,
                                     "label": {lg: f"{worst['node_id']} · {T.VARIABLE[lim][lg]}" for lg in ("en", "si")}})
        key = "title_farmer" if persona == "farmer" else "title_officer"
        products: list[str] = []
        try:
            alerts = (await self.live.alerts())["alerts"]
        except LiveError:
            alerts = []
        if focus:
            mine = self.zone_alerts(alerts, focus)
            zone = focus.get("zone") or (mine[0]["zone"] if mine else None)
            zname = (next((a["zone_label"][lang] for a in alerts if a.get("zone") == zone), None)
                     or T.ZONE_NAME.get(zone or "", {}).get(lang) or ", ".join(focus.get("nodes") or []) or "-")
            if mine:
                head = [T.ZONE_ALERT[lang].format(zone=zname, title=a[key][lang].rstrip("."),
                                                  body=a["body_officer"][lang]) for a in mine[:2]]
                out["map_focus"] = [self.alert_focus(a) for a in mine[:2]] + out["map_focus"]
                # the product named (if any), then the actions for the zone's own problems (urea for N, lime for pH)
                products = list(dict.fromkeys([product] * bool(product) + [
                    T.VARIABLE_PRODUCT[a["variable"]] for a in mine[:2]
                    if a.get("variable") in T.VARIABLE_PRODUCT and a.get("severity") == "crit"]))
            else:
                head = [T.ZONE_NO_ALERT[lang].format(zone=zname)]
                if focus.get("nodes"):
                    out["map_focus"].insert(0, {"kind": "zone" if zone else "node", "nodes": list(focus["nodes"]),
                                                "zone": zone, "variable": focus.get("variable"),
                                                "label": {lg: zname for lg in ("en", "si")}})
            lines = head + lines
        else:
            crit = [a for a in alerts if a.get("severity") == "crit"][:2] or alerts[:1]
            if crit:
                lines.append(T.ALERTS_LINE[lang].format(titles="; ".join(a[key][lang] for a in crit)))
                out["map_focus"] += [self.alert_focus(a) for a in crit if a.get("nodes") and a.get("variable")]
        for prod in products or [product]:  # the product asked about (None: the most urgent action)
            plan_line, plan_values = await self.plan_line(lang, persona, product=prod)
            if plan_line:
                if focus:  # the zone's problems, then what to do about them, then the whole-field summary
                    lines.insert(len(lines) - 1, plan_line)
                else:
                    lines.append(plan_line)
                values += plan_values
        return lines, values

    async def explain_action(self, action_id: str | None, product: str | None, lang: str, persona: str,
                             out: dict) -> list[str]:
        """"Why?" from the Advisory page: what to do, why (Member 2's reason), where, and the expected effect, with
        the recommendation's own RRDI / DOA sources as citations."""
        recs = await self.live.recommendations()
        acts = sorted(recs.get("actions", []), key=lambda x: x.get("window_start") or x.get("date") or "")
        a = next((x for x in acts if x.get("id") == action_id), None)
        if a is None and product:  # the advice changed since the page was opened: the next action of that product
            a = next((x for x in acts if x["product"] == product), None)
        if a is None:
            return [T.ACTION_GONE[lang]]
        start, end = a.get("window_start") or a["date"], a.get("window_end") or a["date"]
        tmpl = T.ACTION_WHAT_ONE_DAY if start == end else T.ACTION_WHAT
        lines = [tmpl[lang].format(title=a["title"][lang], plot=await self.plot_name(), instruction=a["instruction"][lang],
                                   start=fmt_date(start, lang), end=fmt_date(end, lang),
                                   per_acre=num(a.get("kg_per_acre") or 0, 0))]
        reason = (a.get("reason") or {}).get(lang) or ""
        if persona == "farmer" and (a.get("farmer_text") or {}).get(lang):
            reason = a["farmer_text"][lang] + (f" ({reason})" if reason else "")
        if reason:
            lines.append(T.ACTION_WHY[lang].format(reason=reason))
        z = a.get("zone") or {}
        if z and not z.get("whole_field"):
            lines.append(T.ACTION_WHERE[lang].format(zone=z["label"][lang], area=num(z.get("area_acres", 0), 2),
                                                     nodes=", ".join(z.get("nodes") or [])))
            var = next((v for v, p in T.VARIABLE_PRODUCT.items() if p == a["product"]), None)
            out["map_focus"].append({"kind": "zone", "nodes": list(z.get("nodes") or []), "zone": z.get("direction"),
                                     "variable": var, "label": {lg: z["label"][lg] for lg in ("en", "si")}})
        eff = a.get("expected_effect")
        if eff:
            ch = eff["change_median"]
            unit = "" if eff.get("unit") in ("pH", "", None) else f" {eff['unit']}"
            lines.append(T.ACTION_EFFECT[lang].format(
                var=T.VARIABLE.get(eff["target"], {}).get(lang, eff["target"]),
                change=f"{'+' if ch >= 0 else ''}{num(ch, 2 if eff['target'] == 'ph' else 1)}", unit=unit,
                date=fmt_date(eff["at_date"], lang)))
        out["live_values"] = [{"key": f"plan:{a['id']}", "label": a["title"], "value": a.get("kg_total"), "unit": "kg",
                               "node": z.get("label", {}).get(lang, "plot"), "at": start, "status": a.get("priority")}]
        cites = [{"passage_id": None, "title": src["title"], "doc": "advice", "doc_title": T.ADVICE_SOURCE[lang],
                  "page": 0, "url": src["url"], "lang": lang, "score": None, "snippet": reason[:280]}
                 for src in a.get("sources", [])]
        out["citations"] = cites or [{"passage_id": None, "title": T.ADVICE_SOURCE[lang], "doc": "advice",
                                      "doc_title": "/api/m2/recommendations", "page": 0, "url": "/advisory",
                                      "lang": lang, "score": None, "snippet": reason[:280]}]
        return lines

    async def plan_line(self, lang: str, persona: str, product: str | None,
                        intent: str | None = None) -> tuple[str | None, list[dict]]:
        recs = await self.live.recommendations()
        actions = sorted(recs.get("actions", []), key=lambda a: (a.get("window_start") or a.get("date") or ""))
        if product:
            actions = [a for a in actions if a["product"] == product]
        elif intent == "liming_ph":
            actions = [a for a in actions if a["product"] == "dolomite"]
        elif intent in ("fertilizer_amount", "fertilizer_timing"):
            actions = [a for a in actions if a["product"] != "dolomite"]
        if not actions:
            if product or intent:
                return T.NO_ACTION[lang].format(product=product or ("lime" if intent == "liming_ph" else "fertilizer")), []
            return None, []
        crit = [a for a in actions if a.get("priority") == "crit"]
        a = (crit or actions)[0] if not product and not intent else actions[0]
        start, end = a.get("window_start") or a["date"], a.get("window_end") or a["date"]
        reason = (a.get("farmer_text") if persona == "farmer" else a.get("reason")) or {}
        tmpl = T.NEXT_ACTION_ONE_DAY if start == end else T.NEXT_ACTION
        line = tmpl[lang].format(plot=await self.plot_name(), area=num(recs.get("plot_area_acres", 0), 2),
                                 instruction=a["instruction"][lang],
                                 start=fmt_date(start, lang), end=fmt_date(end, lang),
                                 per_acre=num(a.get("kg_per_acre") or 0, 0), reason=reason.get(lang, ""))
        value = {"key": f"plan:{a['id']}", "label": a["title"], "value": a.get("kg_total"), "unit": "kg",
                 "node": a.get("zone", {}).get("label", {}).get(lang, "plot"), "at": start,
                 "status": a.get("priority")}
        return line, [value]

    async def plot_name(self) -> str:
        """The plot's name from Member 3 (/api/m3/field); its id if the name is missing."""
        try:
            f = await self.live.field()
            return f.get("name") or f.get("field_id") or "-"
        except LiveError:
            return "-"

    def product_in(self, message: str) -> str | None:
        m = message.lower()
        for product, words in T.PRODUCT_WORDS.items():
            if any(w in m for w in words):
                return product
        return None

    # ------------------------------------------------------------------ main
    async def answer(self, session_id: str, message: str, persona: str = "farmer", ui_lang: str | None = None,
                     context: dict | None = None) -> dict:
        t0 = time.perf_counter()
        message = message.strip()
        q_lang = detect_language(message)
        lang = answer_language(q_lang, ui_lang)
        history = self.sessions.history(session_id)
        query, follow_up_of = message, None
        if history and is_follow_up(message):
            follow_up_of = history[-1]["question"]
            query = f"{follow_up_of} {message}"

        degraded: list[str] = []
        try:
            pred = await asyncio.to_thread(models.predict_intent, message)  # CPU work off the event loop
            if follow_up_of and pred.intent == "out_of_scope":
                pred = await asyncio.to_thread(models.predict_intent, query)  # "and how much?" means nothing alone
        except Exception as exc:  # noqa: BLE001 - degrade: without the intent model, retrieval (and its threshold) decides
            log.error("intent model failed (%s); answering from retrieval", exc)
            degraded.append("intent")
            pred = models.IntentPrediction("unknown", "unavailable", 0.0, q_lang, {})
        out = {"question": message, "question_lang": q_lang, "answer_lang": lang, "intent": pred.intent,
               "raw_intent": pred.raw_intent, "confidence": round(pred.confidence, 4),
               "intent_threshold": self.intent_threshold, "retrieval_threshold": self.config.retrieval_threshold,
               "retrieval_score": None, "citations": [], "retrieved": [], "live_values": [], "note": None,
               "escalated": False, "escalation_ref": None, "follow_up_of": follow_up_of, "safety": [],
               "sinhala_verified": T.SINHALA_VERIFIED, "persona": persona, "full_passage": None, "highlights": [],
               "field_profile": None, "selection": None, "context": context, "map_focus": [], "degraded": degraded}
        record = {"session_id": session_id, "question": message, "lang": q_lang, "persona": persona,
                  "intent": pred.intent, "confidence": round(pred.confidence, 4)}
        lines: list[str] = []

        ctx = context or {}
        if ctx.get("kind") == "recommendation" and self.live is not None:  # "Why?" from the Advisory page
            try:
                lines = await self.explain_action(ctx.get("action_id"), ctx.get("product"), lang, persona, out)
                out["path"] = "template"
            except LiveError as exc:
                log.warning("live advice unavailable: %s", exc)
                lines, out["escalation_ref"] = await self.refuse("not_found", lang, {**record, "detail": str(exc)})
                lines = [T.LIVE_UNAVAILABLE[lang]] + lines
                out["path"], out["escalated"] = "refused", True
        elif pred.intent == "greeting_help" and not ctx:
            out["path"], lines = "greeting", [T.GREETING[lang]]
        elif pred.intent == "out_of_scope" and not ctx:
            lines, out["escalation_ref"] = await self.refuse("out_of_scope", lang, record)
            out["path"], out["escalated"] = "refused", True
        elif pred.intent == "my_field_status" or ctx.get("kind") == "zone":  # twin's "Ask SoilMind" sends a zone
            try:
                if self.live is None:
                    raise LiveError("no live client")
                focus = ({"zone": ctx.get("zone"), "nodes": ctx.get("nodes") or [], "variable": ctx.get("variable")}
                         if ctx.get("kind") == "zone" else self.focus_in(message))
                lines, out["live_values"] = await self.field_status(lang, persona, self.product_in(message), focus, out)
                out["path"] = "template"
                out["citations"] = [{"passage_id": None, "title": T.LIVE_SOURCE[lang], "doc": "live",
                                     "doc_title": "/api/m3/kpis, /api/m3/alerts, /api/m2/recommendations",
                                     "page": 0, "url": "/twin", "lang": lang, "score": None,
                                     "snippet": "Your plot: sensors, health score and advice"}]
            except LiveError as exc:
                log.warning("live data unavailable: %s", exc)
                lines, out["escalation_ref"] = await self.refuse("not_found", lang, {**record, "detail": str(exc)})
                lines = [T.LIVE_UNAVAILABLE[lang]] + lines
                out["path"], out["escalated"] = "refused", True
        else:
            lines = await self.document_answer(query, message, lang, persona, pred.intent, out, record)

        out["answer_lines"] = lines
        out["answer"] = "\n".join(([out["note"]] if out["note"] else []) + lines)
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self.sessions.add(session_id, {"question": query, "intent": out["intent"], "path": out["path"]})
        return out

    async def document_answer(self, query: str, message: str, lang: str, persona: str, intent: str, out: dict,
                              record: dict) -> list[str]:
        profile = await self.field_profile(message, intent)
        boost = (self.rerank_config or {}).get("profile_boost", 0) if profile else 0
        out["field_profile"] = {k: v for k, v in vars(profile).items()} if profile else None
        q_emb, hits, top = await asyncio.to_thread(self._rank, query, lang, profile, boost)
        out["retrieval_score"] = round(top, 4)
        out["retrieved"] = [{"passage_id": h.passage_id, "title": self.citation(h)["title"], "score": round(h.score, 4),
                             "lang": h.lang, "page": h.page, "doc": h.doc} for h in hits]
        if top < self.config.retrieval_threshold:
            lines, out["escalation_ref"] = await self.refuse("not_found", lang, {**record, "retrieval_score": top})
            out["path"], out["escalated"] = "refused", True
            return lines

        c = None
        if self.rerank_config:  # our sentence reranker (notebook 05) over the top passages
            try:
                rc = self.rerank_config
                cands = [{"id": h.passage_id, "text": h.text, "title": self.passages.loc[h.passage_id, "title"],
                          "lang": h.lang} for h in hits[:rc["n_passages"]]]
                c = await asyncio.to_thread(choose, query, cands, self.reranker, answer_lang=lang,
                                            max_units=rc["max_units"], margin=rc["margin"], lang_delta=rc["lang_delta"],
                                            age=profile.age if profile else None, age_bonus=rc.get("age_bonus", 0.0))
            except Exception as exc:  # noqa: BLE001 - degrade to the retriever's own sentence scores, with a note
                log.error("sentence reranker failed (%s); using the retriever's sentence scores", exc)
                out["degraded"].append("reranker")
        if c is not None:
            chosen = next(h for h in hits if h.passage_id == c.passage_id)
            doc_lines, selected = c.lines, c.selected
            out["selection"] = self.rerank_config.get("picker", "fine_tuned_reranker")
        else:  # notebook 03's selection: retriever cosine per sentence
            lead = hits[0].score
            near = [h for h in hits if h.score >= lead - self.config.same_language_delta]
            chosen = next((h for h in near if h.lang == lang), hits[0])
            ex = await asyncio.to_thread(extract, chosen.passage_id, chosen.text,
                                         self.passages.loc[chosen.passage_id, "title"], q_emb, models.embed,
                                         max_sentences=self.config.max_sentences, margin=self.config.margin)
            doc_lines, selected = ex.lines, ex.lines
            out["selection"] = "retriever_cosine"
        row = self.passages.loc[chosen.passage_id]
        bad = unsupported_numbers("\n".join(doc_lines), row["text"])
        if bad:  # cannot happen for copied text; kept as a guard and logged
            log.error("safety: numbers %s not in passage %s", bad, chosen.passage_id)
            out["safety"].append({"check": "numbers_in_source", "unsupported": sorted(bad)})
            doc_lines = doc_lines[:1]
        if chosen.lang != lang:
            out["note"] = T.OTHER_LANGUAGE_NOTE[lang]
        if "reranker" in out["degraded"]:
            out["note"] = " ".join(x for x in (T.DEGRADED_NOTE[lang], out["note"]) if x)
        out["path"] = "extractive"
        out["full_passage"] = row["text"]
        out["highlights"] = sorted({n for line in selected for n in numbers(line)}, key=lambda n: (-len(n), n))
        out["citations"] = [self.citation(chosen, " ".join(doc_lines))]
        seen = {chosen.doc}
        for h in hits:
            if h.doc not in seen and len(out["citations"]) < 3 and h.score >= self.config.retrieval_threshold:
                out["citations"].append(self.citation(h))
                seen.add(h.doc)

        lines = list(doc_lines)
        if intent in LIVE_INTENTS_WITH_DOCS and self.live is not None:
            try:
                plan, values = await self.plan_line(lang, persona, self.product_in(message), intent)
                if plan:
                    lines.append(plan)
                    out["live_values"] = values
                    out["path"] = "extractive+template"
            except LiveError as exc:
                log.warning("live advice unavailable: %s", exc)
                lines.append(T.LIVE_UNAVAILABLE[lang])
        return lines


def response_dict(r: dict) -> dict:
    """Plain JSON-able copy (for logging / evaluation)."""
    return json.loads(json.dumps(r, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)))
