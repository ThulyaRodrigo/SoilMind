"""HTTP routes for the localized agricultural chatbot. Mounted at /api/m4 by backend/main.py.

| Route | What |
|---|---|
| GET  /ping        | router mounted |
| POST /chat        | answer a question (extractive RAG over DOA/RRDI passages, live templates, refusals) |
| POST /prefill     | the question a cross-link (twin "Ask SoilMind", advisory "Why?") puts in the chat box |
| GET  /sources     | the knowledge-base documents and how their text was obtained |
| GET  /escalations | questions the bot refused and forwarded to an agriculture officer |
| GET  /model-info  | model cards + metrics of the intent classifier, retriever and answer step |
| POST /stt         | Sinhala / English speech -> text (Whisper fine-tuned for Sinhala, local) + farming-term dictionary |
| POST /tts         | answer text -> WAV, spoken sentence by sentence (SinhalaVITS / Piper / MMS English, local) |
| POST /tts/parts   | the answer split into parts to speak one after another (playback starts after one sentence) |
| GET  /voice-info  | configured speech models and the voice evaluation metrics |
"""

from __future__ import annotations

import json
import logging
import os

import pandas as pd
import yaml
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from shared.db import get_db
from shared.schemas import (
    ChatRequest,
    ChatResponse,
    EscalationsResponse,
    M4ModelCard,
    M4ModelInfoResponse,
    PingResponse,
    PrefillRequest,
    PrefillResponse,
    SourceDoc,
    SourcesResponse,
    SttResponse,
    TtsPartsResponse,
    TtsRequest,
    VoiceInfoResponse,
)

from member4_chatbot.chat.live import LiveHttp
from member4_chatbot.chat.store import Escalations, Sessions
from member4_chatbot.paths import (ANSWER_DIR, CHROMA_DIR, INTENT_DIR, PASSAGES_CSV, PROCESSED_DIR, RAW_DOCS_DIR,
                                   RERANKER_DIR, RETRIEVER_DIR, VOICE_EVAL_DIR)

log = logging.getLogger("soilmind.m4")
router = APIRouter(tags=["m4"])

_escalations = Escalations(get_db)
_sessions = Sessions()
_pipelines: dict[int, object] = {}


def on_startup() -> None:
    """Load the chat models once, at server start (called by main.py's lifespan, in a worker thread), then load
    and pre-warm the speech models in a background thread (voice/warmup.py) so the first voice question is fast."""
    try:
        from member4_chatbot import models

        models.predict_intent("hello")
        models.embed(["hello"])
        log.info("m4: intent classifier and retriever loaded")
    except Exception as exc:  # noqa: BLE001 - the rest of the app must still start
        log.warning("m4: models not loaded at startup (%s); /chat will try again on first use", exc)
    if os.environ.get("M4_LOAD_VOICE", "1") != "0":  # 0: the voice models load on first use instead
        from member4_chatbot.voice import warmup

        warmup.start_background()


def health_checks() -> list[dict]:
    """Startup / readiness checks for /api/health: our chat models and index, and the speech models (no outside AI
    service is involved, so there is nothing else to check)."""
    from member4_chatbot import models
    from member4_chatbot.voice import warmup

    files = {"intent model": INTENT_DIR / "model", "retriever model": RETRIEVER_DIR / "model",
             "reranker model": RERANKER_DIR / "model", "answer settings": ANSWER_DIR / "config.json",
             "passages": PASSAGES_CSV}
    missing = [k for k, f in files.items() if not f.exists()]
    loaded = models._intent.cache_info().currsize > 0 and models._retriever.cache_info().currsize > 0
    out = [{"name": "chat_models", "member": "m4", "ok": not missing and loaded,
            "detail": (f"missing: {', '.join(missing)}" if missing else
                       "intent classifier, retriever, reranker, answer settings present; "
                       + ("intent + retriever loaded" if loaded else "NOT loaded yet"))}]
    if models._retriever.cache_info().currsize > 0:
        try:
            n = models._retriever()[1].count()
            out.append({"name": "chroma_index", "member": "m4", "ok": n > 0, "detail": f"{n} passages indexed"})
        except Exception as exc:  # noqa: BLE001
            out.append({"name": "chroma_index", "member": "m4", "ok": False, "detail": f"{exc}"})
    else:
        f = CHROMA_DIR / "chroma.sqlite3"
        out.append({"name": "chroma_index", "member": "m4", "ok": f.exists(),
                    "detail": "index files present (not opened yet)" if f.exists() else f"missing {f}"})
    w = warmup.state()
    skipped = os.environ.get("M4_LOAD_VOICE", "1") == "0"
    out.append({"name": "speech_models", "member": "m4", "ok": w["state"] == "ready", "required": not skipped,
                "pending": w["state"] in ("queued", "loading", "warming"),
                "detail": f"{w['state']}" + (f": {w['detail']}" if w["detail"] else "")
                          + (f" {w['timings_ms']}" if w["timings_ms"] else "")
                          + (" (M4_LOAD_VOICE=0: loads on first use)" if skipped else "")})
    return out


def get_pipeline(request: Request):
    from member4_chatbot.chat.pipeline import ChatPipeline

    app = request.app
    if id(app) not in _pipelines:
        _pipelines[id(app)] = ChatPipeline(LiveHttp.for_app(app), _escalations, _sessions)
    return _pipelines[id(app)]


def get_escalations() -> Escalations:
    return _escalations


@router.get("/ping", response_model=PingResponse)
async def ping() -> PingResponse:
    return PingResponse(member="m4", component="Localized agricultural chatbot")


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, pipeline=Depends(get_pipeline)) -> dict:
    try:
        return await pipeline.answer(body.session_id, body.message, body.persona, body.ui_lang,
                                     context=body.context.model_dump() if body.context else None)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"chatbot models are not trained yet: {exc}") from exc


@router.post("/prefill", response_model=PrefillResponse)
async def prefill(body: PrefillRequest) -> PrefillResponse:
    """The question for a cross-link: a zone of the twin map, or one recommendation of the Advisory page. Send it
    back to /chat with the same `context`."""
    from member4_chatbot.chat import text as T

    ctx = body.context.model_dump() if body.context else None
    return PrefillResponse(question=T.prefill(ctx, body.lang), context=body.context, label=T.prefill_label(ctx))


@router.get("/sources", response_model=SourcesResponse)
async def sources() -> SourcesResponse:
    manifest = yaml.safe_load((RAW_DOCS_DIR / "manifest.yaml").read_text(encoding="utf-8"))["documents"]
    log_df = pd.read_csv(PROCESSED_DIR / "extraction_log.csv", keep_default_na=False).set_index("doc")
    passages = pd.read_csv(PASSAGES_CSV, keep_default_na=False)
    docs = []
    for d in manifest:
        method = log_df.loc[d["id"], "method"] if d["id"] in log_df.index else "not_ingested"
        n = int(log_df.loc[d["id"], "passages"]) if d["id"] in log_df.index else 0
        docs.append(SourceDoc(id=d["id"], title=d["title"], url=d["url"], language=d["language"],
                              publisher=d["publisher"], group=d["group"], method=method, passages=n))
    return SourcesResponse(documents=docs, passages_total=len(passages),
                           passages_by_lang={k: int(v) for k, v in passages.lang.value_counts().items()},
                           waiting_for_ocr=sum(d.method == "skipped_needs_ocr" for d in docs))


@router.get("/escalations", response_model=EscalationsResponse)
async def escalations(limit: int = Query(50, ge=1, le=500), store: Escalations = Depends(get_escalations)):
    items = await store.recent(limit)
    return EscalationsResponse(items=items, count=len(items))


def _read(path, default="") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else default


@router.get("/model-info", response_model=M4ModelInfoResponse)
async def model_info() -> M4ModelInfoResponse:
    cards = []
    for name, role, folder in (("intent", "Intent classifier (fine-tuned xlm-roberta-base)", INTENT_DIR),
                               ("retriever", "Passage retriever (fine-tuned multilingual MiniLM + ChromaDB)", RETRIEVER_DIR),
                               ("reranker", "Sentence reranker (fine-tuned multilingual MiniLM cross-encoder)", RERANKER_DIR),
                               ("answer", "Extractive answer step and guardrail (no text generator)", ANSWER_DIR)):
        metrics = json.loads(_read(folder / "metrics.json", "{}"))
        if name == "intent":  # the epoch history and label list are long; keep the summary
            metrics = {k: v for k, v in metrics.items() if k in ("model", "data", "baseline", "xlmr")}
        if name == "reranker":
            metrics = {k: v for k, v in metrics.items() if k in ("config", "results", "best_epoch", "training_pairs")}
        cards.append(M4ModelCard(name=name, role=role, card_markdown=_read(folder / "model_card.md"), metrics=metrics))
    return M4ModelInfoResponse(
        models=cards, answer_config={**json.loads(_read(ANSWER_DIR / "config.json", "{}")),
                                     "reranker": json.loads(_read(RERANKER_DIR / "config.json", "null"))},
        data_warning="All training and test questions are AI-drafted, template-generated or machine-translated and "
                     "NOT human-checked. The team reviews only eval/qa_testset.xlsx.")


MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 30 s of browser Opus is ~0.2 MB; WAV ~1 MB


@router.post("/stt", response_model=SttResponse)
async def stt(audio: UploadFile = File(..., description="webm / ogg / wav / mp3 recording"),
              lang: str = Form("si", pattern="^(si|en)$")) -> SttResponse:
    from member4_chatbot.voice.audio import AudioError
    from member4_chatbot.voice.stt import get_stt

    data = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "recording too large (max 8 MB)")
    try:
        engine = await run_in_threadpool(get_stt)
        tr = await run_in_threadpool(engine.transcribe, data, lang)
    except AudioError as exc:
        raise HTTPException(422, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, f"speech model not available: {exc}") from exc
    return SttResponse(text=tr.text, language=lang, duration=tr.duration, model=tr.model, raw_text=tr.raw_text,
                       corrections=tr.corrections, input_seconds=tr.audio["input_seconds"],
                       trimmed=tr.audio["trimmed"], capped=tr.audio["capped"], latency_ms=tr.latency_ms)


@router.post("/tts", response_class=Response, responses={
    200: {"content": {"audio/wav": {}}, "description": "WAV, 22.05 kHz mono. Headers: X-TTS-Engines, "
          "X-TTS-Cache (hit/miss), X-TTS-Sentences, X-TTS-Failures, X-TTS-Seconds, X-TTS-Ms"}})
async def tts(body: TtsRequest) -> Response:
    from member4_chatbot.voice.tts import TtsUnavailable, get_tts

    try:
        result = await run_in_threadpool(get_tts().speak, body.text)
    except TtsUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if not result.sentences:
        raise HTTPException(422, "nothing to speak in this text")
    headers = {"X-TTS-Engines": ", ".join(result.engines), "X-TTS-Cache": "hit" if result.cached else "miss",
               "X-TTS-Sentences": str(len(result.sentences)), "X-TTS-Failures": str(result.failures),
               "X-TTS-Seconds": f"{result.seconds:.2f}", "X-TTS-Ms": f"{result.total_ms:.0f}",
               "Cache-Control": "private, max-age=86400",
               "Access-Control-Expose-Headers": "X-TTS-Engines, X-TTS-Cache, X-TTS-Sentences, X-TTS-Failures, "
                                                "X-TTS-Seconds, X-TTS-Ms"}
    return Response(result.wav, media_type="audio/wav", headers=headers)


@router.post("/tts/parts", response_model=TtsPartsResponse)
async def tts_parts(body: TtsRequest) -> TtsPartsResponse:
    """Split an answer for playback in parts: the browser asks /tts for part 1, plays it while part 2 is
    synthesised, and so on, so a long Sinhala answer starts speaking after about one sentence."""
    from member4_chatbot.voice.tts_text import speak_parts

    parts, n = speak_parts(body.text)
    if not parts:
        raise HTTPException(422, "nothing to speak in this text")
    return TtsPartsResponse(parts=parts, sentences=n)


@router.get("/voice-info", response_model=VoiceInfoResponse)
async def voice_info() -> VoiceInfoResponse:
    from shared.config import get_settings

    from member4_chatbot.voice import stt as stt_module
    from member4_chatbot.voice.tts import get_tts

    metrics = json.loads(_read(VOICE_EVAL_DIR / "results" / "metrics.json", "{}"))
    return VoiceInfoResponse(stt_model=stt_module.resolve_model(get_settings().stt_model),
                             stt_loaded=stt_module.get_stt.cache_info().currsize > 0,
                             tts=get_tts().status(), metrics=metrics)
