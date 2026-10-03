"""Folder layout of Member 4's data and artifacts (all relative to backend/member4-chatbot/)."""

from pathlib import Path

MEMBER_DIR = Path(__file__).resolve().parent.parent
PKG_DIR = MEMBER_DIR / "member4_chatbot"
DATA_DIR = MEMBER_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "docs" / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
TRAINING_SOURCES_DIR = TRAINING_DIR / "sources"
CHROMA_DIR = DATA_DIR / "chroma"
ARTIFACTS_DIR = MEMBER_DIR / "artifacts"
INTENT_DIR = ARTIFACTS_DIR / "intent"
RETRIEVER_DIR = ARTIFACTS_DIR / "retriever"
NOTEBOOKS_DIR = MEMBER_DIR / "notebooks"
SOURCES_YAML = PKG_DIR / "collect" / "sources.yaml"
PASSAGES_CSV = PROCESSED_DIR / "passages.csv"
ANSWER_DIR = ARTIFACTS_DIR / "answer"  # extractive answer settings chosen on validation (notebook 03)
EVAL_DIR = MEMBER_DIR / "eval"
RERANKER_DIR = ARTIFACTS_DIR / "reranker"  # sentence reranker + selection settings (notebook 05)
# Phase 6b voice
VOICE_DATA_DIR = DATA_DIR / "voice"  # term dictionary, TTS terms map, downloaded test audio, TTS cache
VOICE_EVAL_DIR = EVAL_DIR / "voice"  # recording script, recorder page, test sets and results
STT_DIR = ARTIFACTS_DIR / "stt"  # our fine-tuned Whisper (after the team records phrases) + metrics
