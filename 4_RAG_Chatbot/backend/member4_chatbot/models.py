"""Load and run Member 4's two trained models, exactly as the notebooks saved them.

- Intent classifier: artifacts/intent/model/ (fine-tuned xlm-roberta-base) + artifacts/intent/inference.json
  (label order, the confidence threshold chosen on validation, max length).
- Retriever: artifacts/retriever/model/ (fine-tuned paraphrase-multilingual-MiniLM-L12-v2) and the ChromaDB index
  of all passages in data/chroma/ (collection "passages", cosine distance).

Both load lazily, once per process, and run on CPU.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache

from member4_chatbot.lang import detect_language
from member4_chatbot.paths import CHROMA_DIR, INTENT_DIR, RETRIEVER_DIR

CHROMA_COLLECTION = "passages"
_lock = threading.Lock()


@dataclass
class IntentPrediction:
    intent: str  # after the threshold: "out_of_scope" when the model is not confident
    raw_intent: str  # the model's top class
    confidence: float
    lang: str
    scores: dict[str, float]


@dataclass
class Hit:
    passage_id: str
    score: float  # cosine similarity
    text: str
    doc: str
    page: int
    lang: str
    url: str


@lru_cache(maxsize=1)
def _intent():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    meta = json.loads((INTENT_DIR / "inference.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(INTENT_DIR / "model")
    model = AutoModelForSequenceClassification.from_pretrained(INTENT_DIR / "model").eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return tok, model, meta


def predict_intent(text: str) -> IntentPrediction:
    import torch

    with _lock:
        tok, model, meta = _intent()
        enc = tok([text], return_tensors="pt", truncation=True, max_length=meta["max_length"])
        with torch.inference_mode():
            probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
    labels = meta["labels"]
    best = max(range(len(labels)), key=lambda i: probs[i])
    raw = labels[best]
    intent = raw if probs[best] >= meta["threshold"] else "out_of_scope"
    return IntentPrediction(intent, raw, probs[best], detect_language(text), dict(zip(labels, probs)))


@lru_cache(maxsize=1)
def _retriever():
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(RETRIEVER_DIR / "model"), device="cpu")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return model, client.get_collection(CHROMA_COLLECTION)


def embed(texts: list[str]):
    """Normalised embeddings from the fine-tuned retriever model (numpy array, one row per text)."""
    with _lock:
        model, _ = _retriever()
        return model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def search(query: str, k: int = 3) -> list[Hit]:
    return search_embedding(embed([query])[0], k)


def search_embedding(emb, k: int = 3) -> list[Hit]:
    with _lock:
        _, col = _retriever()
        res = col.query(query_embeddings=[list(map(float, emb))], n_results=k)
    hits = []
    for pid, dist, text, meta in zip(res["ids"][0], res["distances"][0], res["documents"][0], res["metadatas"][0]):
        hits.append(Hit(pid, 1.0 - dist, text, meta["doc"], int(meta["page"]), meta["lang"], meta["url"]))
    return hits
