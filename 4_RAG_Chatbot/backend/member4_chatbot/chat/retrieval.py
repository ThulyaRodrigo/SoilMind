"""Passage ranking for the chat: our fine-tuned retriever, optionally fused with BM25.

- "dense": cosine similarity with the fine-tuned MiniLM embeddings stored in ChromaDB.
- "rrf": reciprocal rank fusion of the dense ranking and BM25 (score = 1/(60 + rank_dense) + 1/(60 + rank_bm25)).
  BM25 matches single decisive words exactly (irrigated vs rainfed, a variety code) that embeddings blur.
The mode is chosen per answer language on the validation split (notebook 03, artifacts/answer/config.json).
The refusal gate always uses the best *dense* cosine score, whatever the mode, because that is the score the
threshold was calibrated on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[^\s.,;:!?()\[\]\"'/|\-–*]+")  # same tokenisation as notebook 02
RRF_K = 60


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Ranked:
    passage_id: str
    score: float  # dense cosine similarity
    lang: str
    doc: str
    page: int
    url: str
    text: str


class PassageRanker:
    def __init__(self, passages: pd.DataFrame, ids: list[str], embeddings: np.ndarray):
        order = {pid: i for i, pid in enumerate(ids)}
        known = set(passages.id)
        self.p = passages.set_index("id").loc[[pid for pid in ids if pid in known]]
        self.ids = list(self.p.index)
        self.E = embeddings[[order[pid] for pid in self.ids]]
        self.bm25 = BM25Okapi([tokens(t) for t in self.p.text])

    @classmethod
    def from_chroma(cls, passages: pd.DataFrame, collection) -> "PassageRanker":
        got = collection.get(include=["embeddings"])
        return cls(passages, list(got["ids"]), np.asarray(got["embeddings"], dtype=np.float32))

    def rank(self, q_emb: np.ndarray, query: str, mode: str = "dense", k: int = 5) -> list[Ranked]:
        dense = self.E @ q_emb
        if mode == "rrf":
            bm = self.bm25.get_scores(tokens(query))
            fused = np.zeros_like(dense)
            for s in (dense, bm):
                r = np.empty(len(s), dtype=np.int64)
                r[np.argsort(-s)] = np.arange(1, len(s) + 1)
                fused += 1.0 / (RRF_K + r)
            order = np.argsort(-fused)[:k]
        else:
            order = np.argsort(-dense)[:k]
        out = []
        for i in order:
            row = self.p.iloc[i]
            out.append(Ranked(self.ids[i], float(dense[i]), row["lang"], row["doc"], int(row["page"]), row["url"],
                              row["text"]))
        return out

    def rank_profiled(self, q_emb: np.ndarray, query: str, mode: str, k: int, profile, boost: float,
                      pool: int = 20) -> list[Ranked]:
        """Rank as `rank`, then move passages that match the field profile up (and conflicting ones down) by
        `boost` rank positions per matching attribute (zone, water). boost = 0 leaves the ranking unchanged."""
        from member4_chatbot.chat.profile import match_score, passage_tags

        cands = self.rank(q_emb, query, mode, pool)
        if not boost or profile is None:
            return cands[:k]
        keyed = [(i - boost * match_score(profile, passage_tags(h.doc)), i, h) for i, h in enumerate(cands)]
        return [h for _, _, h in sorted(keyed)[:k]]

    def best_dense(self, q_emb: np.ndarray) -> float:
        return float((self.E @ q_emb).max())
