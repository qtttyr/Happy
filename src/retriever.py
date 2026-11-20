"""Hybrid retrieval: exact match -> BM25 (+ optional dense vectors later)."""
from __future__ import annotations

from dataclasses import dataclass

from .index import BM25Index
from .nlp import normalize, tokens


def index_text(item: dict) -> str:
    parts = [item.get("q", ""), *item.get("variants", []), *item.get("tags", [])]
    return " ".join(parts)


@dataclass
class Hit:
    idx: int
    item: dict
    score: float
    rel: float
    matched: int
    shared: int = 0
    coverage: float = 0.0
    exact: bool = False


class Retriever:
    def __init__(self, corpus: list[dict], index: BM25Index) -> None:
        self.corpus = corpus
        self.index = index
        self.exact: dict[str, int] = {}
        for i, item in enumerate(corpus):
            for q in [item.get("q", ""), *item.get("variants", [])]:
                key = normalize(q)
                if key:
                    self.exact.setdefault(key, i)

    def search(self, question: str, top_k: int = 5, topic: str | None = None) -> list[Hit]:
        qtokens = tokens(question)
        q_terms = set(qtokens)
        key = normalize(question)

        hits: list[Hit] = []
        if key in self.exact:
            idx = self.exact[key]
            hit = self._make(idx, 100.0, qtokens, q_terms)
            hit.exact = True
            hit.rel = 1.0
            hits.append(hit)

        mass = self.index.query_mass(qtokens)
        scores = self.index.score(qtokens)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: max(top_k * 8, 30)]
        seen = {h.idx for h in hits}
        for idx, score in ranked:
            if idx in seen:
                continue
            rel = score / mass
            if topic and self.corpus[idx].get("topic") == topic:
                rel *= 1.06
            hit = self._make(idx, score, qtokens, q_terms, rel=min(1.0, rel))
            hits.append(hit)

        hits.sort(key=lambda h: (h.exact, h.rel), reverse=True)
        return hits[:top_k]

    def _make(
        self, idx: int, score: float, qtokens: list[str], q_terms: set[str], rel: float | None = None
    ) -> Hit:
        item = self.corpus[idx]
        doc_terms = set(tokens(index_text(item)))
        shared = len(q_terms & doc_terms)
        coverage = shared / len(q_terms) if q_terms else 0.0
        return Hit(
            idx=idx,
            item=item,
            score=score,
            rel=score if rel is None else rel,
            matched=len(qtokens),
            shared=shared,
            coverage=coverage,
        )
