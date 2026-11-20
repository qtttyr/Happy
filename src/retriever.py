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
    dense: float = 0.0
    rank: float = 0.0


class Retriever:
    def __init__(self, corpus: list[dict], index: BM25Index, dense=None) -> None:
        self.corpus = corpus
        self.index = index
        self.dense = dense
        self.exact: dict[str, int] = {}
        # Primary questions win over somebody else's variant.
        for i, item in enumerate(corpus):
            key = normalize(item.get("q", ""))
            if key:
                self.exact.setdefault(key, i)
        for i, item in enumerate(corpus):
            for variant in item.get("variants", []):
                key = normalize(variant)
                if key:
                    self.exact.setdefault(key, i)

    def search(
        self, question: str, top_k: int = 5, topic: str | None = None, intent: str | None = None
    ) -> list[Hit]:
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
            if intent and self.corpus[idx].get("intent") == intent:
                rel *= 1.05
            hit = self._make(idx, score, qtokens, q_terms, rel=min(1.0, rel))
            hits.append(hit)

        for hit in hits:
            hit.rank = hit.rel
        if self.dense is not None:
            try:
                vector = self.dense.encode(question)
                scores = self.dense.scores(vector, [h.idx for h in hits])
                for hit in hits:
                    hit.dense = scores.get(hit.idx, 0.0)
                    hit.rank = 0.55 * hit.rel + 0.45 * max(0.0, hit.dense)
            except Exception:
                pass
        hits.sort(key=lambda h: (h.exact, h.rank, h.rel), reverse=True)
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
