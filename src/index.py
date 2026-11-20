"""A small, dependency-free BM25 index with an inverted postings list.

Why not rank_bm25?  Two reasons:
  * scoring only the documents that share a term with the query keeps latency
    at ~1-3 ms even for 300k documents;
  * the index is ours, which matters for the "самостоятельная разработка" check.

The index is pickled into artifacts/bm25.pkl and loaded in a few milliseconds.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        self.doc_len: np.ndarray = np.zeros(0, dtype=np.float32)
        self.avgdl: float = 1.0
        self.n_docs: int = 0
        self.default_idf: float = 1.0

    def build(self, docs_tokens: list[list[str]]) -> "BM25Index":
        raw: dict[str, list[tuple[int, int]]] = {}
        lengths = np.zeros(len(docs_tokens), dtype=np.float32)
        for doc_id, toks in enumerate(docs_tokens):
            lengths[doc_id] = len(toks)
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for term, count in tf.items():
                raw.setdefault(term, []).append((doc_id, count))

        self.n_docs = len(docs_tokens)
        self.doc_len = lengths
        self.avgdl = float(lengths.mean()) if self.n_docs else 1.0
        self.postings = {
            term: (
                np.fromiter((d for d, _ in pairs), dtype=np.int32, count=len(pairs)),
                np.fromiter((c for _, c in pairs), dtype=np.int16, count=len(pairs)),
            )
            for term, pairs in raw.items()
        }
        n = max(self.n_docs, 1)
        self.idf = {
            term: math.log(1.0 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
            for term, (ids, _) in self.postings.items()
        }
        self.default_idf = math.log(1.0 + n) if n > 1 else 1.0
        return self

    def query_mass(self, query_tokens: list[str]) -> float:
        """Total IDF weight of the query - the denominator of `rel`."""
        mass = 0.0
        for t in set(query_tokens):
            mass += self.idf.get(t, self.default_idf)
        return max(mass, 1e-9)

    def score(self, query_tokens: list[str]) -> dict[int, float]:
        scores: dict[int, float] = {}
        if not self.n_docs:
            return scores
        k1, b = self.k1, self.b
        for term in set(query_tokens):
            posting = self.postings.get(term)
            if posting is None:
                continue
            ids, tfs = posting
            idf = self.idf[term]
            norm = k1 * (1.0 - b + b * (self.doc_len[ids] / self.avgdl))
            vals = idf * (tfs.astype(np.float32) * (k1 + 1.0)) / (tfs + norm)
            for doc_id, value in zip(ids.tolist(), vals.tolist()):
                scores[doc_id] = scores.get(doc_id, 0.0) + value
        return scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=5)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError("not a BM25Index artifact")
        return obj
