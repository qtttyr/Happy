"""Dense half of the hybrid: load the from-scratch bi-encoder at runtime.

Optional by design. If torch is missing or artifacts are stale, the bot keeps
working on BM25 alone - the stage must never depend on a neural import.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class DenseIndex:
    def __init__(self, artifacts_dir: str | Path, corpus_size: int | None = None) -> None:
        import torch

        from .models import BiEncoder
        from .tokenizer import BPETokenizer

        root = Path(artifacts_dir)
        self.torch = torch
        self.tokenizer = BPETokenizer.load(root / "vocab.json")
        checkpoint = torch.load(root / "bi_encoder.pt", map_location="cpu", weights_only=False)
        self.model = BiEncoder(self.tokenizer.vocab_size, **checkpoint["config"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.vectors = np.load(root / "corpus_vectors.npy").astype(np.float32)
        if corpus_size is not None and len(self.vectors) != corpus_size:
            raise ValueError(
                f"vectors ({len(self.vectors)}) != corpus ({corpus_size}); rerun train.py"
            )

    def encode(self, text: str) -> np.ndarray:
        ids = self.tokenizer.encode(text)
        tensor = self.torch.tensor([ids], dtype=self.torch.long)
        with self.torch.no_grad():
            return self.model(tensor).numpy()[0]

    def scores(self, vector: np.ndarray, indices: list[int]) -> dict[int, float]:
        if not indices:
            return {}
        subset = self.vectors[indices]
        raw = subset @ vector
        return {int(i): float(s) for i, s in zip(indices, raw)}
