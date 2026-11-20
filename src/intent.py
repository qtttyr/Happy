"""Runtime intent classifier (BiLSTM + attention), loaded from artifacts.

Used as a soft signal: a small retrieval boost and a trap fallback. The regex
router stays in charge of hard cases, so a classifier mistake cannot break the
answer by itself.
"""
from __future__ import annotations

from pathlib import Path


class IntentPredictor:
    def __init__(self, artifacts_dir: str | Path) -> None:
        import torch

        from .models import IntentClassifier
        from .tokenizer import BPETokenizer

        root = Path(artifacts_dir)
        self.torch = torch
        self.tokenizer = BPETokenizer.load(root / "vocab.json")
        checkpoint = torch.load(root / "intent_model.pt", map_location="cpu", weights_only=False)
        self.labels: list[str] = checkpoint["labels"]
        self.model = IntentClassifier(
            self.tokenizer.vocab_size, len(self.labels), **checkpoint["config"]
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def predict(self, text: str) -> tuple[str, float]:
        ids = self.tokenizer.encode(text)
        tensor = self.torch.tensor([ids], dtype=self.torch.long)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(tensor), dim=-1)[0]
        index = int(probs.argmax())
        return self.labels[index], float(probs[index])
