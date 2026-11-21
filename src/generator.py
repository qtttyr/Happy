"""Runtime wrapper for the tiny seq2seq fallback + its quality gate.

The generator is deliberately paranoid: any weird output is rejected and the
normal human fallback is used instead. It never overrides a retrieval hit.
"""
from __future__ import annotations

import re
from pathlib import Path

CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
BANNED = (
    "бот", "ии", "нейросет", "chatgpt", "gpt", "не могу", "как модель",
    "языковая", "api", "http", "<", ">",
)
GENERIC = {"привет", "как дела", "норм", "да", "нет", "хорошо", "пока", "ок", "ага"}
BAD_LAST = {
    "и", "а", "но", "что", "не", "в", "на", "у", "с", "по", "за", "из", "от",
    "до", "же", "ли", "бы", "меня", "тебя", "его", "её", "их", "это", "как",
}


def cyrillic_ratio(text: str) -> float:
    letters = [ch for ch in text.lower() if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch in CYRILLIC for ch in letters) / len(letters)


class Generator:
    def __init__(self, artifacts_dir: str | Path) -> None:
        import torch

        from .seq2seq import Seq2Seq
        from .tokenizer import BPETokenizer

        root = Path(artifacts_dir)
        checkpoint = torch.load(root / "seq2seq.pt", map_location="cpu", weights_only=False)
        self.torch = torch
        self.tokenizer = BPETokenizer.load(root / "vocab.json")
        self.model = Seq2Seq(self.tokenizer.vocab_size, **checkpoint["config"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def generate(self, question: str, max_len: int = 20) -> tuple[str, float]:
        ids = self.tokenizer.encode(question, max_len=24)
        tensor = self.torch.tensor([ids], dtype=self.torch.long)
        out, score = self.model.generate(tensor, max_len=max_len, temperature=0.0)
        return self.tokenizer.decode(out), float(score)

    @staticmethod
    def quality_ok(text: str, score: float, question: str) -> bool:
        from .nlp import overlap

        low = text.lower().strip()
        if not (10 <= len(text) <= 140):
            return False
        if cyrillic_ratio(text) < 0.7:
            return False
        if any(bad in low for bad in BANNED):
            return False
        if "?" in text or low in GENERIC:
            return False
        if len(text.split()) < 2:
            return False
        if text.rstrip().endswith((",", ":", ";", "-")):
            return False
        if any(text.count(word) > 2 for word in set(text.split()) if len(word) > 2):
            return False
        last = text.rstrip(").! ").split()[-1].lower()
        if last in BAD_LAST:
            return False
        if low.count(" не ") > 1 or "ничего не" in low:
            return False
        if re.search(r"(.)\1{3,}", text):
            return False
        if score < -1.85:
            return False
        if overlap(text, question) > 0.8:
            return False
        return True
