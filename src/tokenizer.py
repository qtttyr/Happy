"""A small BPE tokenizer trained from scratch on our own corpus.

No pretrained vocabulary, no SentencePiece, no Hugging Face tokenizer.
The merge list is the artifact: it is trained on the curated corpus plus the
dialogue pairs and shipped as ``vocab.json``.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[а-яa-z]+")
PAD, UNK, EOS = "<pad>", "<unk>", "</s>"


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        max_len: int = 32,
    ) -> None:
        self.vocab = vocab or {PAD: 0, UNK: 1, EOS: 2}
        self.inv = {i: t for t, i in self.vocab.items()}
        self.merges = merges or []
        self.ranks = {tuple(m): i for i, m in enumerate(self.merges)}
        self.max_len = max_len
        self._cache: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------------ train
    @classmethod
    def train(
        cls,
        texts: list[str],
        num_merges: int = 2000,
        min_freq: int = 2,
        max_len: int = 32,
    ) -> "BPETokenizer":
        word_freq: Counter[str] = Counter()
        for text in texts:
            word_freq.update(WORD_RE.findall((text or "").lower()))

        words: dict[tuple[str, ...], int] = {
            tuple(w) + ("</w>",): c for w, c in word_freq.items() if w
        }
        merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            pairs: Counter[tuple[str, str]] = Counter()
            for word, freq in words.items():
                for a, b in zip(word, word[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            (a, b), freq = pairs.most_common(1)[0]
            if freq < min_freq:
                break
            merges.append((a, b))
            merged_token = a + b
            new_words: dict[tuple[str, ...], int] = {}
            for word, count in words.items():
                merged = cls._merge_word(word, a, b, merged_token)
                new_words[merged] = new_words.get(merged, 0) + count
            words = new_words

        vocab: dict[str, int] = {PAD: 0, UNK: 1, EOS: 2}
        for word in word_freq:
            for ch in word:
                vocab.setdefault(ch, len(vocab))
        vocab.setdefault("</w>", len(vocab))
        for a, b in merges:
            vocab.setdefault(a + b, len(vocab))
        return cls(vocab, merges, max_len)

    @staticmethod
    def _merge_word(
        word: tuple[str, ...], a: str, b: str, merged_token: str
    ) -> tuple[str, ...]:
        out: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                out.append(merged_token)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    # ----------------------------------------------------------------- encode
    def _encode_word(self, word: str) -> tuple[int, ...]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        symbols = list(word) + ["</w>"]
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            best = min(pairs, key=lambda p: self.ranks.get(p, 10**9))
            if best not in self.ranks:
                break
            a, b = best
            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        ids = tuple(self.vocab.get(symbol, self.vocab[UNK]) for symbol in symbols)
        self._cache[word] = ids
        return ids

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        limit = (max_len or self.max_len) - 1
        ids: list[int] = []
        for word in WORD_RE.findall((text or "").lower()):
            ids.extend(self._encode_word(word))
            if len(ids) >= limit:
                break
        ids = ids[:limit] + [self.vocab[EOS]]
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------- disk
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "max_len": self.max_len,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(m) for m in payload.get("merges", [])]
        return cls(payload["vocab"], merges, payload.get("max_len", 32))
