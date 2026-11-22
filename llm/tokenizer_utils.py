"""Tokenizer trained from scratch with the `tokenizers` library (no pretrained vocab)."""
from __future__ import annotations

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer, Tokenizer

SPECIALS = ["<pad>", "<unk>", "<s>", "</s>", "<|user|>", "<|bot|>", "<|end|>"]


def train_tokenizer(texts, vocab_size: int, path: str | Path, min_frequency: int = 2):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIALS,
        show_progress=True,
    )
    tokenizer.save(str(path))
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def special_id(tokenizer: Tokenizer, token: str) -> int:
    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"missing special token {token}")
    return int(value)


def encode_turn(tokenizer: Tokenizer, question: str, answer: str) -> list[int]:
    user = special_id(tokenizer, "<|user|>")
    bot = special_id(tokenizer, "<|bot|>")
    end = special_id(tokenizer, "<|end|>")
    return (
        [user]
        + tokenizer.encode(question).ids
        + [bot]
        + tokenizer.encode(answer).ids
        + [end]
    )


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    text = tokenizer.decode(ids, skip_special_tokens=True)
    for token in SPECIALS:
        text = text.replace(token, "")
    return " ".join(text.split()).strip()
