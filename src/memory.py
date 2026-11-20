"""Round memory: anti-repetition, novelty guard and light context.

A human in a live round does not answer five questions with the same opener.
The audience notices repetition much faster than a slightly weak answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .nlp import overlap


@dataclass
class Turn:
    question: str
    answer: str
    intent: str = ""
    topic: str = ""
    source: str = ""


@dataclass
class Memory:
    window: int = 8
    turns: list[Turn] = field(default_factory=list)
    used_openers: set[str] = field(default_factory=set)

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.window * 2:
            self.turns = self.turns[-self.window * 2:]

    def reset(self) -> None:
        self.turns.clear()
        self.used_openers.clear()

    def recent_answers(self, limit: int | None = None) -> list[str]:
        n = limit or self.window
        return [t.answer for t in self.turns[-n:]]

    def is_novel(self, text: str, threshold: float = 0.62) -> bool:
        return all(overlap(text, prev) < threshold for prev in self.recent_answers())

    def was_asked(self, question: str) -> bool:
        return any(overlap(question, t.question) > 0.8 for t in self.turns)
