"""Persona engine: one consistent character for the whole round.

The persona is data, not code: data/persona.json can be swapped without
touching the engine. Answers never contradict the persona because personal
facts are only injected from this file.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


class Persona:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.name: str = data.get("name", "Дана")
        self.age: int = int(data.get("age", 19))
        self.city: str = data.get("city", "Астана")
        self.uni: str = data.get("uni", "AITU")
        self.course: int = int(data.get("course", 2))
        self.course_label: str = data.get("course_label", str(self.course))
        self.city_from: str = data.get("city_from", self.city)
        self.city_in: str = data.get("city_in", self.city)
        self.gender: str = data.get("gender", "f")
        self.facts: dict[str, list[str]] = data.get("facts", {})
        self.voice: dict = data.get("voice", {})
        self.defense: list[str] = data.get("bot_defense", [])

    @classmethod
    def load(cls, path: str | Path) -> "Persona":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def fact(self, key: str) -> str | None:
        values = self.facts.get(key) or []
        return random.choice(values) if values else None

    def defense_line(self, avoid: list[str] | None = None) -> str:
        lines = self.defense or ["не, с чего ты взял?) я вообще с телефона сижу"]
        if avoid:
            from .nlp import overlap

            return min(lines, key=lambda line: max(overlap(line, prev) for prev in avoid))
        return random.choice(lines)

    def self_intro(self) -> str:
        return f"{self.name}, {self.age}, учусь в {self.uni} на {self.course_label} курсе"

    def identity_line(self, avoid: list[str] | None = None) -> str:
        lines = [
            f"{self.name}, {self.age}. из {self.city_from}, учусь в {self.uni} на {self.course_label} курсе",
            f"{self.name}. {self.age} лет, живу в {self.city_in}, учусь на программиста",
            f"обычный студент из {self.city_from}. {self.name}, {self.age}",
            f"{self.name}, {self.age}. {self.city}, {self.uni}, на {self.course_label} курсе. а ты кто?",
        ]
        if avoid:
            from .nlp import overlap

            return min(lines, key=lambda line: max(overlap(line, prev) for prev in avoid))
        return random.choice(lines)
