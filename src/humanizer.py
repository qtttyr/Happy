"""Humanizer: the layer that removes "bookish AI" from a retrieved answer.

Design rule: never change the meaning, only the surface.  The humanizer is
fully deterministic given a seed, so a run can be reproduced on camera.
"""
from __future__ import annotations

import random
import re

_SPACES = re.compile(r"[ \t]+")
_MD = re.compile(r"(\*\*|__|`|~~)")
_BULLET = re.compile(r"(?m)^\s*[-•*]\s+")

_BOOKISH: list[tuple[str, str]] = [
    (r"^\s*Конечно[,!]?\s*", ""),
    (r"^\s*Отличный вопрос[!,.]?\s*", ""),
    (r"^\s*Хороший вопрос[!,.]?\s*", ""),
    (r"\bВажно отметить,?\s*что\s*", "кстати, "),
    (r"\bСледует отметить,?\s*что\s*", "вот, "),
    (r"\bСтоит отметить,?\s*что\s*", "кстати, "),
    (r"\bТаким образом,?\s*", "короче, "),
    (r"\bВ заключение,?\s*", ""),
    (r"\bЯ считаю,?\s*что\s*", "по-моему, "),
    (r"\bНеобходимо\s*", "надо "),
    (r"\bосуществлять\s*", "делать "),
    (r"\bявляется\s*", "это "),
    (r"\bданный\s*", "этот "),
    (r"\bоднако\b", "но"),
    (r"\bтакже\b", "ещё"),
    (r"\bнапример,?\s*", "например "),
    (r"\bв целом,?\s*", "вообще "),
    (r"\bдостаточно\s+", "довольно "),
]

_TYPO_SAFE = re.compile(r"^[а-яa-z]{6,}$")


class Humanizer:
    def __init__(self, voice: dict | None = None, max_len: int = 300) -> None:
        self.voice = voice or {}
        self.max_len = max_len

    # ------------------------------------------------------------------ public
    def humanize(
        self, text: str, rng: random.Random | None = None, allow_typo: bool = False
    ) -> str:
        rng = rng or random
        text = self._clean(text)
        text = self._debookish(text)
        text = self._case(text, rng)
        text = self._trim(text)
        text = self._punctuation(text, rng)
        if allow_typo:
            text = self._maybe_typo(text, rng)
        return text.strip()

    def typing_delay(self, text: str, rng: random.Random | None = None) -> float:
        """How long a human would take to type this, in seconds."""
        rng = rng or random
        base = 0.8 + len(text) / 13.0
        return round(min(9.0, base + rng.uniform(0.2, 1.6)), 2)

    # ----------------------------------------------------------------- private
    def _clean(self, text: str) -> str:
        text = _MD.sub("", text or "")
        text = _BULLET.sub("", text)
        text = text.replace("\n", " ").replace("—", " — ")
        text = _SPACES.sub(" ", text).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\.{2,}", "...", text)
        text = re.sub(r"\s*-\s*", " - ", text) if " - " in text else text
        return text

    def _debookish(self, text: str) -> str:
        for pattern, repl in _BOOKISH:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        text = re.sub(r"[,;:]\s*\)", ")", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return _SPACES.sub(" ", text).strip()

    def _case(self, text: str, rng: random.Random) -> str:
        if not text:
            return text
        first = text.split(" ", 1)[0]
        protected = {self.voice.get("name", "").lower(), "астана", "казахстан", "aitu", "chatgpt"}
        proper = any(ch.isupper() for ch in first[1:])
        if rng.random() < float(self.voice.get("lowercase_p", 0.55)):
            if (
                first.strip(",.!?)").lower() not in protected
                and not first[0].isdigit()
                and not proper
            ):
                text = text[0].lower() + text[1:]
        return text

    def _trim(self, text: str) -> str:
        if len(text) <= self.max_len:
            return text
        limit = self.max_len - 1
        cut = text[:limit]
        boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        if boundary > limit * 0.6:
            return cut[: boundary + 1].strip()
        space = cut.rfind(" ")
        return (cut[:space] if space > 0 else cut).strip() + "…"

    def _punctuation(self, text: str, rng: random.Random) -> str:
        if not text:
            return text
        text = text.rstrip(" ,;:")
        if text.endswith(".") and rng.random() < float(self.voice.get("drop_period_p", 0.6)):
            text = text[:-1]
        text = text.rstrip(" ,;:")
        if not text:
            return text
        if text[-1] not in ".!?…)" and rng.random() < float(self.voice.get("paren_p", 0.22)):
            text += ")"
        if text.endswith(".") and rng.random() < float(self.voice.get("paren_p", 0.22)) * 0.6:
            text = text[:-1] + ")"
        return text

    def _maybe_typo(self, text: str, rng: random.Random) -> str:
        if rng.random() >= float(self.voice.get("typo_p", 0.06)):
            return text
        words = text.split(" ")
        candidates = [i for i, w in enumerate(words) if i > 0 and _TYPO_SAFE.match(w)]
        if not candidates:
            return text
        i = rng.choice(candidates)
        word = words[i]
        j = rng.randrange(1, len(word) - 1)
        words[i] = word[:j] + word[j + 1] + word[j] + word[j + 2:]
        return " ".join(words)
