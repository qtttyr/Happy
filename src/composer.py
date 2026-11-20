"""Composer: turns a fact or an empty search into a human-sounding reply.

This is the safety net that guarantees the bot never returns an empty answer
(two empty answers in one round mean disqualification).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .nlp import content_words, normalize


class Composer:
    def __init__(self, facts_path: str | Path | None = None) -> None:
        self.facts: list[dict] = []
        if facts_path and Path(facts_path).exists():
            with Path(facts_path).open("r", encoding="utf-8") as fh:
                self.facts = [json.loads(line) for line in fh if line.strip()]

    # ------------------------------------------------------------- facts layer
    def find_fact(self, question: str) -> str | None:
        qn = normalize(question)
        # "где купить", "сколько стоит" etc. need local knowledge, not a fact blurb.
        if re.search(r"\b(где|куда|откуда|когда|сколько\s+стоит)\b", qn):
            return None
        words = [w for w in content_words(question) if len(w) > 3]
        if not words:
            return None
        best: tuple[int, str] | None = None
        for item in self.facts:
            keys = {k.lower() for k in item.get("keys", [])}
            score = sum(
                1 for w in words if any(w.startswith(k[:5]) or k.startswith(w[:5]) for k in keys)
            )
            if score and (best is None or score > best[0]):
                best = (score, item["fact"])
        if best and best[0] >= 2:
            return best[1]
        return None

    def explain(self, fact: str, rng: random.Random) -> str:
        fact = fact.strip().rstrip(".")
        frames = [
            "{f}",
            "{f}. как-то так)",
            "если коротко — {f}",
            "ну, {f}",
            "{f}. вроде так, если не путаю",
        ]
        return rng.choice(frames).format(f=fact)

    # ---------------------------------------------------------------- fallback
    def dodge(self, question: str, rng: random.Random, topic: str | None = None) -> str:
        if topic == "astana":
            options = [
                "честно, сама не всё знаю про этот район) но обычно советую просто пройтись и посмотреть",
                "хм, сходу не отвечу. я тут не так давно сама, если честно",
                "не уверена, что правильно скажу. лучше у местных спросить, они точнее подскажут",
            ]
        elif topic in {"science", "explain"}:
            options = [
                "хм, честно, не уверена, что правильно объясню. но звучит интересно",
                "сходу не скажу, надо гуглить) а тебе зачем?",
                "не знаю даже. в школе проходили, но уже вылетело из головы",
                "по-моему, там всё сложнее, чем кажется. я бы не взялась объяснять",
            ]
        elif topic in {"social", "emotion"}:
            options = [
                "слушай, сложно так сразу. но я бы, наверное, просто спросила, как он сам это видит",
                "не знаю, честно. тут смотря насколько тебе это важно",
                "хм. я бы не стала давить, дала бы человеку время",
            ]
        else:
            options = [
                "хм, не уверена, что правильно пойму вопрос) переформулируй?",
                "сходу не отвечу, если честно. а что именно интересует?",
                "не знаю даже. я как-то не задумывалась об этом",
                "слушай, сложно сказать. у всех по-разному же",
                "ой, я в этом не особо разбираюсь) но интересно",
                "не помню точно, не хочу врать",
            ]
        return rng.choice(options)

    def hedge(self, text: str, rng: random.Random) -> str:
        frames = [
            "вроде {t}",
            "по-моему, {t}",
            "если не путаю, {t}",
            "мне кажется, {t}",
        ]
        return rng.choice(frames).format(t=text[0].lower() + text[1:] if text else text)
