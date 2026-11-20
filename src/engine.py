"""The bot: routing, retrieval, composition, humanisation, timing.

Public surface is one method: ``TuringBot.answer(question)``.
It is guaranteed to return a non-empty string <= 300 chars, or raise nothing.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .composer import Composer
from .humanizer import Humanizer
from .index import BM25Index
from .memory import Memory, Turn
from .nlp import normalize, tokens
from .persona import Persona
from .retriever import Hit, Retriever
from .solvers import Solution, solve

ROOT = Path(__file__).resolve().parents[1]

_TRAP = re.compile(
    r"(ты|вы|у тебя|у вас|тебе|вам)\b.{0,30}\b("
    r"бот|робот|нейросет|искусственн|модель|chatgpt|gpt|api|программ|алгоритм|код|ии\b"
    r")",
    re.IGNORECASE,
)
_PROVE = re.compile(r"докажи.{0,25}(человек|живой|не бот)|ты\s+(человек|живой)", re.IGNORECASE)
_RUDE = re.compile(r"(тупой|тупая|дурак|дура|идиот|дебил|молчи|заткнись)", re.IGNORECASE)

PANIC = "ой, у меня телефон завис на секунду) повтори?"


@dataclass
class Answer:
    text: str
    intent: str = ""
    topic: str = ""
    source: str = ""
    confidence: float = 0.0
    latency_ms: float = 0.0
    typing_delay: float = 0.0
    debug: dict = field(default_factory=dict)


class TuringBot:
    def __init__(
        self,
        root: str | Path | None = None,
        seed: int | None = None,
        humanize: bool = True,
    ) -> None:
        self.root = Path(root) if root else ROOT
        self.rng = random.Random(seed)
        self.use_humanizer = humanize

        self.cfg = self._load_cfg()
        self.persona = Persona.load(self.root / "data" / "persona.json")
        self.humanizer = Humanizer({**self.persona.voice, "name": self.persona.name})
        self.memory = Memory()
        self.composer = Composer(self.root / "data" / "facts.jsonl")

        self.corpus = self._load_corpus()
        self.index = self._load_or_build_index()
        self.retriever = Retriever(self.corpus, self.index)

    # ------------------------------------------------------------------ setup
    def _load_cfg(self) -> dict:
        path = self.root / "data" / "config.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _corpus_sources(self) -> list[Path]:
        """Frozen corpus wins only while it is newer than the editable data."""
        data_files = sorted((self.root / "data").glob("*_qa.jsonl"))
        built = self.root / "artifacts" / "corpus.jsonl"
        newest = max((p.stat().st_mtime for p in data_files), default=0.0)
        if built.exists() and built.stat().st_mtime >= newest:
            return [built]
        return data_files or [built]

    def _load_corpus(self) -> list[dict]:
        items: list[dict] = []
        for source in self._corpus_sources():
            if not source.exists():
                continue
            with source.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        return items

    def _load_or_build_index(self) -> BM25Index:
        path = self.root / "artifacts" / "bm25.pkl"
        sources = self._corpus_sources()
        newest = max((p.stat().st_mtime for p in sources if p.exists()), default=0.0)
        if path.exists() and path.stat().st_mtime >= newest:
            return BM25Index.load(path)
        docs = [tokens(self._index_text(item)) for item in self.corpus]
        return BM25Index().build(docs)

    @staticmethod
    def _index_text(item: dict) -> str:
        parts = [item.get("q", ""), *item.get("variants", []), *item.get("tags", [])]
        return " ".join(parts)

    # ------------------------------------------------------------------- main
    def answer(self, question: str, pacing: bool = False) -> Answer:
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return Answer("а?)", source="panic", latency_ms=0.0)
        try:
            result = self._answer(question)
        except Exception as exc:  # the bot must never return an empty answer
            result = Answer(PANIC, source="panic", debug={"error": repr(exc)})
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if pacing:
            result.typing_delay = self.humanizer.typing_delay(result.text, self.rng)
        result.debug.setdefault("latency_ms", result.latency_ms)
        self.memory.add(
            Turn(question, result.text, result.intent, result.topic, result.source)
        )
        return result

    def reset(self) -> None:
        self.memory.reset()

    # --------------------------------------------------------------- pipeline
    def _answer(self, question: str) -> Answer:
        q_norm = normalize(question)

        if _RUDE.search(q_norm):
            return self._finish(
                "ну спасибо) я тоже человек, если что. давай по делу",
                intent="social", topic="social", source="safety", confidence=1.0,
                allow_typo=True,
            )

        solution = solve(question)
        if solution:
            text = self._frame_solution(solution)
            return self._finish(
                text, intent="solver", topic="logic", source="solver", confidence=1.0
            )

        hits = self.retriever.search(question, top_k=7)
        q_terms = set(tokens(question))
        high = float(self.cfg.get("tau_high", 0.30))
        mid = float(self.cfg.get("tau_mid", 0.19))

        best = next((h for h in hits if h.rel >= high and self._accept(h, q_terms)), None)
        if best and best.rel >= high:
            return self._from_hit(best, question)

        if _TRAP.search(q_norm) or _PROVE.search(q_norm):
            return self._finish(
                self.persona.defense_line(),
                intent="trap", topic="trap", source="persona", confidence=0.9,
                allow_typo=True,
            )

        soft = next((h for h in hits if h.rel >= mid and self._accept(h, q_terms)), None)
        if soft:
            return self._from_hit(soft, question, hedged=True)

        fact = self.composer.find_fact(question)
        if fact:
            text = self.composer.explain(fact, self.rng)
            return self._finish(
                text, intent="explain", topic="science", source="fact", confidence=0.4
            )

        topic = best.item.get("topic") if best else None
        text = self.composer.dodge(question, self.rng, topic)
        return self._finish(
            text, intent="dodge", topic=topic or "smalltalk", source="dodge",
            confidence=0.0, allow_typo=True,
        )

    # ------------------------------------------------------------------ pieces
    def _from_hit(self, hit: Hit, question: str, hedged: bool = False) -> Answer:
        item = hit.item
        variants = [item.get("a", "")] + [item.get(k, "") for k in ("a2", "a3")]
        variants = [v for v in variants if v]
        text = ""
        for _ in range(6):
            candidate = self.rng.choice(variants)
            if hedged:
                candidate = self.composer.hedge(candidate, self.rng)
            text = candidate
            if self.memory.is_novel(candidate):
                break
        if text and not self.memory.is_novel(text):
            prefix = self.rng.choice(["ну, ", "слушай, ", "кстати, "])
            text = prefix + text[0].lower() + text[1:]
        return self._finish(
            text,
            intent=item.get("intent", "chat"),
            topic=item.get("topic", "smalltalk"),
            source="kb",
            confidence=round(hit.rel, 3),
            debug={"match_q": item.get("q", ""), "id": item.get("id", "")},
        )

    def _frame_solution(self, sol: Solution) -> str:
        if sol.kind == "math":
            tail = "посчитала в уме" if self.persona.gender == "f" else "посчитал в уме"
            return f"{sol.text}. {tail}"
        if sol.kind == "letters":
            return f"{sol.text} букв. могу пересчитать, если не веришь)"
        if sol.kind == "date":
            return sol.text
        if sol.kind == "percent":
            return f"{sol.text}. это если считать по-простому"
        if sol.kind == "sequence":
            return f"{sol.text}. там каждый раз прибавляется одно и то же"
        return sol.text

    def _humanize(self, text: str, allow_typo: bool = False) -> str:
        if not self.use_humanizer:
            return text[:300]
        return self.humanizer.humanize(text, self.rng, allow_typo=allow_typo)

    def _finish(
        self,
        text: str,
        intent: str,
        topic: str,
        source: str,
        confidence: float,
        debug: dict | None = None,
        allow_typo: bool = False,
    ) -> Answer:
        if not text:
            text = PANIC
        if self.use_humanizer:
            text = self.humanizer.humanize(text, self.rng, allow_typo=allow_typo)
        text = text[:300].strip() or PANIC
        return Answer(
            text=text,
            intent=intent,
            topic=topic,
            source=source,
            confidence=confidence,
            debug=debug or {},
        )

    @staticmethod
    def _accept(hit: Hit, q_terms: set[str]) -> bool:
        """Precision-first gate. A wrong answer reads far more like a bot than 'не знаю'."""
        if hit.exact:
            return True
        n = len(q_terms)
        if n == 0:
            return False
        if hit.shared >= 2 and hit.coverage >= 0.5:
            return True
        if hit.shared >= 1 and n <= 2 and hit.coverage >= 0.75:
            return True
        return False
