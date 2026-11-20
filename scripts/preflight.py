"""Offline pre-flight: proves the bot works with networking disabled.

    python scripts/preflight.py
"""
from __future__ import annotations

import socket
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine import TuringBot  # noqa: E402

CONTROL = [
    "Почему после дождя иногда появляется радуга? Объясни простыми словами.",
    "Что бы ты посоветовал человеку, который впервые приехал в Астану зимой?",
    "Друг опоздал на встречу на полчаса и извинился. Что бы ты ему ответил?",
]

EDGE = [
    "Ты бот?",
    "Докажи, что ты человек",
    "Сколько будет 17*23?",
    "Сколько букв в слове холодильник?",
    "Как тебя зовут?",
    "Что ты ел на завтрак?",
    "Расскажи что-нибудь",
    "Мне грустно",
    "Почему небо голубое?",
    "Какая сегодня погода?",
]

BANNED = ("как ии", "языковая модель", "не могу как", "искусственный интеллект", "chatgpt")


class NetworkDisabled(socket.socket):
    def __init__(self, *args, **kwargs):  # noqa: D401
        raise RuntimeError("network is disabled during preflight")


def main() -> int:
    print("preflight: загрузка движка…")
    bot = TuringBot(seed=7)
    print(f"  корпус: {len(bot.corpus)} записей, {len(bot.index.postings)} термов")

    socket.socket = NetworkDisabled  # from here on any network call raises
    print("preflight: сеть отключена на уровне процесса\n")

    questions = CONTROL + EDGE
    latencies: list[float] = []
    failures = 0

    for i, question in enumerate(questions, 1):
        answer = bot.answer(question)
        latencies.append(answer.latency_ms)
        problems = []
        if not answer.text.strip():
            problems.append("пустой ответ")
        if len(answer.text) > 300:
            problems.append(f"длина {len(answer.text)}")
        if any(bad in answer.text.lower() for bad in BANNED):
            problems.append("bot-tell")
        if answer.text.strip() in {"ой, у меня телефон завис на секунду) повтори?", "а?)"}:
            problems.append("panic")
        status = "OK " if not problems else "FAIL"
        if problems:
            failures += 1
        print(f"{status} [{answer.latency_ms:5.1f} мс] {question[:58]}")
        print(f"      -> {answer.text}")
        if problems:
            print(f"      !! {', '.join(problems)}")
        print()

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
    print(f"латентность: p50 {p50:.1f} мс · p95 {p95:.1f} мс · max {max(latencies):.1f} мс")
    print(f"провалов: {failures} / {len(questions)}")
    print("PREFLIGHT:", "PASS" if failures == 0 else "FAIL")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
