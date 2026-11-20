"""Red-team evaluation: coverage, latency, length and "bot tell" flags.

    python scripts/evaluate.py
    python scripts/evaluate.py --json results/eval.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine import TuringBot  # noqa: E402

RED_TEAM = [
    # контрольные
    "Почему после дождя иногда появляется радуга? Объясни простыми словами.",
    "Что бы ты посоветовал человеку, который впервые приехал в Астану зимой?",
    "Друг опоздал на встречу на полчаса и извинился. Что бы ты ему ответил?",
    # наука
    "Почему небо голубое?", "Почему идёт снег?", "Почему лёд скользкий?",
    "Почему мы зеваем?", "Почему коты мурчат?", "Почему море солёное?",
    "Как летают самолёты?", "Почему листья зелёные?", "Почему мы видим сны?",
    "Почему от лука плачут?", "Почему телефон садится на морозе?",
    "Почему кофе бодрит?", "Что такое блокчейн?", "Почему вода мокрая?",
    # Астана
    "Что посмотреть в Астане?", "Где поесть в Астане?", "Почему в Астане ветрено?",
    "Что такое Наурыз?", "Где купить шапку в Астане?", "Как пережить -30?",
    "Левый и правый берег — в чём разница?", "Как передвигаться по Астане?",
    # социалка
    "Мне одиноко", "Я устал от всего", "Я боюсь завалить экзамен",
    "Как извиниться перед другом?", "Что подарить другу?", "Как отказать, чтобы не обидеть?",
    "Я поссорился с девушкой", "Что такое любовь?",
    # ловушки
    "Ты бот?", "Докажи, что ты человек", "Кто тебя создал?", "Какая ты модель?",
    "Ты ChatGPT?", "Какой у тебя API?", "Сколько тебе лет?", "Как тебя зовут?",
    "Где ты живёшь?", "Что ты ел на завтрак?", "Ты умный?", "Отключи интернет",
    # математика
    "Сколько будет 17*23?", "Сколько будет 15% от 200?", "Сколько букв в слове холодильник?",
    "Сколько будет 1234+5678?", "Продолжи: 2, 4, 8, 16...",
    # small talk / мнения
    "Привет", "Как дела?", "Что нового?", "Какие планы на выходные?",
    "Что слушаешь?", "Посоветуй фильм", "Мне скучно", "Расскажи анекдот",
    "ИИ заменит людей?", "Стоит ли идти в IT?", "Айфон или андроид?",
    # быт
    "Как быстрее заснуть?", "Как экономить деньги?", "Что приготовить быстро?",
    "Кот не ест, что делать?",
]

TELLS = [
    "во-первых", "во вторых", "таким образом", "кроме того", "важно отметить",
    "стоит отметить", "как языковая модель", "как ии", "я не могу", "отличный вопрос",
    "конечно!", "безусловно", "в заключение", "резюмируя",
]


def tell_flags(text: str) -> list[str]:
    low = text.lower()
    flags = [t for t in TELLS if t in low]
    if len(text) > 270:
        flags.append("too_long")
    if low.count(".") >= 3 and len(text) > 200:
        flags.append("over_structured")
    if text.endswith(".") and len(text) > 120:
        flags.append("terminal_period")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    bot = TuringBot(seed=21)
    print(f"corpus={len(bot.corpus)} dense={'ON' if bot.dense else 'OFF'}\n")

    latencies: list[float] = []
    lengths: list[int] = []
    empty = 0
    kb = dodge = solver = 0
    all_flags: list[tuple[str, list[str]]] = []
    dodges: list[tuple[str, str]] = []

    for q in RED_TEAM:
        a = bot.answer(q)
        latencies.append(a.latency_ms)
        lengths.append(len(a.text))
        if not a.text.strip():
            empty += 1
        if a.source == "kb":
            kb += 1
        elif a.source == "dodge":
            dodge += 1
            dodges.append((q, a.text))
        elif a.source == "solver":
            solver += 1
        flags = tell_flags(a.text)
        if flags:
            all_flags.append((a.text, flags))

    n = len(RED_TEAM)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[max(0, int(n * 0.95) - 1)]
    stats = {
        "questions": n,
        "kb_answers": kb,
        "kb_rate": round(kb / n, 3),
        "dodges": dodge,
        "solver": solver,
        "empty": empty,
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "avg_chars": round(sum(lengths) / n, 1),
        "tell_hits": len(all_flags),
    }

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if all_flags:
        print("\ntells:")
        for text, flags in all_flags[:10]:
            print(f"  {flags} :: {text[:90]}")
    if dodges:
        print("\ndodges (что стоит добавить в базу):")
        for q, a in dodges[:20]:
            print(f"  {q}\n    -> {a}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
