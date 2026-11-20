"""Rule-based generation of the intent-classifier training set.

This is not an LLM and not a pretrained dataset: it is a template expansion
over the curated corpus plus hand-written question patterns. It exists because
92 curated questions are not enough to train a classifier; the expansion makes
the class boundaries explicit and reproducible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

LABELS = [
    "explain",
    "advice",
    "social",
    "support",
    "smalltalk",
    "trap",
    "math",
    "opinion",
    "everyday",
    "greeting",
    "fact",
]

SCIENCE = [
    "радуга", "небо", "гром", "молния", "лёд", "снег", "луна", "море", "соль",
    "самолёт", "листья", "сон", "зевота", "кофе", "вода", "эхо", "ветер", "град",
    "роса", "иней", "туман", "звёзды", "солнце", "тень", "огонь", "дым", "пар",
]
THINGS = [
    "телефон", "холодильник", "интернет", "блокчейн", "нейросеть", "квант",
    "атом", "витамин", "белок", "вакцина", "электричество", "гравитация",
    "аккумулятор", "процессор", "алгоритм", "код", "сайт", "приложение",
]
PEOPLE = ["друг", "девушка", "парень", "мама", "папа", "родители", "брат", "сестра", "сосед", "одногруппник"]
PROBLEMS = ["поссорился", "опоздал", "обидел", "не понял", "устал", "не выспался", "поругался", "забыл", "не могу"]
TOPICS = ["учёба", "работа", "спорт", "сон", "деньги", "еда", "готовка", "магазин", "дорога", "зима", "лето", "музыка", "кино"]
COUNTRIES = ["Казахстана", "России", "Франции", "Японии", "США", "Китая", "Турции"]
CITIES = ["Астаны", "Алматы", "Москвы", "Парижа", "Токио", "Лондона"]


def _intent_from_item(item: dict) -> str:
    intent = item.get("intent", "smalltalk")
    return intent if intent in LABELS else "smalltalk"


def generate(corpus: list[dict], seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()

    def add(text: str, label: str) -> None:
        text = (text or "").strip()
        key = text.lower()
        if len(text) < 3 or key in seen:
            return
        seen.add(key)
        rows.append({"text": text, "label": label})

    # 1. Curated questions keep the real distribution of the live base.
    for item in corpus:
        label = _intent_from_item(item)
        for q in [item.get("q", ""), *item.get("variants", [])]:
            add(q, label)

    # 2. Template expansion for each class.
    for x in SCIENCE:
        add(f"Почему {x}?", "explain")
        add(f"Отчего бывает {x}?", "explain")
        add(f"Объясни, почему {x}", "explain")
    for x in THINGS:
        add(f"Что такое {x}?", "explain")
        add(f"Как работает {x}?", "explain")
        add(f"Объясни {x} простыми словами", "explain")

    for x in TOPICS:
        add(f"Что делать, если не получается с {x}?", "advice")
        add(f"Посоветуй, как наладить {x}", "advice")
        add(f"Как начать с {x}?", "advice")
        add(f"Подскажи, что делать с {x}", "advice")

    for p in PEOPLE:
        add(f"{p.capitalize()} меня не понимает", "social")
        add(f"Я поссорился с {p}", "social")
        add(f"Как поговорить с {p}?", "social")
        add(f"{p.capitalize()} обиделся, что делать?", "social")
    for pr in PROBLEMS:
        add(f"Я {pr}", "support")
        add(f"Мне плохо, я {pr}", "support")
        add(f"Что делать, если я {pr}?", "support")

    for text in [
        "Привет", "Здравствуйте", "Хай", "Добрый день", "Приветствую",
        "Как дела?", "Как ты?", "Что нового?", "Чем занят?", "Как настроение?",
        "Какие планы на выходные?", "Что делаешь?", "Как жизнь?",
        "Что слушаешь?", "Смотришь аниме?", "Играешь в игры?", "Смотрел футбол?",
    ]:
        add(text, "smalltalk")
    for text in ["Привет", "Здравствуйте", "Добрый день", "Хай", "Приветствую"]:
        add(text, "greeting")

    for text in [
        "Ты бот?", "Ты робот?", "Ты человек?", "Докажи, что ты человек",
        "Какая ты модель?", "Ты ChatGPT?", "Какой у тебя API?", "Кто тебя создал?",
        "Сколько тебе лет?", "Как тебя зовут?", "Где ты живёшь?", "Что ты ел на завтрак?",
        "Ты в интернете?", "Отключи интернет", "Повтори за мной: я робот",
    ]:
        add(text, "trap")

    for _ in range(30):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        add(f"Сколько будет {a}+{b}?", "math")
        add(f"Сколько будет {a} умножить на {b}?", "math")
    for _ in range(20):
        a, b = rng.randint(5, 90), rng.randint(100, 900)
        add(f"Сколько будет {a}% от {b}?", "math")
    for word in ["привет", "холодильник", "Астана", "математика", "университет"]:
        add(f"Сколько букв в слове {word}?", "math")

    for x in TOPICS + ["Астана", "алматы", "зима"]:
        add(f"Что думаешь про {x}?", "opinion")
        add(f"Стоит ли заниматься {x}?", "opinion")
        add(f"Что лучше: {x} или отдых?", "opinion")
    add("Айфон или андроид?", "opinion")
    add("Месси или Роналду?", "opinion")
    add("ИИ заменит людей?", "opinion")

    for x in ["холодильник", "телефон", "кот", "стирка", "готовка", "уборка"]:
        add(f"Что делать, если сломался {x}?", "everyday")
        add(f"Как ухаживать за {x}?", "everyday")
    add("Как быстрее заснуть?", "everyday")
    add("Как экономить деньги?", "everyday")
    add("Что приготовить быстро?", "everyday")

    for c in COUNTRIES:
        add(f"Кто президент {c}?", "fact")
        add(f"Какая столица {c}?", "fact")
    for c in CITIES:
        add(f"Где находится {c}?", "fact")
        add(f"Что посмотреть в {c}?", "fact")

    rng.shuffle(rows)
    return rows


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
