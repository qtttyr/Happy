"""Fill the official DOCX template with the project data.

Personal data lives in private/form_values.json (gitignored).
    python scripts/fill_form.py
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT.parent / "info" / "Тьюринг Арена - техническая форма (название команды).docx"
VALUES = ROOT / "private" / "form_values.json"
OUT = ROOT / "private" / "Тьюринг Арена - техническая форма (mk studio).docx"

ARCH_51 = """1. Нормализация: нижний регистр, ё→е, очистка пунктуации, токенизация, лемматизация pymorphy3, ключевые слова. Отдельно работают детерминированные решатели: арифметика, проценты, буквы, даты, последовательности, площадь/периметр, время.
2. Маршрутизация: решатель / защита персоны («ты бот?») / поиск по базе / безопасный фолбэк.
3. Гибридный поиск: собственный BM25 (инвертированный индекс) + косинус по векторам собственного Siamese bi-encoder. Итог: 0.55*BM25 + 0.45*cosine + небольшой буст по интенту.
4. Гейт уверенности: precision-first. При слабом совпадении бот отвечает человеческим «не знаю», а не уверенной ошибкой.
5. Персона и память: один персонаж (Мирас, 17 лет, Астана, AITU, первый курс) и антиповтор в рамках раунда.
6. Humanizer: убирает книжный стиль, варьирует регистр, добавляет «)», держит длину ≤300, считает человеческую паузу печати 2–9 секунд.
Гарантия: три уровня фолбэка и финальная PANIC-фраза. На 400 тестовых вопросах — 0 пустых ответов."""

ARCH_52 = """Собственный BM25; собственный BPE-токенизатор (2 000 слияний); IntentClassifier (BiLSTM + attention pooling); Siamese bi-encoder (BiLSTM, mean+max pooling, 128-d, контрастивная функция InfoNCE с in-batch негативами); детерминированные решатели; композер фраз; персона, память раунда, humanizer. Гибридный подход: нейросети + поиск + правила + шаблоны."""

ARCH_53 = """Предобученные веса не используются: нет transformers, sentence-transformers, word2vec, внешних LLM API. Токенизатор, IntentClassifier (BiLSTM + attention) и bi-encoder (Siamese BiLSTM, 128-d) обучены с нуля на корпусе проекта: 1 142 записи / 2 403 формулировки, 2 693 размеченных примера интентов, 3 746 контрастивных пар. Артефакты: vocab.json, intent_model.pt, bi_encoder.pt, bm25.pkl, corpus.jsonl, corpus_vectors.npy. Скрипт обучения: scripts/train.py. Метрики: intent accuracy 0.733; contrastive loss 0.694; Recall@1 0.832, Recall@5 0.929; p50 1.0 мс, p95 1.7 мс; 0 пустых ответов."""

DATA_61 = """Авторский курируемый корпус (data/src/*.txt → data/kb_*_qa.jsonl; 1 142 записи, 2 403 формулировки) — ответы, голос, контрольные вопросы. Авторский data/curated_qa.jsonl — ядро, персона, ловушки. Den4ikAI/russian_dialogues (MIT) — только обучение bi-encoder, не выдача ответов; фильтр длины, кириллицы, ссылок, личных данных и мата. Данные, сгенерированные чужими LLM, не используются. SberQuAD не используется."""

DATA_62 = """Python 3.12.16; numpy ≥2.0 (векторы, индекс); pymorphy3 ≥2.0 (морфологический словарь); torch ≥2.4 (обучение собственных сетей, dense-режим); huggingface-hub ≥0.30 (выгрузка артефактов, не рантайм). Полный список — requirements.txt и requirements-dense.txt в GitHub. transformers и внешние LLM API не используются."""

RUN_71 = """python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dense.txt
python scripts/pull_model.py --repo vaprooll/Happy  (или артефакты уже в репозитории)
python scripts/preflight.py  (проверка офлайн-режима, сокеты запрещены)
python run_ui.py  → http://127.0.0.1:8000/
python run_cli.py  — терминальный режим
После установки зависимостей интернет не нужен."""


def set_text(paragraph, text: str) -> None:
    for run in paragraph.runs:
        run.text = ""
    run = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
    parts = text.split("\n")
    run.text = parts[0]
    for part in parts[1:]:
        run.add_break()
        run.add_text(part)


def main() -> None:
    values = json.loads(VALUES.read_text(encoding="utf-8"))
    doc = Document(str(TEMPLATE))

    replacements = {
        "Название команды:": "Название команды: mk studio",
        "Формат участия:": "Формат участия: индивидуально (команда из одного участника)",
        "Образовательная организация:": "Образовательная организация: Cardiff University Kazakhstan",
        "Город:": "Город: Астана",
        "Контактное лицо (капитан команды):": f"Контактное лицо (капитан команды): {values['fio']}",
        "Электронная почта:": f"Электронная почта: {values['email']}",
        "Телефон:": f"Телефон: {values['phone']}",
        "Участник 2:": "Участник 2: — (участие индивидуальное)",
        "Участник 3:": "Участник 3: — (участие индивидуальное)",
        "Язык программирования и версия:": "Язык программирования и версия: Python 3.12.16",
        "Операционная система:": "Операционная система: macOS 27.2.1 (Apple Silicon)",
        "Видеодемонстрация на YouTube:": f"Видеодемонстрация на YouTube: {values['youtube']}",
        "Репозиторий GitHub:": "Репозиторий GitHub: https://github.com/qtttyr/Happy",
        "Дополнительные материалы (при наличии):": f"Дополнительные материалы (при наличии): {values['extra']}",
        "[Опишите основные компоненты и порядок обработки вопроса]": ARCH_51,
        "[Правила, поиск, собственная нейросеть, шаблоны, гибридный подход и т. п.]": ARCH_52,
        "[Если применимо: архитектура, автор, способ обучения, источник начальных весов; иначе «Не используются»]": ARCH_53,
        "[Названия и источники наборов данных, назначение, способ подготовки, лицензия при наличии]": DATA_61,
        "[Основные библиотеки, версии и назначение; полный список зависимостей — в GitHub]": DATA_62,
        "[Команды запуска, необходимые локальные файлы, порядок установки; подробный README — в GitHub]": RUN_71,
    }

    used: set[str] = set()
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        for prefix, replacement in replacements.items():
            if text.startswith(prefix):
                set_text(paragraph, replacement)
                used.add(prefix)
                break

    missing = set(replacements) - used
    if missing:
        print("WARNING: not matched:", missing)

    OUT.parent.mkdir(exist_ok=True)
    doc.save(str(OUT))
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
