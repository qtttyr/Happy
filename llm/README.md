# Happy LLM — отдельный эксперимент

Маленькая **собственная** языковая модель, обученная с нуля на русских
диалогах. Отдельная папка, не трогает рабочего бота в `src/`.

> Это не замена основному боту. Это отдельный трек: «мы умеем не только
> retrieval, но и свою генеративную модель». В продакшене она подключается
> только через quality gate.

---

## Что внутри

| Компонент | Файл |
|---|---|
| Модель (decoder-only Transformer) | `llm/model.py` |
| Токенизатор (BPE 16k с нуля) | `llm/tokenizer_utils.py` |
| Подготовка данных | `llm/prepare_data.py` |
| Обучение с чекпоинтами и resume | `llm/train.py` |
| Сэмплинг + sanity gate | `llm/sample.py` |
| Kaggle | `llm/kaggle/train_llm.ipynb` |

Конфиг по умолчанию: 6 слоёв, 6 голов, 384 embedding, context 256, ~17M
параметров. Никаких предобученных весов — всё с нуля.

---

## Данные

| Источник | Что | Лицензия |
|---|---|---|
| `Den4ikAI/russian_dialogues` | 2M пар Telegram, casual | MIT |
| `Den4ikAI/russian_dialogues_2` | 1.6M многоходовых | MIT |
| `Happy` curated KB | 1.2k записей / 2.5k формулировок + identity | авторский |

Формат потока:

```
<|user|> вопрос <|bot|> ответ <|end|>
```

Фильтр: русский, длина, без ссылок/мата, без повторов. Данные, сгенерированные
чужими LLM, не используются.

---

## Локальный smoke

```bash
cd "/Users/miras/Desktop/Mira AI/Happy/happy"
.venv/bin/python llm/prepare_data.py --source local --max-pairs 3000 --smoke --out llm/data_smoke
.venv/bin/python llm/train.py --smoke --data-dir llm/data_smoke --no-push
.venv/bin/python llm/sample.py --run llm-main --tokenizer llm/data_smoke/tokenizer.json
```

---

## Kaggle (полный прогон)

1. Создать HF model-репозиторий `vaprooll/Happy-LLM` (private).
2. Секреты в Kaggle: `HF_TOKEN`, **`HF_REPO_LLM=vaprooll/Happy-LLM`**, `GH_REPO=qtttyr/Happy`.
3. Импортировать `llm/kaggle/train_llm.ipynb`, GPU T4 x2, Internet On.
4. `SMOKE = True` — проверка цепочки (3–5 минут).
5. `SMOKE = False` — полный прогон.
6. **Save & Run All (Commit)** и можно закрыть ноутбук.

После прогона:

```bash
python llm/pull.py --repo vaprooll/Happy-LLM --run llm-main
python llm/sample.py --run llm-main --tokenizer llm/artifacts/llm-main/tokenizer.json
```

### Почему ничего не теряется

- каждые 500 шагов — чекпоинт в `/kaggle/working/llm/checkpoints/{run}/`;
- каждые 1000 шагов — пуш в HF: `{run}/ckpt/latest.pt`;
- при падении сессии `--resume` подтягивает последний чекпоинт и продолжает;
- в конце — `llm_model.pt`, `tokenizer.json`, `metrics.json`, `samples.txt`, `DONE`.

`--max-hours 8` останавливает обучение заранее, чтобы успеть сохранить всё
до лимита Kaggle в 12 часов.

---

## Ожидания по качеству

Модель ~17M параметров на 20–40M токенов — это **не ChatGPT**. Она может
выдавать короткие живые реплики, но иногда будет путаться. Поэтому:

- сэмплы смотреть глазами;
- в основной бот подключать только через quality gate;
- если качество не устроит — оставить как исследовательский трек и показать
  на презентации честные метрики/примеры.

Это нормальный инженерный результат: мы обучили свою модель с нуля и измерили
её, а не прикрутили чужую LLM.
