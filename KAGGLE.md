# Kaggle: обучение и железобетонное сохранение артефактов

Kaggle — основной путь обучения Happy. Локальный запуск остаётся только для
smoke-теста и отладки. Здесь описано, как сделать так, чтобы **падение сессии
не стоило больше одной эпохи**.

---

## 0. Что уже готово

| Компонент | Файл |
|---|---|
| Обучение с нуля (BPE, intent, bi-encoder, индекс, метрики) | `scripts/train.py` |
| Чекпоинт + автопуш на HF после каждой эпохи | `scripts/train.py` + `src/hf_store.py` |
| Ноутбук для Kaggle | `kaggle/train_kaggle.ipynb` |
| Метаданные для `kaggle kernels push` | `kaggle/kernel-metadata.json` |
| Скачивание и проверка артефактов | `scripts/pull_model.py` |

В ноутбуке **нет логики обучения** — он клонирует репозиторий и запускает
`scripts/train.py`. Это важно для техпроверки: организаторы видят тот же код,
что и в репозитории.

---

## 1. Один раз настроить

### Kaggle

1. Аккаунт Kaggle.
2. **Подтвердить телефон** (Account → Settings → Phone Verification).
   Без этого не включается GPU и интернет в ноутбуке.
3. API-токен: Account → **Create New Token** → скачается `kaggle.json`.
   Положить в `~/.kaggle/kaggle.json` и `chmod 600`.

### Hugging Face

1. Аккаунт HF.
2. Settings → **Access Tokens** → New token → тип **Write**.
3. Создать **model**-репозиторий, например `mkstudio/happy-artifacts`, **private**.
   (Скрипт создаст его сам, если токен с write.)

### GitHub

Репозиторий с кодом, например `mkstudio/happy`.
Для Kaggle удобнее **публичный** — тогда не нужен `GH_TOKEN`.
Секретов в коде нет, поэтому публичность безопасна.

---

## 2. Секреты в Kaggle

В ноутбуке: **Add-ons → Secrets → Add a new secret**.

| Имя | Значение | Обязательно |
|---|---|---|
| `HF_TOKEN` | `hf_...` (write) | да |
| `HF_REPO` | `mkstudio/happy-artifacts` | да |
| `GH_REPO` | `mkstudio/happy` | да |
| `GH_TOKEN` | GitHub token (только для private repo) | нет |

Ноутбук читает их через `UserSecretsClient`, в код они не попадают.

---

## 3. Запуск

Ноутбук сначала скачивает и фильтрует 100 тыс. открытых русских диалоговых пар
(`scripts/fetch_data.py`) — они нужны только для обучения bi-encoder и **никогда
не выдаются как ответы**. Если скачивание не удалось, обучение продолжится на
курируемом корпусе.

### Вариант A — через веб-интерфейс

1. Kaggle → **Create → Notebook**.
2. **File → Import Notebook** → `kaggle/train_kaggle.ipynb`.
3. Справа: **Accelerator = GPU T4 x2**, **Internet = On**.
4. Add-ons → Secrets (см. выше).
5. **Save & Run All (Commit)**.
6. Можно закрыть ноутбук: commit выполняется в отдельной сессии.

### Вариант B — через CLI (удобнее для повторных прогонов)

```bash
pip install kaggle
# отредактируй kaggle/kernel-metadata.json: "id": "ТВОЙ_НИК/happy-train"

kaggle kernels push -p kaggle
kaggle kernels status ТВОЙ_НИК/happy-train
kaggle kernels output ТВОЙ_НИК/happy-train -p kaggle_out   # третья копия
```

`kaggle kernels output` — это третья копия артефактов, независимая от HF.

---

## 4. Лестница сохранения (почему ничего не потеряется)

| Уровень | Что | Когда |
|---|---|---|
| 1 | Атомарный чекпоинт в `artifacts/checkpoints/` | после каждой эпохи |
| 2 | Пуш этого чекпоинта в HF: `{run}/checkpoints/*.pt` | сразу после эпохи |
| 3 | Финальные веса + `manifest.json` + `latest.json` | в конце |
| 4 | `happy-artifacts-{run}.zip` в `/kaggle/working` | в конце |
| 5 | `kaggle kernels output` локально | по желанию |
| 6 | `scripts/pull_model.py` с проверкой sha256 | перед демо |

Если сессия умирает на 5-й эпохе:
- эпохи 1–5 уже в HF;
- запускаешь тот же ноутбук с `--resume`;
- обучение продолжается с 6-й, а не с нуля.

**Важно:** `/kaggle/working` сохраняется как output только когда commit
завершился успешно. Поэтому уровень 2 (HF) — единственный, который переживает
падение сессии гарантированно.

---

## 5. Что делать, если упало

1. Открой **тот же** ноутбук.
2. Убедись, что `RUN = "kaggle-main"` не менялся.
3. Снова **Save & Run All (Commit)**.
4. В логах должно быть `resumed from epoch N`.
5. Если HF недоступен — в логах `hf push: OFF`; тогда скачай output ноутбука
   и продолжай локально.

---

## 6. Забрать артефакты на MacBook

```bash
export HF_TOKEN=hf_...
python scripts/pull_model.py --repo mkstudio/happy-artifacts
```

Скрипт:
- читает `latest.json` → понимает, какой run последний;
- качает `manifest.json`;
- качает каждый файл и сверяет sha256;
- уже скачанные файлы с правильным хешем пропускает.

После этого `artifacts/` готов к **полностью офлайн** запуску:

```bash
python scripts/preflight.py
python run_ui.py
```

---

## 7. Ограничения Kaggle (актуально на 2026)

| Параметр | Значение |
|---|---|
| GPU-квота | ~30 часов в неделю |
| Максимальная сессия | 12 часов (CPU/GPU), 9 часов (TPU) |
| Idle в интерактивной сессии | ~20 минут |
| `/kaggle/working` | 20 ГБ, сохраняется как output |
| Размер датасета | до 200 ГБ |
| Интернет | только после подтверждения телефона |
| Commit | встаёт в очередь; может ждать 10–60+ минут |

Наши модели крошечные: без больших диалоговых корпусов полный прогон —
**~1–3 минуты**. С 100 тыс. диалоговых пар — **15–40 минут** на T4. Kaggle нужен не для скорости, а для
воспроизводимого облачного прогона и независимого бэкапа.

---

## 8. Smoke-тест всей цепочки (сделай первым делом)

Перед большим прогоном проверь, что HF-пуш работает. В ноутбуке замени команду на:

```python
cmd = [
    sys.executable, "scripts/train.py",
    "--smoke",
    "--out", "artifacts_smoke",
    "--run-name", "smoke",
    "--hf-repo", HF_REPO,
    "--push-every-epoch",
]
```

Через ~3 минуты в HF должны появиться `smoke/vocab.json`, `smoke/DONE`,
`smoke/manifest.json`. Если появились — цепочка рабочая, запускай полный прогон.

---

## 9. Частые ошибки

| Симптом | Причина | Решение |
|---|---|---|
| `401 Unauthorized` | токен без write или не тот | пересоздать HF-токен, обновить secret |
| `Internet is disabled` | не подтверждён телефон | Account → Phone Verification |
| `CUDA out of memory` | слишком большой batch | `--batch-size 64` |
| `No space left on device` | 20 ГБ output | удалить старые чекпоинты в ноутбуке |
| `repo not found` при clone | private repo без токена | добавить `GH_TOKEN` или сделать repo публичным |
| `resume` не сработал | другой `run-name` | всегда `kaggle-main` |
| `fetch_data` упал | сеть/лимит HF | обучение продолжится; повтори позже |
| Commit «Queued» долго | очередь Kaggle | подождать; ничего не перезапускать |
| HF push падает, обучение идёт | сеть/лимит HF | чекпоинты локально; повторить позже |

---

## 10. Чек-лист перед 26 ноября

- [ ] Kaggle: телефон подтверждён, GPU и Internet включаются
- [ ] `HF_TOKEN` (write) и `HF_REPO` добавлены в Secrets
- [ ] Smoke-прогон дошёл до `smoke/DONE` в HF
- [ ] Полный прогон: `kaggle-main/DONE` + `manifest.json`
- [ ] `pull_model.py` скачал артефакты и все sha256 совпали
- [ ] `preflight.py` проходит **с выключенным Wi-Fi**
- [ ] `kaggle kernels output` скачан как третья копия
- [ ] ZIP артефактов лежит на флешке
