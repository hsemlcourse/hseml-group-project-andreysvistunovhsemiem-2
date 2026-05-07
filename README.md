# ML Project — Предсказание дефолта по кредитной карте

**Студент:** Свистунов Андрей

**Группа:** БИВ238

Бинарная классификация: предсказать, допустит ли клиент дефолт по платежу в следующем месяце, на основе демографии, кредитного лимита, истории платежей и выписок (датасет UCI "Default of Credit Card Clients", Taiwan, 30 000 клиентов).

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Docker](#docker)
5. [Данные](#данные)
6. [Результаты](#результаты)
7. [Отчёт](#отчёт)
8. [Чекпоинты](#чекпоинты)

## Описание задачи

- **Задача:** бинарная классификация.
- **Датасет:** [UCI Credit Card Default (Taiwan)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset), 30 000 строк × 25 колонок.
- **Таргет:** `default.payment.next.month` (0/1). Дисбаланс ~78/22%.
- **Основная метрика:** `ROC-AUC` — устойчива к дисбалансу, отражает качество ранжирования клиентов по риску. Доп.: `PR-AUC`, `F1`, `Recall` класса 1 (пропущенный дефолт обходится банку дороже ложной тревоги).

## Структура репозитория

```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы (не коммитится)
├── models                      # Сохранённые модели (baseline + tuned_*)
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline (LogReg без FE)
│   ├── 03_experiments.ipynb    # Сравнение 5 моделей + ансамблей (CP1)
│   └── 04_tuning.ipynb         # CV / тюнинг / PCA / калибровка / интерпретируемость (CP2)
├── presentation
├── report
│   ├── images
│   ├── tuning_results.csv      # таблица CP2 (CV-AUC + test-метрики)
│   ├── tuning_best_params.json # лучшие гиперпараметры
│   └── report.md
├── src
│   ├── config.py               # Pydantic-settings (.env)
│   ├── data.py                 # Загрузка с Kaggle (kagglehub)
│   ├── preprocessing.py        # clean + feature_engineering + ColumnTransformer
│   ├── modeling.py             # Baseline + 5 моделей на едином сплите
│   ├── cv.py                   # StratifiedKFold-обёртка
│   ├── tuning.py               # RandomizedSearchCV для 4 кандидатов
│   ├── threshold.py            # Подбор порога + калибровка
│   ├── dim_reduction.py        # PCA-эксперимент
│   ├── interpret.py            # Feature / permutation importance
│   ├── metrics.py              # Обёртка метрик
│   ├── finalize.py             # Финальная модель: tuned + isotonic calibration
│   ├── serve.py                # Scorer + Pydantic ClientPayload
│   ├── api.py                  # FastAPI: /health, /predict, /predict_batch
│   └── ui.py                   # Streamlit-форма скоринга
├── scripts
│   └── build_report_pdf.sh     # md → html → pdf через pandoc + Chrome
├── tests
│   ├── test.py                 # Тесты CP1 (анти-leakage, FE, smoke run)
│   ├── test_cp2.py             # Тесты CP2 (cv, threshold, pca, importances)
│   └── test_api.py             # Тесты CP3 (FastAPI TestClient)
├── Dockerfile                  # python:3.11-slim + uv (frozen lock)
├── docker-compose.yml          # сервисы train / tune / test
├── pyproject.toml              # uv + ruff + pytest (markers)
├── uv.lock
├── REQUIREMENTS.md             # Зеркало требований курса
├── .env.example
└── README.md
```

## Запуск

Менеджер пакетов — [`uv`](https://docs.astral.sh/uv/). Установка: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
# 1. Установить зависимости (создаст .venv и развернёт окружение из uv.lock)
uv sync --group dev

# 2. (опционально) настроить Kaggle API
cp .env.example .env
# и прописать KAGGLE_USERNAME / KAGGLE_KEY (или положить ~/.kaggle/kaggle.json)

# 3. Скачать датасет в data/raw/UCI_Credit_Card.csv
uv run python -m src.data

# 4. Линтер + тесты
uv run ruff check src/ tests/
uv run pytest -q

# 5. Baseline + сравнение 5 моделей (CP1)
uv run python -m src.modeling

# 6. Гиперпараметрический поиск + сохранение лучших пайплайнов (CP2)
uv run python -m src.tuning

# 7. Ноутбуки
uv run jupyter lab
#  или неинтерактивно:
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## Docker

Воспроизводимая сборка через uv + lock-файл. Артефакты (data/, models/, report/) монтируются с хоста.

```bash
docker compose build
docker compose run --rm train     # обучение CP1
docker compose run --rm tune      # тюнинг CP2
docker compose run --rm test      # pytest
```

## Данные

- `data/raw/UCI_Credit_Card.csv` — исходник с Kaggle (в git не коммитится).
- `data/processed/` — промежуточные артефакты, если понадобятся.

## Результаты

### CP1: дефолтные модели (test, seed=42)

| Модель              | ROC-AUC | PR-AUC | F1    | Recall | Accuracy |
|---------------------|---------|--------|-------|--------|----------|
| DummyClassifier     | 0.500   | 0.221  | 0.221 | 0.220  | 0.657    |
| KNN (k=25)          | 0.749   | 0.507  | 0.436 | 0.330  | 0.811    |
| LogReg (baseline)   | 0.763   | 0.536  | 0.529 | 0.566  | 0.777    |
| RandomForest (400)  | 0.764   | 0.530  | 0.449 | 0.346  | 0.812    |
| LightGBM (400)      | 0.766   | 0.541  | 0.518 | 0.548  | 0.774    |
| LogReg + FE         | 0.768   | 0.537  | 0.520 | 0.601  | 0.754    |
| **XGBoost (400)**   | **0.774** | **0.551** | 0.468 | 0.363 | 0.818 |

### CP2: результаты тюнинга

Таблица обновляется при запуске `uv run python -m src.tuning` и сохраняется в `report/tuning_results.csv`. Подробности и интерпретация — в [`report/report.md`](report/report.md).

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md).

## CP3 — деплой

После того как `models/final_model.joblib` и `models/threshold.json` лежат на месте
(они идут в репозитории; пересобрать можно через `uv run python -m src.tuning && uv run python -m src.finalize`):

```bash
# FastAPI с авто-сваггером на /docs (порт 8000)
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000

# Streamlit-форма (порт 8501)
uv run streamlit run src/ui.py

# Через docker compose: api + ui (ui ждёт healthcheck api)
docker compose up api ui
# или с пересборкой: docker compose up --build api ui
```

Эндпоинты: `GET /health`, `POST /predict`, `POST /predict_batch`. Тесты — `tests/test_api.py`.

Сборка PDF-версии отчёта (требует pandoc и Chrome/Chromium):

```bash
./scripts/build_report_pdf.sh   # пишет report/report.pdf
```

## Чекпоинты

- `cp1` — baseline + 5 моделей, отчёт §1–5.
- `cp2` — CV, тюнинг, PCA, калибровка, порог, интерпретируемость, Docker.
- `cp3` — текущая ветка: FastAPI + Streamlit + финальный артефакт + PDF-отчёт.

Полное описание требований к каждому чекпоинту — [`REQUIREMENTS.md`](REQUIREMENTS.md).
