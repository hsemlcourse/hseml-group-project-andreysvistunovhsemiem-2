# ML Project — Предсказание дефолта по кредитной карте

**Студент:** Свистунов Андрей

**Группа:** БИВ239

Бинарная классификация: предсказать, допустит ли клиент дефолт по платежу в следующем месяце, на основе демографии, кредитного лимита, истории платежей и выписок (датасет UCI "Default of Credit Card Clients", Taiwan, 30 000 клиентов).

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)

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
├── models                      # Сохранённые модели
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель (LogReg без FE)
│   └── 03_experiments.ipynb    # Сравнение 5 моделей + ансамблей
├── presentation
├── report
│   ├── images
│   └── report.md
├── src
│   ├── config.py               # Pydantic-settings (.env)
│   ├── data.py                 # Загрузка датасета с Kaggle
│   ├── preprocessing.py        # Очистка + feature engineering + ColumnTransformer
│   ├── metrics.py              # Обёртка метрик
│   └── modeling.py             # Baseline-пайплайн
├── tests
│   └── test.py
├── pyproject.toml              # uv + ruff + pytest
├── uv.lock
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

# 4. Запустить тесты и линтер
uv run ruff check src/
uv run pytest -q

# 5. Обучить baseline + прогнать все эксперименты (LogReg/KNN/RF/XGBoost/LightGBM),
#    сохраняет models/*.joblib и печатает сводную таблицу метрик.
uv run python -m src.modeling

# 6. Запустить ноутбуки
uv run jupyter lab
#   или неинтерактивно:
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## Данные

- `data/raw/UCI_Credit_Card.csv` — исходник с Kaggle (в git не коммитится).
- `data/processed/` — промежуточные артефакты, если понадобятся.

## Результаты

Оценка на стратифицированном test (20%, seed=42):

| Модель              | ROC-AUC | PR-AUC | F1    | Recall | Accuracy |
|---------------------|---------|--------|-------|--------|----------|
| DummyClassifier     | 0.500   | 0.221  | 0.221 | 0.220  | 0.657    |
| KNN (k=25)          | 0.749   | 0.507  | 0.436 | 0.330  | 0.811    |
| LogReg (baseline)   | 0.763   | 0.536  | 0.529 | 0.566  | 0.777    |
| RandomForest (400)  | 0.764   | 0.530  | 0.449 | 0.346  | 0.812    |
| LightGBM (400)      | 0.766   | 0.541  | 0.518 | 0.548  | 0.774    |
| LogReg + FE         | 0.768   | 0.537  | 0.520 | 0.601  | 0.754    |
| **XGBoost (400)**   | **0.774** | **0.551** | 0.468 | 0.363 | 0.818 |

XGBoost лидирует по ROC-AUC. В CP2 — тюнинг гиперпараметров, подбор порога под recall, PCA, калибровка.

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md).
