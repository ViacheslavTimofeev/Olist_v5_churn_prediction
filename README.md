# Olist Churn Prediction

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#) [![Docs](https://img.shields.io/badge/docs-Sphinx-informational.svg)](#) [![MLflow](https://img.shields.io/badge/tracking-MLflow-lightgrey.svg)](#)

Предсказание **оттока клиентов** для бразильского e‑commerce Olist. Репозиторий оформлен по шаблону Cookiecutter Data Science и расширен CLI‑скриптами (Typer), конфигами YAML, валидацией данных и трекингом экспериментов через MLflow.

---

## 🔑 Ключевые возможности
- **Валидация входящих данных** через YAML‑манифест (без pydantic), отчёт об отклонениях.
- **Предобработка и фичеинжиниринг**: объединения таблиц, выбор столбцов, обработка пропусков, нормализация категориальных признаков.
- **Создание таргета `churned`** по окну неактивности (например, >120 дней).
- **Бейзлайн‑модели** (scikit‑learn Pipelines, ColumnTransformer).
- **Кросс‑валидация и логирование метрик** (F1, Recall, ROC‑AUC и др.) в **MLflow**.
- **Документация Sphinx** с автогенерацией API‑разделов.

---

## 🗂️ Структура проекта
```
├── LICENSE
├── Makefile
├── README.md
├── data/
│   ├── raw/            # исходные таблицы Olist
│   ├── interim/        # промежуточные таблицы после join/clean
│   └── processed/      # готовые к моделированию датасеты
├── docs/               # Sphinx-проект (источники RST)
├── models/             # сериализованные модели/предсказания (опционально)
├── notebooks/          # исследования и EDA
├── reports/
│   └── figures/        # графики и иллюстрации
├── requirements.txt
├── setup.py            # pip install -e .
├── configs/
│   ├── preprocessing.yaml
│   └── baseline.yaml
├── validations/
│   ├── manifest.yaml
│   └── suites/         # правила/профили для отдельных датасетов
├── mlruns/             # артефакты MLflow (локально)
└── src/
    └── olist_churn_prediction/
        ├── __init__.py
        ├── feature_processing.py
        ├── targets.py
        ├── validator_cli.py         # Typer CLI: валидация по manifest.yaml
        ├── preprocessing_cli.py     # Typer CLI: join/select/drop/clean
        └── baseline_cli.py          # Typer CLI: CV/train/predict по YAML
```

> Папки `mlruns/` и большие бинарные артефакты рекомендуется добавить в `.gitignore`.

---

## 🚀 Быстрый старт

### 1) Установка окружения
```bash
# conda (рекомендуется)
conda create -n olist-ml python=3.10 -y
conda activate olist-ml

# зависимости
pip install -r requirements.txt
# локальная установка пакета
pip install -e .
```

### 2) Данные
Скачайте таблицы Olist и поместите **как есть** в `data/raw/`. Пути к файлам указываются в ваших YAML‑конфигах.

### 3) Валидация входящих данных
Манифест описывает, какие таблицы проверять и какие пороги использовать.
```yaml
# validations/manifest.yaml (пример)
defaults:
  null_delta_pp: 5.0     # допустимое изменение доли пропусков (в п.п.)
  new_cat_ratio: 0.02    # доля новых категорий

datasets:
  - name: customers
    path: data/raw/olist_customers_dataset.csv
    suite: validations/suites/customers.yaml
  - name: orders
    path: data/raw/olist_orders_dataset.csv
    suite: validations/suites/orders.yaml
```
Запуск:
```bash
python -m olist_churn_prediction.validator_cli validate-all \
  --manifest validations/manifest.yaml
```

### 4) Предобработка и фичи
Пример конфига для объединений, выбора и очистки:
```yaml
# configs/preprocessing.yaml
joins:
  - left: data/raw/olist_orders_dataset.csv
    right: data/raw/olist_customers_dataset.csv
    on: [customer_id]
    how: left
  - left: <prev>
    right: data/raw/olist_order_items_dataset.csv
    on: [order_id]
    how: left

select_columns:
  keep: [order_id, customer_id, product_id, customer_state,
         product_category_name_english, order_purchase_timestamp]

ops:
  - drop_columns: [some_redundant_col]
  - dropna_rows:  {subset: null}   # удалить строки с любыми NaN

output:
  interim_path: data/interim/master_basic.parquet
  processed_path: data/processed/baseline.parquet
```
Запуск:
```bash
python -m olist_churn_prediction.preprocessing_cli run \
  --config configs/preprocessing.yaml
```

### 5) Создание целевой переменной `churned`
Если генерация таргета делается на этапе предобработки — используйте соответствующий шаг в `preprocessing_cli`.
Или добавьте после предобработки (пример в ноутбуке/скрипте):
```python
import pandas as pd

df = pd.read_parquet("data/processed/baseline.parquet")
ref = df["order_purchase_timestamp"].max()
last = df.groupby("customer_id")["order_purchase_timestamp"].max().reset_index()
last["days_since_last_order"] = (ref - last["order_purchase_timestamp"]).dt.days
last["churned"] = (last["days_since_last_order"] > 120).astype(int)  # важно: astype(int)

df = df.merge(last[["customer_id","churned"]], on="customer_id", how="left")
df["churned"].fillna(0, inplace=True)  # при необходимости
```
Сохраните обновлённый датасет в `data/processed/`.

### 6) Бейзлайн‑моделирование и кросс‑валидация
```yaml
# configs/baseline.yaml (пример)
random_state: 42
cv:
  folds: 5
  shuffle: true

features:
  numeric: [feat_1, feat_2, feat_3]
  categorical: [customer_state, product_category_name_english]

model:
  name: rf
  params:
    n_estimators: 300
    max_depth: 12
    n_jobs: -1

metrics: [f1, recall, roc_auc]
log_model_artifact: false   # чтобы не сохранять тяжелый pickle
```
Запуск CV/Train:
```bash
python -m olist_churn_prediction.baseline_cli cv     --config configs/baseline.yaml
python -m olist_churn_prediction.baseline_cli train  --config configs/baseline.yaml
```

### 7) Трекинг экспериментов (MLflow)
```bash
mlflow ui --backend-store-uri mlruns
# затем откройте http://127.0.0.1:5000
```
> Советы: избегайте логирования больших артефактов; держите `mlruns/` вне гита.

### 8) Документация
```bash
# Linux/Mac
make -C docs html
# Windows (PowerShell)
cd docs; .\make.bat html
```
Собранные HTML‑страницы появятся в `docs/_build/html`.

---

## ⚙️ Makefile (опционально)
Доступные цели (настройте под себя):
```
make data        # подготовка/загрузка сырых данных
make validate    # валидация по manifest.yaml
make features    # предобработка/фичи
make train       # запуск обучения/кросс-валидации
make docs        # сборка документации
```

---

## 🧪 Тесты и стиль
- `pytest` для модульных тестов (`tests/`).
- `black`, `isort`, `flake8` и `pre-commit` для единого стиля.

---

## 🗺️ Roadmap
- [ ] Расширить пайплайн фичейнжиниринга (лаги, агрегации, OOF target encoding).
- [ ] Добавить Optuna‑тюнинг (XGBoost/LightGBM/CatBoost) с top‑k accuracy и macro‑F1.
- [ ] Экспорт артефактов в S3/MinIO, CI/CD (GitHub Actions), Docker‑образ.
- [ ] Улучшить Sphinx‑доки: API‑разделы, примеры CLI, диаграмма пайплайна.

---

## 🤝 Contributing
PR‑ы приветствуются. Перед коммитом запустите локально `pre-commit`.

---

## 📄 Лицензия
Проект распространяется по лицензии MIT (см. `LICENSE`).

---

## 🙏 Благодарности
- Шаблон: Cookiecutter Data Science.
- Сообщество Olist и открытые датасеты.

