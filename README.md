# Olist Churn Prediction

[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#) [![Docs](https://img.shields.io/badge/docs-Sphinx-informational.svg)](#) [![MLflow](https://img.shields.io/badge/tracking-MLflow-lightgrey.svg)](#)

Предсказание **оттока клиентов** для бразильского e‑commerce Olist. Репозиторий оформлен по шаблону Cookiecutter Data Science и расширен CLI‑скриптами (Typer), конфигами YAML, валидацией данных и трекингом экспериментов через MLflow. Добавлены Jupyter-ноутбуки со всеми этапами проекта.

---

## 🔑 Ключевые возможности
- **Валидация входящих данных** через YAML‑манифест, отчёт об отклонениях.
- **Предобработка и фичеинжиниринг**: объединения таблиц, выбор столбцов, обработка пропусков, нормализация категориальных признаков.
- **Генерация «скелетов»** - правил валидации (suites)
- **Создание таргета `churned`** по окну неактивности (размер окна по выбору, default >120 дней).
- **Baseline‑модели** (scikit‑learn Pipeline, ColumnTransformer).
- **Кросс‑валидация и логирование метрик** (F1, Recall, ROC‑AUC и др.) в **MLflow**.
- **Документация Sphinx** с автогенерацией API‑разделов.

---

## 🗂️ Структура проекта
```
├── LICENSE
├── Makefile
├── README.md
├── artifacts           # артефакты логирования
│   └── baseline
├── configs             # YAML-конфиги
│   ├── baseline.yaml
│   ├── preprocessing_manifest.yaml   # манифест для предобработки
│   └── validation_manifest.yaml   # манифест для валидации
├── data
│   ├── interim         # промежуточные таблицы после join/clean
│   ├── processed       # готовые к моделированию датасеты
│   └── raw             # исходные таблицы Olist
├── docs                # Sphinx-проект (источники RST)
│   ├── Makefile
│   ├── _build          # сборка документации через make
│   ├── api
│   ├── commands.rst    # описание CLI-команд
│   ├── conf.py         # конфиг для сборки html
│   ├── getting-started.rst
│   ├── index.rst
│   └── make.bat
├── features
├── notebooks            # исследования и EDA
│   ├── EDA
│   ├── feature engineering
│   └── preprocessing
├── pyproject.toml
├── references
├── reports
│   └── figures          # графики и иллюстрации
├── setup.py
├── src                  # модули
│   ├── olist_churn_prediction    # локальная библиотека
├── typed_schemas         # типы, к которым нужно привести данные
│   ├── payments_types.yaml
│   ├── product_measures_types.yaml
│   ├── public_customers_types.yaml
│   ├── public_data_types.yaml
│   ├── sellers_types.yaml
│   └── translation_types.yaml
└── validations
    ├── reports
    └── suites                     # "скелеты" правила валидации
```

> Папки `mlruns/` и большие бинарные артефакты рекомендуется добавить в `.gitignore`.

---

## 🚀 Быстрый старт

### 1) Установка окружения
```bash
# из корня репозитория
git clone git@github.com:ViacheslavTimofeev/Olist_v5_churn_prediction.git
cd <ваш репозиторий>

conda env create -f environment.yml
conda activate olist-ml        # или имя из поля `name:` в вашем environment.yml

# привязать окружение к Jupyter
python -m ipykernel install --user --name olist-ml --display-name "olist-ml"

# локальная установка пакета
pip install -e .

# проверка
python -c "import olist_churn_prediction as p; print('OK:', p.__name__)"
```

### 2) Данные
Скачайте таблицы Olist и поместите **как есть** в `data/raw/`. Пути к файлам указываются в ваших YAML‑конфигах.

### 3) Валидация входящих данных
Манифест описывает, какие таблицы проверять и какие пороги использовать.
```yaml
# часть validation_manifest.yaml
defaults:
  cast:
    mode: strict  # проверка состава колонок, идёт против схемы типов (поведение описывается в typed_schemas/*.yaml)
    keep_unknown_columns: false # при false будет ошибка при наличии незадекларированных колонок (поведение описывается в typed_schemas/*.yaml)
    csv_sep: ","
    output_dir: data/interim
  validate:
    null_delta_pp: 5.0  # допустимый рост доли пропусков относительно базовой (из suite)
    new_cat_ratio: 0.02  # доля строк с новыми категориями, при превышении — ошибка
    oob_ratio: 0.01  # доля значений вне числовых/датовых границ (из suite), при превышении — ошибка
    strict_structure: true  # если true, наличие лишних колонок относительно suite — ошибка (отсутствующие колонки — всегда ошибка)
    dayfirst: false  # в разработке
    sample: null  # доля от общего для выборки, null - взять весь датасет

datasets:  # имена датасетов для валидации, отсутствие датасета в списке пропустит для него этап валидации
  - name: public_customers
    path: "data/raw/olist_public_dataset_v2_customers.csv"  # путь к сырому датасету
    cast:
      schema: typed_schemas/public_customers_types.yaml
      output: data/interim/cli_related/typed/public_customers_typed.parquet 
      mode: strict
    validate:
      suite: "validations/suites/public_customers.json"
```
Структура команд для запуска:
```bash
# для одного
python -m olist_churn_prediction.validator_cli validate \
       <относительный путь датасета на вход> \
       <относительный путь "скелета" для сверки>

# для всех сразу
python -m olist_churn_prediction.validator_cli validate-all \
       <относительный путь к манифесту валидации>
```
Пример:
```bash
# для одного
python -m olist_churn_prediction.validator_cli validate \
       data/raw/olist_public_dataset_v2.csv \
       validations/suites/public_data.json

# для всех сразу
python -m olist_churn_prediction.validator_cli validate-all \
       configs/validation_manifest.yaml
```
### 4) Приведение к типам
На основании индивидуальных манифестов для каждого датасета происходит приведение к описанным в них типам данных. Манифесты создаются вручную в папке typed_schemas.
```yaml
# пример схемы для таблицы payments
schema:
  order_id:  # название колонки
    type: string  # тип для приведения. Доступно int, float, string, bool, datetime (можно с timezone), category
  installments:
    type: int
    nullable: false  # при True вернет Int64, при False - int64
  sequential:
    type: int
    nullable: false
  payment_type:
    type: string
  value:
    type: float
  date:
    type: datetime
    format: "%Y-%m-%d %H:%M:%S.%f"  # формат по аналогии с методом pd.to_datetime()
    drop_tz: true
options:
  mode: strict      # strict | coerce
  keep_unknown_columns: false
```
Структура команд для запуска:
```bash
# для одного
python -m olist_churn_prediction.types_cli cast \
       <относительный путь к сырому датасету> \
       --schema <относительный путь к манифесту с типами> \
       --output <относительный путь для сохранения>

# для всех сразу
python -m olist_churn_prediction.types_cli cast-all \
       <относительный путь к манифесту валидации>
```
Пример:
```bash
# для одного
python -m olist_churn_prediction.types_cli cast \
       data/raw/payments_olist_public_dataset.csv \
       --schema typed_schemas/payments_types.yaml \
       --output data/interim/cli_related/typed/payments_typed.parquet

# для всех сразу
python -m olist_churn_prediction.types_cli cast-all \
       configs/validation_manifest.yaml
```

### 5) Предобработка и фичи
Пример конфига для выбора, очистки и объединений.

Чистка и выбор:
```yaml
# часть preprocessing_manifest.yaml
datasets:
  - name: public_data_basic
    input: data/interim/cli_related/typed/public_data_typed.parquet  # относительный путь к типизированному датасету
    sample: null  # доля от всего датасета для взятия (выборка)
    steps:
      - op: drop_columns  # удаление колонок
        cols: [customer_zip_code_prefix]
      - op: drop_duplicates  # удаление дубликатов
        subset: all  # взять все колонки
        keep: first  # оставить первый дубликат
        ignore_index: true
      - op: dropna_columns  # удаление колонок при условии пропусков
        cols: [order_status]
        min_missing_ratio: 0.4  # минимальная доля пропусков для удаления
      - op: dropna_rows  # удаление строк при условии пропусков
        subset: [order_id]
        how: any  # при любом пропуске
      - op: lowercase_categoricals  # приведение к нижнему регистру
        cat_cols: [order_status]
      - op: disambiguate_city_state  # унификация городов (см. документацию preprocessing_cli.py)
        city_col: customer_city
        state_col: customer_state
        suffix_sep: "_"  # разделительный символ
      - op: rename_columns  # переименовывание колонок
        mapping:
          product_name_lenght: product_name_length
          product_description_lenght: product_description_length
      - op: groupby_aggregate  # агрегация по условиям
        by: order_id
        sum_cols:   [order_products_value, order_freight_value]  # суммирование внутри агрегации
        mean_cols:  [product_name_length, product_description_length]  # среднее
        min_cols:   [review_creation_date, review_answer_timestamp]  # минимум
        first_for_rest: true   # все остальные колонки агрегируем как first
    output: data/interim/cli_related/basic/public_data_basic.parquet  # относительный путь для сохранения
```
Структура команд для запуска:
```bash
# для всех сразу
python -m olist_churn_prediction.preprocessing_cli run \
       <относительный путь к манифесту препроцессинга>
```
Запуск:
```bash
# для всех сразу
python -m olist_churn_prediction.preprocessing_cli run \
       configs/preprocessing_manifest.yaml
```
Объединение:
```yaml
# часть preprocessing_manifest.yaml
datasets:
  - name: master_basic
    input: data/interim/cli_related/basic/public_data_basic.parquet  # база
    steps:
      - op: join
        right: data/interim/cli_related/basic/payments_basic.parquet  # какой датасет добавляем в объединение
        on: [order_id]  # по какой колонке объединяем
        how: left  # метод объединения
        suffix_right: _pay  # суффикс для колонок после объединения для совместимости

      - op: join
        right: data/interim/cli_related/basic/product_measures_basic.parquet
        on: [product_id]
        how: left
        suffix_right: _prod
    output: data/processed/cli_related/master_basic.parquet  # объединенный неочищенный датасет

  - name: master_clean  # чистка
    input: data/processed/cli_related/master_basic.parquet
    steps:  # шаги обработки
      - op: drop_duplicates  
        subset: all
      - op: drop_columns
        cols: [review_comment_message, review_comment_title, order_status]
      - op: dropna_rows
        how: any
    output: data/processed/cli_related/master_clean.parquet  # объединенный чистый датасет
```
### 6) Создание целевой переменной `churned`

Вариант 1 (default, на основе манифеста, детерминировано):
```yaml
# часть preprocessing_manifest.yaml
- name: master_clean_churned
    input: data/processed/cli_related/master_clean.parquet

    steps:
      - op: make_label
        customer_col: customer_id
        purchase_ts_col: order_purchase_timestamp
        target_col: churned
        horizon_days: 120
        reference_date: max             # или конкретная дата
        filter_status_col: order_status
        keep_statuses: [delivered]      # можно строкой "delivered, shipped"

    output: data/processed/cli_related/master_clean_churned.parquet
```
Вариант 2 (требует уже готового master_clean):
```bash
# ВАЖНО!: для датасета master_clean после всех join-ов. Для пояснения команд см. документацию preprocessing_cli.make_label
python src/olist_churn_prediction/preprocessing_cli.py label \
       --input-path data/interim/master_basic.parquet \
       --output-path data/processed/master_with_target.parquet \
       --customer-col customer_id \
       --purchase-ts-col order_purchase_timestamp \
       --target-col churned \
       --horizon-days 120 \
       --reference-date max \
       --filter-status-col order_status \
       --keep-statuses delivered
```
Вариант 3 (вручную в ноутбуке, пример в target_creation.ipynb):
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

### 7) Baseline‑моделирование и кросс‑валидация
Часть, ответственная за создание baseline-датасета
```yaml
# часть preprocessing_manifest.yaml
colsets:  # какие колонки включать в baseline
  baseline: &baseline_cols
    - customer_state
    - seller_state
    - order_products_value

datasets:
  - name: baseline_dataset
    input: data/processed/cli_related/master_clean_churned.parquet  # churned-датасет создается с помощью CLI
    steps:
      - op: select_columns
        include: *baseline_cols  # какие колонки включить
    output: data/processed/baseline_dataset.parquet  # полностью готовый к обучению датасет
```
Часть настроек для обучения модели
```yaml
# часть baseline.yaml
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
python -m olist_churn_prediction.baseline_cli cv --config configs/baseline.yaml
python -m olist_churn_prediction.baseline_cli train --config configs/baseline.yaml
```

### 8) Трекинг экспериментов (MLflow)
```bash
mlflow ui --backend-store-uri mlruns
# затем откройте http://127.0.0.1:5000
```
Советы: избегайте логирования больших артефактов; держите `mlruns/` вне гита.

### 9) Документация
```bash
# Linux/Mac
make -C docs html
# Windows (PowerShell)
cd docs; .\make.bat html
```
Собранные HTML‑страницы появятся в `docs/_build/html`.

---

## ⚙️ Makefile **(в разработке)**
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
- `pytest` для модульных тестов (`tests/`). **(в разработке)**
- `black`, `isort`, `flake8` и `pre-commit` для единого стиля.

---

## 🗺️ Roadmap
- [ ] Расширить пайплайн фичейнжиниринга (лаги, агрегации, OOF target encoding).
- [ ] Добавить Optuna‑тюнинг (XGBoost/LightGBM/CatBoost) с recall и macro‑F1.
- [ ] Экспорт артефактов в S3/MinIO, CI/CD (GitHub Actions), Docker‑образ.
- [ ] Улучшить Sphinx‑доки: API‑разделы, примеры CLI, диаграмма пайплайна.

---

## 📄 Лицензия
Проект распространяется по лицензии MIT (см. `LICENSE`).

---

## 🙏 Благодарности
- Шаблон: Cookiecutter Data Science.
- Сообщество Olist и открытые датасеты.