.. _getting-started:

Getting Started
===============

Краткое описание
-------
Этот проект решает задачу предсказания оттока клиентов по данным Olist. Ниже — шаги, чтобы развернуть окружение, подготовить данные и запустить базовый эксперимент (baseline).

Требования
----------
- Python 3.13+ (рекомендуется conda)
- Git
- (Опционально) Make (для сборки документации)

Установка
---------
.. code-block:: bash

   # 1) Клонирование
   git clone https://github.com/ViacheslavTimofeev/Olist_v5_churn_prediction.git
   cd <ваш-репозиторий>

   # 2) Окружение (через conda)
   conda create -n olist-ml python=3.13 -y
   conda activate olist-ml

   # 3) Установка пакета в editable-режиме
   pip install -e .

.. note::
   Если не используете conda — создайте venv стандартными средствами Python.

Подготовка данных
-----------------
Сырые данные ожидаются в каталоге ``data/raw``.
Минимум: таблицы заказов, клиентов, продавцов, платежей и т.д. (формат parquet/csv).

Ожидаемая структура (пример):

.. code-block:: text

   data/
     raw/
       geolocation_olist_public_dataset.csv
       olist_classified_public_dataset.csv
       olist_public-dataset_v2.csv
       ...
     interim/
     processed/

Быстрый старт (CLI)
-------------------
Все команды запускаются как **модули** из пакета, чтобы не путать пути.

1) Валидация входных датасетов:

.. code-block:: bash

   python -m olist_churn_prediction.validator_cli validate-all --manifest configs/validation_manifest.yaml

2) Предобработка (join/фичепроцессинг по YAML):

.. code-block:: bash

   python -m olist_churn_prediction.preprocessing_cli run --config configs/preprocessing.yaml

3) Базовый эксперимент (кросс-валидация, логирование в MLflow):

.. code-block:: bash

   python -m olist_churn_prediction.baseline_cli cv --config configs/baseline.yaml

.. note::
   Запуск через ``python -m ...`` решает типичные проблемы импорта, если файлы лежат в ``src/olist_churn_prediction``.
   Убедитесь, что в пакете есть ``__init__.py`` и проект установлен через ``pip install -e .``.

Пример конфигурации (фрагмент)
------------------------------
.. code-block:: yaml

   # configs/baseline.yaml
   data_path: "data/processed/baseline_dataset.parquet"
   target: "churned"
   id_cols: []
   numeric_features: []
   categorical_features: []

   model:
     name: "rf"               # 'logreg' | 'rf'
     params:
       n_estimators: 300
       max_depth: 12
       random_state: 42

   test_size: 0.2
   random_state: 42
   output_dir: "artifacts/baseline"
   cv:
     n_splits: 5

   mlflow:
     enabled: true
     tracking_uri: "file:./mlruns"
     experiment: "olist_baseline_cli"
     run_name: "baseline_rf"
     autolog: true
     log_model: true

.. warning::
   Параметры в ``params`` должны соответствовать выбранной модели (``model.name``).
   Нельзя оставлять параметры от ``logreg`` при ``name: rf`` — это вызовет ошибку.

Структура проекта
-----------------
.. code-block:: text

   .
   ├─ configs/
   │   ├─ baseline.yaml
   ├─ data/
   │   ├─ raw/
   │   ├─ interim/
   │   └─ processed/
   ├─ docs/
   │   ├─ conf.py
   │   ├─ index.rst
   │   └─ getting-started.rst   ← вы здесь
   ├─ src/
   │   └─ olist_churn_prediction/
   │       ├─ __init__.py
   │       ├─ validator_cli.py
   │       ├─ preprocessing_cli.py
   │       ├─ baseline_cli.py
   │       └─ feature_processing.py
   └─ validations/
       └─ manifest.yaml

Сборка документации
-------------------
.. code-block:: bash

   cd docs
   make html            # Терминал в JupyterLab/Jupyter Notebook из папки olist_churn/docs

Готовая документация появится в ``_build/html/index.html``.

Частые проблемы и решения
-------------------------
- **Sphinx: "Unexpected indentation" / "Block quote ends without a blank line"**  
  Убедитесь, что после директив (например, ``.. code-block::`` или ``.. toctree::``) есть **пустая строка**, а блоки содержимого правильно отступлены на 3–4 пробела.

- **Sphinx: "invalid option block" в toctree**  
  Правильный синтаксис:

  .. code-block:: rst

     .. toctree::
        :maxdepth: 2
        :caption: Содержание

        getting-started
        commands
        api/index

  Параметры (``:maxdepth:``, ``:caption:``) ставятся **сразу** под директивой, затем пустая строка и список файлов.  
  Обычно ``toctree`` располагают в ``index.rst``, а не внутри этой страницы.

- **"Failed to import olist_churn_prediction.baseline_cli"**  
  1) Установите пакет: ``pip install -e .``  
  2) Запускайте как модуль: ``python -m olist_churn_prediction.baseline_cli ...``  
  3) Проверьте, что ``docs/conf.py`` добавляет ``../src`` в ``sys.path`` (или что пакет установлен).

Дальше
------
- :doc:`commands` — справочник по CLI (все команды и опции)
- :doc:`api/index` — автодокументация модулей и функций

