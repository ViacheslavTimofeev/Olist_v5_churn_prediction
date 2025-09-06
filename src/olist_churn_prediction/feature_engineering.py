"""Инженерия признаков: сбор матрицы X и целевой переменной y.

Модуль предоставляет простой пайплайн признаков для Olist Churn и CLI-заготовку
для сохранения результатов на диск. Докстринги — в стиле Google
(совместим с Sphinx `napoleon`).

Состав:

- :class:`DateDiffTransformer` — трансформер sklearn для расчёта разницы дат
  в часах между двумя колонками.
- :func:`build_feature_matrix` — строит матрицу признаков ``X``, цель ``y`` и
  возвращает собранный препроцессор/пайплайн.
- :func:`parse_args` — парсит аргументы CLI.

Пример (из кода)::

    df = pd.read_parquet("data/raw/dataset.parquet")
    X, y, pipe = build_feature_matrix(df)

Пример запуска CLI::

    python feature_engineering.py \
        --input data/raw/dataset.parquet \
        --output data/processed/features.joblib
"""

from __future__ import annotations
from datetime import datetime
from typing import Type, get_origin, get_args, Union, Optional
from pydantic import BaseModel, ValidationError
import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from olist_churn_prediction.paths import RAW_DIR, PROCESSED_DIR

TARGET_COL = "order_status" # целевая переменная для примера базового пайплайна

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class DateDiffTransformer(BaseEstimator, TransformerMixin):
    """Считает разницу между двумя datetime-колонками и возвращает одну фичу.
    
    Разница вычисляется в **часах** (целое число) как ``end - start``.
    
    Args:
        start_col: Имя колонки с начальной датой/временем.
        end_col: Имя колонки с конечной датой/временем.
        new_col: Имя создаваемой фичи с разницей во времени (в часах).
    
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ... 't0': pd.to_datetime(['2020-01-01 00:00', '2020-01-02 12:00']),
        ... 't1': pd.to_datetime(['2020-01-01 06:00', '2020-01-03 12:00'])
        ... })
        >>> DateDiffTransformer('t0', 't1', 'delta_h').fit_transform(df).ravel().tolist()
        [6, 48]
    """
    def __init__(self, start_col: str, end_col: str, new_col: str) -> None:
        self.start_col = start_col
        self.end_col = end_col
        self.new_col = new_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Ничего не обучает, совместимость со sklearn API.

        Args:
            X: Входной датафрейм.
            y: Не используется.
    
        Returns:
            ``self``.
        """
        return self

    def transform(self, X) -> pd.DataFrame:
        """Добавляет колонку с разницей времени и возвращает только её.

        Ожидается, что колонки ``start_col`` и ``end_col`` имеют тип DateDiff
        ``datetime64[ns]`` (или будут корректно приведены ранее).
        
        Args:
            X: Входной датафрейм с двумя datetime-колонками.
        
        Returns:
            Датафрейм из одной колонки ``new_col`` (целые часы).
        """
        X = X.copy()
        X[self.new_col] = ((X[self.end_col] - X[self.start_col]).dt.total_seconds() // 3600).astype(int)
        return X[[self.new_col]]


def build_feature_matrix(df: pd.DataFrame, *, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
    """Строит матрицу признаков ``X``, целевую ``y`` и препроцессор.

    Состав признаков (примерный baseline):
      - Числовые: ``freight_value``, ``payment_value`` → ``StandardScaler``.
      - Категориальные: ``payment_type``, ``customer_state`` → ``OneHotEncoder``.
      - Датовые: разница часов между ``order_purchase_timestamp`` и
        ``order_delivered_customer_date`` через :class:`DateDiffTransformer`.
    
    Args:
        df: Исходный датафрейм с необходимыми колонками.
        fit: Если ``True`` — вызывает ``fit_transform``; иначе ``transform``.
    
    Returns:
        Кортеж ``(X, y, pipe)``:
            - ``X``: матрица признаков (может быть ``np.ndarray``/sparse matrix в зависимости от OHE).
            - ``y``: целевая переменная (серия с ``TARGET_COL``).
            - ``pipe``: собранный ``Pipeline`` с ``ColumnTransformer``.
    
    Raises:
        KeyError: Если отсутствуют необходимые колонки во входном датафрейме.
    """
    num_cols = ["freight_value", "payment_value"]
    cat_cols = ["payment_type", "customer_state"]

    date_diff = DateDiffTransformer("order_purchase_timestamp",
                                    "order_delivered_customer_date",
                                    "delivery_days")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("date", date_diff, ["order_purchase_timestamp",
                                 "order_delivered_customer_date"])
        ]
    )

    pipe = Pipeline(steps=[("preprocessor", preprocessor)])

    if fit:
        X = pipe.fit_transform(df)
    else:
        X = pipe.transform(df)

    y = df[TARGET_COL].copy()
    return X, y, pipe


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки для скрипта.

    Options:
        --input: Путь к входному датасету (``.parquet``), по умолчанию ``RAW_DIR/dataset.parquet``.
        --output: Куда сохранить результат (``.joblib``), по умолчанию ``PROCESSED_DIR/features.joblib``.
    
    Returns:
        Пространство имён с аргументами (``argparse.Namespace``).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=RAW_DIR / "dataset.parquet")
    p.add_argument("--output", type=Path,
                   default=PROCESSED_DIR / "features.joblib")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Примечание: функция `load_data` упоминается как внешняя утилита чтения датасета.
    # Здесь предполагается, что она определена в другом модуле проекта и доступна по импортам.
    df_raw = load_data(args.input)
    X, y, pipe = build_feature_matrix(df_raw)
    joblib.dump({"X": X, "y": y, "pipe": pipe}, args.output)
    logging.info("Features saved to %s", args.output)
