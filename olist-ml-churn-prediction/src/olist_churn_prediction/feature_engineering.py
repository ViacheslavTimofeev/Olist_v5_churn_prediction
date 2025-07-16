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

RAW_DIR = Path("../../data/raw")
PROCESSED_DIR = Path("../../data/processed")
TARGET_COL = "order_status"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _split_schema_fields(schema: Type[BaseModel]):
    """Возвращает (date_cols, dtype_map) из Pydantic-схемы."""
    date_cols: list[str] = []
    dtype_map: dict[str, str] = {}

    for name, field in schema.model_fields.items():            # Pydantic v2
        ann = field.annotation
        # Проверяем на datetime | Optional[datetime]
        if ann is datetime or (get_origin(ann) is Union and datetime in get_args(ann)):
            date_cols.append(name)
        else:
            # Подбираем минимальный совместимый dtype
            if ann is str:
                dtype_map[name] = "string"
            elif ann in (float, Optional[float]):              # float64 по умолчанию
                dtype_map[name] = "float32"
            elif ann in (int, Optional[int]):
                dtype_map[name] = "int64"
            # остальные типы (bool, category…) оставляем pandas'у

    return date_cols, dtype_map


def load_data(path: str | Path,
              schema: Type[BaseModel] | None = None,
              validate: bool = False) -> pd.DataFrame:
    """
    Универсальный загрузчик CSV / Parquet.

    Parameters
    ----------
    path : str | Path
        Путь к файлу данных (.csv или .parquet).
    schema : BaseModel subclass | None
        Pydantic-модель для определения типов
        (и, опционально, валидации строк).
    validate : bool, default False
        Проверять ли каждую строку через schema. Может быть медленно
        на больших таблицах, используйте для отладки.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if schema:
        date_cols, dtype_map = _split_schema_fields(schema)
    else:
        date_cols, dtype_map = [], {}

    # --- чтение файла ---
    if path.suffix == ".csv":
        df = pd.read_csv(
            path,
            parse_dates=date_cols,      # авто-конверт дат
            dtype=dtype_map,            # явные типы
            low_memory=False
        )
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        # На всякий случай убеждаемся, что date-колонки в datetime64
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")

    # --- валидация (необяз.) ---
    if validate and schema:
        bad_rows: list[int] = []
        for idx, record in df.iterrows():
            try:
                schema.model_validate(record.to_dict())
            except ValidationError:
                bad_rows.append(idx)

        if bad_rows:
            raise ValueError(f"Schema validation failed for rows: {bad_rows[:10]} "
                             f"(total={len(bad_rows)})")

    return df


class DateDiffTransformer(BaseEstimator, TransformerMixin):
    """Вычисляет разницу (дни) между двумя datetime-колонками."""
    def __init__(self, start_col: str, end_col: str, new_col: str):
        self.start_col = start_col
        self.end_col = end_col
        self.new_col = new_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.new_col] = (X[self.end_col] - X[self.start_col]).dt.days
        return X[[self.new_col]]


def build_feature_matrix(df: pd.DataFrame, *, fit: bool = True):
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=RAW_DIR / "dataset.parquet")
    p.add_argument("--output", type=Path,
                   default=PROCESSED_DIR / "features.joblib")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df_raw = load_data(args.input)
    X, y, pipe = build_feature_matrix(df_raw)
    joblib.dump({"X": X, "y": y, "pipe": pipe}, args.output)
    logging.info("Features saved to %s", args.output)
