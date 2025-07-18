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
