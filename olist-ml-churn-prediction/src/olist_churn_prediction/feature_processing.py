from __future__ import annotations
from datetime import datetime
from typing import Type, get_origin, get_args, Union, Optional, Dict, List, Callable, Iterable
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

import itertools

from pandas.api.types import (
    is_string_dtype,
    is_categorical_dtype,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _split_schema_fields(schema):
    """Возвращает (date_cols, dtype_map) из Pydantic-схемы."""
    date_cols, dtypes = [], {}

    for name, field in schema.model_fields.items():      # Pydantic v2 API
        anno = field.annotation                          # исходная аннотация
        # 1️⃣  «разворачиваем» Optional[T]  →  T
        if get_origin(anno) is Optional:
            anno = get_args(anno)[0]
        # 2️⃣  теперь проверяем базовый тип
        if anno is datetime or (get_origin(anno) is Union and datetime in get_args(anno)):
            date_cols.append(name)
        elif anno is str:
            dtypes[name] = "string"      # всегда строковый dtype, даже с <NA>
        elif anno is float:
            dtypes[name] = "float32"
        elif anno is int:
            dtypes[name] = "Int64"       # nullable целые

    return date_cols, dtypes


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


def lowercase_categoricals(
    df: pd.DataFrame,
    cat_cols: list[str],
    inplace: bool = False
) -> pd.DataFrame:
    """
    Приводит строковые (категориальные) значения к нижнему регистру.

    Параметры
    ----------
    df : pd.DataFrame
        Входной датасет.
    cat_cols : list[str] | None
        Список столбцов-категорий.  
        • Если None — автоматически берём все столбцы dtype == 'object'  
          или категориальные ('category').
    inplace : bool
        • True  — преобразует переданный df на месте и возвращает его же.  
        • False — создаёт копию и возвращает её (оригинал не меняется).

    Возвращает
    ----------
    pd.DataFrame
        Датасет с приведёнными к lower-case категориальными столбцами.

    Пример
    -------
    >>> df = pd.DataFrame({'city': ['Moscow', 'LONDON', 'PaRiS'], 'value': [1, 2, 3]})
    >>> lowercase_categoricals(df)
        city  value
    0  moscow      1
    1  london      2
    2   paris      3
    """
    
    target = df if inplace else df.copy()

    for col in cat_cols:
        s = target[col]

        # --- Категориальные колонки ---
        if is_categorical_dtype(s):
            # преобразуем именно категории,
            # чтобы сам столбец остался category
            new_cats = (
                s.cat.categories
                 .astype("string")                 # <-- уже StringDtype, не object
                 .str.lower()
                 .str.replace(r"\s+", "_", regex=True)
            )
            target[col] = s.cat.rename_categories(new_cats)
            continue

        # --- Строковые расширенные типы (StringDtype) ---
        if is_string_dtype(s):
            target[col] = (
                s.str.lower()
                 .str.replace(r"\s+", "_", regex=True)
            )
            # dtype остаётся string
            continue

        # --- Обычный object (python-строки) ---
        target[col] = (
            s.astype("string")                    # << вместо str → StringDtype
             .str.lower()
             .str.replace(r"\s+", "_", regex=True)
             # при желании вернём object:
             # .astype("object")
        )

    return target


def disambiguate_city_state(
    df: pd.DataFrame,
    city_col: str,
    state_col: str,
    *,
    suffix_sep: str = "_",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Устраняет неоднозначность «город-штат»: если один и тот же `city`
    встречается в нескольких штатах, добавляет цифровой суффикс
    (`_1`, `_2`, …) начиная со второго штата.

    Алгоритм
    --------
    1. Находим города, у которых `nunique(state) > 1`.
    2. Для каждого такого города сортируем список штатов (чтобы
       результат был воспроизводим).
    3. • Для первого штата оставляем имя без изменений.  
       • Для второго и последующих добавляем \"_<idx>\".

    Параметры
    ---------
    df : pd.DataFrame
        Таблица, содержащая столбцы с городом и штатом.
    city_col : str
        Название столбца с городами (нормализованными: lower + '_' вместо пробелов).
    state_col : str
        Название столбца со штатами/регионами.
    suffix_sep : str
        Разделитель перед номером («_» по умолчанию → `alvorada_1`).
    inplace : bool
        • True  — меняет `df` на месте и возвращает его.  
        • False — работает с копией и возвращает новую таблицу.

    Возвращает
    ----------
    pd.DataFrame
        Датафрейм, где неоднозначные города переименованы.

    Пример
    -------
    >>> data = {
    ...     "city":  ["alvorada", "alvorada", "alvorada", "porto_alegre"],
    ...     "state": ["RS",       "TO",        "BA",       "RS"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> disambiguate_city_state(df)
              city state
    0     alvorada    RS
    1  alvorada_1    TO
    2  alvorada_2    BA
    3  porto_alegre    RS
    """
    target = df if inplace else df.copy()

    # 1. Города, встретившиеся более чем в одном штате
    dup_cities = (
        target.groupby(city_col)[state_col]
        .nunique()
        .loc[lambda s: s > 1]
        .index
    )

    # 2. Обрабатываем каждый «двойник»
    for city in dup_cities:
        # список штатов отсортирован
        states = sorted(target.loc[target[city_col] == city, state_col].unique())
        for idx, st in enumerate(states):
            # суффикс только с 1-го дубликата
            suffix = "" if idx == 0 else f"{suffix_sep}{idx}"
            new_name = f"{city}{suffix}"
            mask = (target[city_col] == city) & (target[state_col] == st)
            target.loc[mask, city_col] = new_name

    return target


NumericAgg = Union[str, Callable[[pd.Series], float]]

def _to_list(obj: Union[str, List[str]]) -> List[str]:
    return [obj] if isinstance(obj, str) else list(obj)


def group_by_features(
    df: pd.DataFrame,
    group_mapping: Dict[str, Union[str, List[str]]],
    agg_funcs: Union[Dict[str, NumericAgg], List[NumericAgg], NumericAgg] = "sum",
    keep_original: bool = False,
    prefix: str | None = None,
) -> pd.DataFrame:
    '''Автоматизирует создание новых признаков путём агрегирования (группировки)
    заданного набора колонок.

    Parameters
    ----------
    df : pd.DataFrame
        Входной датафрейм.
    group_mapping : dict
        Отображение {'новый_признак': ['col1','col2', ...]}.
    agg_funcs : dict | list | str | callable, default "sum"
        Функции агрегирования:
            * одна строка/функция — применяется ко всем группам;
            * список — позиционно для каждого ключа group_mapping;
            * словарь — {'новый_признак': 'mean'}.
    keep_original : bool, default False
        Если True сохранит исходные колонки.
    prefix : str | None, optional
        Префикс для итоговых колонок, удобно для стэкинга в конвейере.
    Returns
    -------
    pd.DataFrame
        Датафрейм со сгенерированными колонками.
    '''
    df_out = df.copy()

    # Нормализуем agg_funcs к словарю
    if isinstance(agg_funcs, (str, Callable)):
        agg_funcs = {k: agg_funcs for k in group_mapping}
    elif isinstance(agg_funcs, list):
        if len(agg_funcs) != len(group_mapping):
            raise ValueError("Длина agg_funcs не совпадает с group_mapping")
        agg_funcs = dict(zip(group_mapping, agg_funcs))
    elif isinstance(agg_funcs, dict):
        missing = set(group_mapping) - set(agg_funcs)
        if missing:
            raise ValueError(f"Для групп {missing} не указаны функции агрегации")
    else:
        raise TypeError("agg_funcs должен быть str, callable, list или dict")

    new_cols = {}
    for new_feat, cols in group_mapping.items():
        cols_list = _to_list(cols)
        agg_fn = agg_funcs[new_feat]
        new_name = f"{prefix}_{new_feat}" if prefix else new_feat
        new_cols[new_name] = df_out[cols_list].aggregate(agg_fn, axis=1)

    # Удаляем оригинальные колонки, если нужно
    if not keep_original:
        cols_to_drop: Iterable[str] = set(
            itertools.chain.from_iterable(_to_list(v) for v in group_mapping.values())
        )
        df_out = df_out.drop(columns=list(cols_to_drop))

    # Добавляем новые
    df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df_out


class FeatureGrouper:
    '''
    Обёртка, совместимая со Scikit-Learn API (fit/transform).
    Позволяет легко подключать группировку в Pipeline/ColumnTransformer.

    Пример
    ------
    >>> grouper = FeatureGrouper(mapping, agg_funcs='sum')
    >>> X_new = grouper.fit_transform(X)
    '''

    def __init__(
        self,
        group_mapping: Dict[str, Union[str, List[str]]],
        agg_funcs: Union[Dict[str, NumericAgg], List[NumericAgg], NumericAgg] = "sum",
        keep_original: bool = False,
        prefix: str | None = None,
    ):
        self.group_mapping = group_mapping
        self.agg_funcs = agg_funcs
        self.keep_original = keep_original
        self.prefix = prefix

    def fit(self, X: pd.DataFrame, y=None):
        # Ничего обучать не нужно
        return self

    def transform(self, X: pd.DataFrame):
        return group_by_features(
            X,
            self.group_mapping,
            agg_funcs=self.agg_funcs,
            keep_original=self.keep_original,
            prefix=self.prefix,
        )

    def get_feature_names_out(self, input_features=None):
        prefix = f"{self.prefix}_" if self.prefix else ""
        return [f"{prefix}{name}" for name in self.group_mapping]


# ---------------------- Планы на будущее ----------------------
# 1. Конфигурация через YAML/JSON.
# 2. Поддержка категориальных агрегаций (mode, first, any).
# 3. Версия на polars/dask для больших данных.
# 4. Добавить integration tests в /tests.
