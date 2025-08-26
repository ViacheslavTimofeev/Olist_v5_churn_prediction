from __future__ import annotations

from typing import Dict, List, Iterable, Callable
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_string_dtype


# ===================== Вспомогательные =====================

def _to_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _infer_categoricals(df: pd.DataFrame) -> List[str]:
    """Определяем строковые/категориальные столбцы."""
    cols: List[str] = []
    for c in df.columns:
        s = df[c]
        if is_categorical_dtype(s) or is_string_dtype(s) or s.dtype == "object":
            cols.append(c)
    return cols


# ===================== Операции предобработки =====================

def lowercase_categoricals(
    df: pd.DataFrame,
    cat_cols: Iterable[str] | None = None,
    inplace: bool = False
) -> pd.DataFrame:
    """Приводит строковые/категориальные колонки к нижнему регистру и заменяет пробелы на ``_``.

    Args:
        df (pd.DataFrame): Входной датафрейм.
        cat_cols (Iterable[str] | None): Явный список колонок.  
            Если ``None`` — берём все строковые/категориальные столбцы автоматически.
        inplace (bool): Если ``True`` — модифицируем исходный ``df``, иначе возвращаем копию.

    Returns:
        pd.DataFrame: Датафрейм с обновлёнными значениями категориальных колонок.
    """
    target = df if inplace else df.copy()
    cat_cols = list(cat_cols) if cat_cols is not None else _infer_categoricals(target)

    for col in cat_cols:
        if col not in target.columns:
            continue
        s = target[col]

        # category — преобразуем список категорий, чтобы dtype остался category
        if is_categorical_dtype(s):
            new_cats = (
                s.cat.categories.astype("string")
                 .str.lower()
                 .str.replace(r"\s+", "_", regex=True)
            )
            target[col] = s.cat.rename_categories(new_cats)
            continue

        # StringDtype или object — преобразуем значения
        if is_string_dtype(s) or s.dtype == "object":
            target[col] = (
                s.astype("string")
                 .str.lower()
                 .str.replace(r"\s+", "_", regex=True)
            )

    return target


def disambiguate_city_state(
    df: pd.DataFrame,
    city_col: str,
    state_col: str,
    *,
    suffix_sep: str = "_",
    inplace: bool = False
) -> pd.DataFrame:
    """Разрешает неоднозначность городов, добавляя к одноимённым городам суффикс штата/региона.

    Например:
        - ``paris`` из разных штатов → ``paris_sp``, ``paris_rj``.

    Требует, чтобы ``city_col`` и ``state_col`` уже были приведены к нижнему регистру
    (можно вызвать перед этим ``lowercase_categoricals``).

    Args:
        df (pd.DataFrame): Входной датафрейм.
        city_col (str): Имя колонки с городами.
        state_col (str): Имя колонки с регионами/штатами.
        suffix_sep (str): Разделитель между названием города и регионом (по умолчанию ``"_"``).

    Returns:
        pd.DataFrame: Датафрейм с обновлённой колонкой ``city_col``.
    """
    target = df if inplace else df.copy()

    if city_col not in target or state_col not in target:
        return target

    # Вычисляем, какие города встречаются в >1 штате
    combos = target.groupby([city_col, state_col]).size().reset_index(name="n")
    dup_cities = (
        combos.groupby(city_col)[state_col]
        .nunique()
        .reset_index(name="nu")
    )
    ambiguous = set(dup_cities.loc[dup_cities["nu"] > 1, city_col])

    if not ambiguous:
        return target

    # Для неоднозначных городов добавляем суффикс штата
    mask = target[city_col].isin(ambiguous)
    target.loc[mask, city_col] = (
        target.loc[mask, city_col].astype("string") + suffix_sep +
        target.loc[mask, state_col].astype("string")
    )

    return target


def group_by_features(
    df: pd.DataFrame,
    group_mapping: Dict[str, List[str] | str],
    *,
    agg_funcs: str | Callable | List[str | Callable] | Dict[str, str | Callable] = "sum",
    keep_original: bool = False,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Построчные агрегации признаков (по колонкам), с гибкими функциями.

    Args:
        df (pd.DataFrame): Входной датафрейм.
        group_mapping (dict):  
            - ``{"new_feature": ["col1", "col2", ...], ...}``  
            - или ``{"new_feature": "col"}``.
        agg_funcs (str | callable | list | dict, optional):  
            - строка: одна agg-функция для всех групп (``"sum"``, ``"mean"``, ``"max"`` и др.),  
            - callable: одна функция для всех групп,  
            - список: список функций по порядку для каждой группы,  
            - dict: ``{"new_feature": "sum" | callable, ...}``.
        keep_original (bool): Если ``True`` — оставить исходные колонки.
        prefix (str | None): Префикс для новых колонок.

    Returns:
        pd.DataFrame: Датафрейм с добавленными агрегированными признаками.
    """
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

        # aggregate по строкам
        new_cols[new_name] = df_out[cols_list].aggregate(agg_fn, axis=1)

    if keep_original:
        for k, v in new_cols.items():
            df_out[k] = v
        return df_out

    # оставляем только новые + колонки, не участвовавшие ни в одной группе
    used = set(sum((_to_list(v) for v in group_mapping.values()), []))
    base = df_out.drop(columns=[c for c in used if c in df_out.columns], errors="ignore")
    for k, v in new_cols.items():
        base[k] = v
    return base


# ===================== Утилитарные шаги (по желанию) =====================

def rename_columns(df: pd.DataFrame, mapping: Dict[str, str], inplace: bool = False) -> pd.DataFrame:
    """Переименование колонок."""
    target = df if inplace else df.copy()
    return target.rename(columns=mapping)

def drop_columns(df: pd.DataFrame, cols: Iterable[str], inplace: bool = False) -> pd.DataFrame:
    """Удаление колонок (ignore missing)."""
    target = df if inplace else df.copy()
    return target.drop(columns=list(cols), errors="ignore")


# ---------------------- Планы на будущее ----------------------
# 1. Конфигурация через YAML/JSON.
# 2. Поддержка категориальных агрегаций (mode, first, any).
# 3. Версия на polars/dask для больших данных.
# 4. Добавить integration tests в /tests.
