"""Функции предобработки признаков для проекта Olist Churn.

Модуль содержит небольшие утилиты и операции подготовки данных.

Основные операции:
- :func:`lowercase_categoricals` — нормализация строковых/категориальных значений
(нижний регистр, пробелы → ``_``).
- :func:`disambiguate_city_state` — устранение неоднозначности городов путём
добавления суффикса штата/региона.
- :func:`group_by_features` — построчные агрегации по наборам колонок.
- Утилиты: :func:`rename_columns`, :func:`drop_columns`.

Приватные помощники (:func:`_to_list`, :func:`_infer_categoricals`) остаются
внутренними, но задокументированы для удобства сопровождения и IDE.
"""
from __future__ import annotations

from typing import Dict, List, Iterable, Callable
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_string_dtype


# ===================== Вспомогательные =====================

def _to_list(x) -> List[str]:
    """Приводит значение к списку строк.

    Args:
        x: Значение. Может быть ``None``, строкой, списком или кортежем.

    Returns:
        Список строк. Для ``None`` возвращает пустой список.
    
    Examples:
        >>> _to_list("a")
        ['a']
        >>> _to_list(["a", "b"]) # уже список
        ['a', 'b']
        >>> _to_list(None)
        []
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _infer_categoricals(df: pd.DataFrame) -> List[str]:
    """Определяет вероятные категориальные колонки в датафрейме.

    Колонка считается категориальной, если её dtype — ``category``,
    pandas ``StringDtype`` или обычный ``object`` (строки).
    
    Args:
        df: Исходный датафрейм.
    
    Returns:
        Список имён колонок с категориальным/строковым типом.
    """
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
    
    Для колонок типа ``category`` преобразуются **сами категории**, чтобы
    сохранить dtype. Для строковых колонок (``object``/``StringDtype``)
    значения нормализуются через векторные string-операции.
    
    Args:
        df: Входной датафрейм.
        cat_cols: Явный список колонок. Если ``None`` — берутся все строковые/
            категориальные столбцы автоматически.
        inplace: Если ``True`` — модифицировать исходный ``df``, иначе вернуть копию.
    
    Returns:
        Датафрейм с обновлёнными значениями категориальных колонок.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"city": ["New York", "Sao Paulo"], "x": [1, 2]})
        >>> lowercase_categoricals(df)["city"].tolist()
        ['new_york', 'sao_paulo']
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

    Требует, чтобы ``city_col`` и ``state_col`` уже были приведены к нижнему
    регистру (можно вызвать перед этим :func:`lowercase_categoricals`).
    
    Args:
        df: Входной датафрейм.
        city_col: Имя колонки с городами.
        state_col: Имя колонки с регионами/штатами.
        suffix_sep: Разделитель между названием города и регионом (по умолчанию ``"_"``).
        inplace: Если ``True``, модифицировать исходный датафрейм.
    
    Returns:
        Датафрейм с обновлённой колонкой ``city_col``.
    
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"city": ["paris", "paris"], "state": ["sp", "rj"]})
        >>> disambiguate_city_state(df, "city", "state")["city"].tolist()
        ['paris_sp', 'paris_rj']
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
    """Вычисляет новые признаки построчной агрегацией над группами колонок.
    
    Args:
        df: Входной датафрейм.
        group_mapping: Сопоставление "новый_признак" → список/колонка. Примеры::
        
            {"amount_total": ["price", "freight_value"]}
            {"items_total": "order_items"}
    
        agg_funcs: Правило агрегации. Допускаются формы:
            - строка (``"sum"``, ``"mean"``, ``"max"`` и др.) — одна функция для всех групп;
            - callable — одна функция для всех групп;
            - список функций той же длины, что и ``group_mapping``;
            - словарь ``{"новый_признак": функция|строка}``.
        keep_original: Если ``True``, оставить исходные колонки и добавить новые; если ``False``,
            удалить использованные столбцы и оставить только базовые + новые.
        prefix: Дополнительный префикс для имён новых колонок.
    
    Returns:
        Датафрейм с добавленными агрегированными признаками.
    
    Raises:
        ValueError: Если длина ``agg_funcs`` не совпадает с ``group_mapping`` или
            отсутствуют функции для некоторых групп.
        TypeError: Если ``agg_funcs`` неподдерживаемого типа.
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
    """Переименовывает колонки согласно отображению имён.

    Args:
        df: Исходный датафрейм.
        mapping: Словарь ``{"старое": "новое"}``.
        inplace: Если ``True``, модифицировать исходный датафрейм.
    
    Returns:
        Датафрейм с обновлёнными именами колонок.
    """
    target = df if inplace else df.copy()
    return target.rename(columns=mapping)

def drop_columns(df: pd.DataFrame, cols: Iterable[str], inplace: bool = False) -> pd.DataFrame:
    """Удаляет указанные колонки из датафрейма.

    Args:
        df: Исходный датафрейм.
        cols: Iterable имён колонок для удаления.
        inplace: Если ``True``, модифицировать исходный датафрейм.
    
    Returns:
        Датафрейм без выбранных колонок (или исходный, если колонки отсутствуют).
    """
    target = df if inplace else df.copy()
    return target.drop(columns=list(cols), errors="ignore")