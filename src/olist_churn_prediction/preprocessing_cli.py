"""Манифест‑driven предобработка данных для проекта Olist Churn.

Модуль содержит CLI-команды (Typer) и ядро предобработки, которое
применяет последовательность шагов к входному ``DataFrame``.

Основные элементы:

- :func:`_load_df` — загрузка датасета из файла (glob) или SQL.
- :func:`_save_df` — сохранение датасета в ``.parquet``/``.csv``.
- :func:`_apply_steps` — поочерёдное применение шагов из манифеста.
- CLI-команды: :func:`apply`, :func:`run`, :func:`make_label`.

Пример запуска (из консоли)::

    # единичный датасет без манифеста
    python -m olist_churn_prediction.preprocessing_cli apply data/raw/orders.parquet data/interim/orders_clean.parquet --steps-json '[{"op":"lowercase_categoricals"}]'

    # пакетный режим с манифестом
    python -m olist_churn_prediction.preprocessing_cli run --manifest preprocessings/preprocessing_manifest.yaml
"""

from __future__ import annotations

import json
import glob
import os
import yaml
import typer
from typer.main import get_command
from pathlib import Path
from typing import Dict, Any, Union, List, Callable
import pandas as pd
from olist_churn_prediction.targets import create_churn_label
from olist_churn_prediction import feature_processing as fp  # lowercase_categoricals, disambiguate_city_state, group_by_features

app = typer.Typer()


def _load_df(entry: Dict[str, Any]) -> pd.DataFrame:
    """Загружает ``DataFrame`` по описанию источника.
    
    Поддерживаются два режима:
    - ``"file"`` — читает **последний** (по времени модификации) файл по
    glob-маске из ``entry["input"]``; формат определяется по расширению.
    - ``"sql"`` — выполняет SQL-запрос ``entry["query"]`` с использованием
    строки подключения из переменной окружения ``entry["conn_env"]``.
    
    Args:
        entry: Описание источника данных. Ключи (в зависимости от режима):
            * ``reader``: ``"file"`` | ``"sql"`` (по умолчанию ``"file"``).
            * для ``file``: ``input`` — glob-маска пути к файлам.
            * для ``sql``: ``query`` (SQL-строка), ``conn_env`` (имя env-переменной
            со строкой подключения), опционально ``params``.
    
    Returns:
        Загруженный датафрейм.
    
    Raises:
        KeyError: Если отсутствуют обязательные ключи для выбранного режима.
        FileNotFoundError: Если по glob-маске не найден ни один файл.
        ValueError: Если указан неподдерживаемый ``reader``.
    """
    reader = entry.get("reader", "file")
    if reader == "sql":
        import sqlalchemy as sa
        url = os.environ[entry["conn_env"]]
        engine = sa.create_engine(url)
        return pd.read_sql(sa.text(entry["query"]), engine, params=entry.get("params"))
    else:
        path = sorted(glob.glob(entry["input"]))[-1]
        return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _save_df(df: pd.DataFrame, output: str) -> None:
    """Сохраняет ``DataFrame`` на диск в ``.parquet`` или ``.csv``.

    Папки создаются автоматически.
    
    Args:
        df: Датафрейм для сохранения.
        output: Путь к файлу результата (окончание ``.parquet`` или ``.csv``).
    
    Returns:
        None
    """
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    typer.echo(f"💾 Saved: {out}")

    
def _apply_steps(
    df: pd.DataFrame,
    steps: List[Dict[str, Any]],
    defaults: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Применяет последовательность шагов предобработки к ``DataFrame``.
    
    Поддерживаемые операции (``op``):
    ``lowercase_categoricals``, ``disambiguate_city_state``,
    ``group_by_features``, ``groupby_aggregate``, ``dropna_rows``,
    ``dropna_columns``, ``drop_duplicates``, ``rename_columns``,
    ``drop_columns``, ``select_columns``, ``join``.
    
    Args:
        df: Исходный датафрейм.
        steps: Последовательность шагов, где каждый элемент — словарь
            с ключом ``op`` и параметрами операции.
        defaults: Глобальные значения по умолчанию из манифеста (необязательны).
    
    Returns:
        Обновлённый датафрейм после всех применённых операций.
    
    Raises:
        ValueError: Если шаг не содержит ключ ``op`` или указан неизвестный ``op``.
    """
    defaults = defaults or {}
    X = df

    for i, step in enumerate(steps, start=1):
        op = step.get("op")
        if not op:
            raise ValueError(f"Step #{i} has no 'op'")

        if op == "lowercase_categoricals":
            cat_cols = step.get("cat_cols")
            if cat_cols is None:
                # если не передали, используем все object/string/category
                cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object", "string") or pd.api.types.is_categorical_dtype(X[c])]
            X = fp.lowercase_categoricals(X, cat_cols=cat_cols, inplace=False)

        elif op == "disambiguate_city_state":
            city_col = step["city_col"]
            state_col = step["state_col"]
            suffix_sep = step.get("suffix_sep", "_")
            X = fp.disambiguate_city_state(X, city_col, state_col, suffix_sep=suffix_sep, inplace=False)

        elif op == "group_by_features":
            group_mapping = step["group_mapping"]             # {"new_feat": ["col1","col2"], ...}
            agg_funcs     = step.get("agg_funcs", "sum")      # может быть str|callable|list|dict
            keep_original = step.get("keep_original", False)
            prefix        = step.get("prefix")
            X = fp.group_by_features(
                X, group_mapping=group_mapping, agg_funcs=agg_funcs,
                keep_original=keep_original, prefix=prefix
            )

        elif op == "groupby_aggregate":
            # Группировка по ключу(ам) с разными аггрегаторами по колонкам.
            # Параметры:
            #   by: str | list[str]        — ключ(и) группировки
            #   sum_cols:  list[str]
            #   mean_cols: list[str]
            #   min_cols:  list[str]
            #   first_for_rest: bool=True  — для всех остальных колонок берём 'first'
            by = step["by"]
            if isinstance(by, str):
                by = [by]

            sum_cols  = step.get("sum_cols", []) or []
            mean_cols = step.get("mean_cols", []) or []
            min_cols  = step.get("min_cols", []) or []
            first_for_rest = bool(step.get("first_for_rest", True))

            # 1) строим словарь аггрегаций
            agg_dict = {}
            for c in sum_cols:  agg_dict[c]  = "sum"
            for c in mean_cols: agg_dict[c] = "mean"
            for c in min_cols:  agg_dict[c]  = "min"

            # 2) остальные колонки — 'first' (кроме ключей и уже перечисленных)
            if first_for_rest:
                selected = set(by) | set(sum_cols) | set(mean_cols) | set(min_cols)
                for c in X.columns:
                    if c not in selected:
                        agg_dict[c] = "first"

            # 3) сам groupby
            # dropna=False чтобы не терять группы с NaN-ключом.
            X = (
                X.groupby(by, dropna=False)
                 .agg(agg_dict)
                 .reset_index()
            )
            
        elif op == "dropna_rows":
        # Удаление строк с пропусками.
        # Параметры:
        #   subset: str | list[str] — по каким колонкам проверять NaN (если не задано — по всем)
        #   how: "any"|"all"        — удалить строку, если есть любой NaN ("any") или все NaN ("all")
        #   thresh: int             — минимальное число НЕ-пустых значений, чтобы строка осталась (если задано, 'how' не используется)
            subset = step.get("subset")
            if isinstance(subset, str):
                subset = [subset]

            thresh = step.get("thresh", None)
            before = len(X)

            if thresh is not None:
                # применяем по правилу "оставить строки с >= thresh непустыми значениями"
                X = X.dropna(axis=0, subset=subset, thresh=int(thresh))
            else:
                how = step.get("how", "any")
                X = X.dropna(axis=0, subset=subset, how=how)

            removed = before - len(X)
            if removed:
                typer.echo(f"   • dropna_rows: removed {removed} rows")

        elif op == "dropna_columns":
        # Удаление столбцов.
        # Режимы:
        #   A) только cols -> удалить их безусловно
        #   B) только min_missing_ratio -> удалить все колонки с NaN-дельтой >= порога
        #   C) cols + min_missing_ratio -> проверить ТОЛЬКО cols и удалить те, где доля NaN >= порога
            cols = step.get("cols")
            if isinstance(cols, str):
                cols = [cols]
            min_ratio = step.get("min_missing_ratio", None)

            to_drop = set()
            if min_ratio is None:
                # A) безусловный дроп перечисленных колонок
                if cols:
                    to_drop.update(cols)
            else:
                # B/C) пороговая логика
                if cols:
                    candidates = [c for c in cols if c in X.columns]
                else:
                    candidates = list(X.columns)
                if candidates:
                    ratios = X[candidates].isna().mean()
                    auto_drop = ratios[ratios >= float(min_ratio)].index.tolist()
                    to_drop.update(auto_drop)

            if to_drop:
                existing = [c for c in to_drop if c in X.columns]
                if existing:
                    X = X.drop(columns=existing, errors="ignore")
                    typer.echo(f"   • dropna_columns: dropped {len(existing)} columns: {existing[:10]}{'...' if len(existing)>10 else ''}")


        elif op == "drop_duplicates":
            # Удаление дубликатов.
            # Параметры (все опциональны):
            #   subset: "all" | str | list[str] — по каким колонкам искать дубли ("all" = по всем)
            #   keep: "first"|"last"|False      — какое вхождение оставить (дефолт 'first'; False = удалить все повторы)
            #   ignore_index: bool              — пересоздать индексы (дефолт True)
            subset = step.get("subset")
            if subset == "all":
                subset = None  # pandas: None => все колонки
            elif isinstance(subset, str):
                subset = [subset]

            keep = step.get("keep", "first")
            ignore_index = bool(step.get("ignore_index", True))

            before = len(X)
            X = X.drop_duplicates(subset=subset, keep=keep, ignore_index=ignore_index)
            removed = before - len(X)
            if removed:
                typer.echo(f"   • drop_duplicates: removed {removed} rows")

        elif op == "rename_columns":
            # утилитарный шаг: {"old":"new", ...}
            mapping = step["mapping"]
            X = X.rename(columns=mapping)

        elif op == "drop_columns":
            cols = step["cols"]
            X = X.drop(columns=cols, errors="ignore")
            
        elif op == "select_columns":
            include = step.get("include")
            exclude = step.get("exclude")

            if include and exclude:
                raise typer.BadParameter("select_columns: use either 'include' or 'exclude', not both")

            if include:
                if isinstance(include, str):
                    include = [include]
                keep = [c for c in include if c in X.columns]
                missing = sorted(set(include) - set(keep))
                if missing:
                    typer.echo(f"   • select_columns: missing {missing[:10]}{'...' if len(missing)>10 else ''}")
                X = X[keep]
            elif exclude:
                if isinstance(exclude, str):
                    exclude = [exclude]
                X = X[[c for c in X.columns if c not in set(exclude)]]
            else:
                typer.echo("   • select_columns: nothing to do (no include/exclude)")

        elif op == "join":
            # Параметры:
            #  right: str (путь к файлу .csv/.parquet)
            #  on: str|list[str]  (или left_on/right_on)
            #  how: left|inner|outer (по умолчанию left)
            #  select: list[str]  — колонки правой таблицы, которые оставить (плюс ключи)
            #  suffix_right: str  — суффикс для коллизий имён
            right = step["right"]
            how = step.get("how", "left")
            on = step.get("on")
            left_on = step.get("left_on")
            right_on = step.get("right_on")
            select = step.get("select")
            suffix_right = step.get("suffix_right", "_r")

            rdf = _load_df({"input": right})
            if select:
                keys = on if isinstance(on, list) else ([on] if on else [])
                if left_on and right_on:
                    keys = [right_on] if isinstance(right_on, str) else list(right_on)
                keep = list(dict.fromkeys(list(keys) + list(select)))
                rdf = rdf[[c for c in keep if c in rdf.columns]]

            prev_cols = set(X.columns)
            if on:
                X = X.merge(rdf, how=how, on=on, suffixes=(None, suffix_right))
            else:
                X = X.merge(rdf, how=how, left_on=left_on, right_on=right_on, suffixes=(None, suffix_right))
            typer.echo(f"   • join {Path(right).name}: +{len([c for c in X.columns if c not in prev_cols])} cols")

        else:
            raise ValueError(f"Unknown op '{op}' in step #{i}")
            

    return X


@app.command()
def apply(
    input: str = typer.Argument(..., help="Входной файл (.csv/.parquet)"),
    output: str = typer.Argument(..., help="Куда сохранить результат"),
    steps_json: str = typer.Option(None, help="JSON со списком шагов"),
    sample: float = typer.Option(None, help="Доля сэмпла для отладки, напр. 0.1"),
) -> None:
    """Применяет шаги предобработки к одному датасету (без манифеста).

    Пример:

    preproc apply data/raw.csv data/interim/clean.parquet \
      --steps-json '[{"op":"lowercase_categoricals", "cat_cols":["customer_city"]}]'
    
    Args:
        input: Входной файл (``.csv`` или ``.parquet``).
        output: Куда сохранить результат.
        steps_json: JSON-строка со списком шагов предобработки.
        sample: Доля сэмпла для отладки (например, ``0.1``).
    
    Returns:
        None
    """
    df = pd.read_parquet(input) if input.endswith(".parquet") else pd.read_csv(input)
    if sample:
        df = df.sample(frac=float(sample), random_state=42)
    steps = json.loads(steps_json) if steps_json else []
    df_out = _apply_steps(df, steps)
    _save_df(df_out, output)


@app.command()
def run(manifest: str = typer.Option(
    "preprocessings/preprocessing_manifest.yaml",
    "--manifest", "-m",
    help="Путь к YAML-манифесту предобработки"
    )
) -> None:
    """Выполняет пакетную предобработку согласно YAML-манифесту.

    Читает секции ``defaults`` и ``datasets``. Для каждого датасета:
      1) загружает источник (:func:`_load_df`),
      2) при необходимости сэмплирует,
      3) применяет :func:`_apply_steps`,
      4) сохраняет результат.
    
    Args:
        manifest: Путь к YAML-манифесту предобработки.
    
    Raises:
        typer.Exit: Если хотя бы для одного датасета произошла ошибка загрузки,
            обработки или сохранения.
    
    Returns:
        None
    """
    cfg = yaml.safe_load(Path(manifest).read_text())
    defaults = cfg.get("defaults", {})

    errors_total: list[str] = []

    for ds in cfg["datasets"]:
        if ds.get("enabled") is False:
            continue

        name = ds["name"]
        typer.echo(f"▶ {name}")

        # 1) загрузка
        try:
            df = _load_df(ds)   # та же идея, что и в validator_cli
        except Exception as e:
            msg = f"load error: {e}"
            typer.echo(f"❌ {name}: {msg}")
            errors_total.append(f"{name}: {msg}")
            continue

        # 2) sample (как в validator_cli.profile_all)
        sample = ds.get("sample", defaults.get("sample"))
        if sample:
            df = df.sample(frac=float(sample), random_state=42)

        # 3) шаги
        try:
            steps = ds.get("steps", [])
            df_out = _apply_steps(df, steps, defaults=defaults)
        except Exception as e:
            msg = f"processing error: {e}"
            typer.echo(f"❌ {name}: {msg}")
            errors_total.append(f"{name}: {msg}")
            continue

        # 4) сохранение
        try:
            output = ds["output"]
            _save_df(df_out, output)
            typer.echo(f"✅ {name}: OK")
        except Exception as e:
            msg = f"save error: {e}"
            typer.echo(f"❌ {name}: {msg}")
            errors_total.append(f"{name}: {msg}")

    if errors_total:
        raise typer.Exit(code=1)

    typer.echo("✅ Все датасеты успешно предобработаны")


@app.command("label")
def make_label(
    input_path: Path = typer.Option(..., help="Путь к мастер-датасету после join-ов"),
    output_path: Path = typer.Option(..., help="Куда сохранить с таргетом"),
    customer_col: str = "customer_id",
    purchase_ts_col: str = "order_purchase_timestamp",
    target_col: str = "churned",
    horizon_days: int = 120,
    reference_date: str = "max",  # "max" или '2018-09-01'
    filter_status_col: str = "order_status",
    keep_statuses: str = "delivered",  # через запятую для нескольких
    force: bool = False,
) -> None:
    """Создаёт столбец таргета оттока и сохраняет расширенный датасет.
    
    Оборачивает :func:`olist_churn_prediction.targets.create_churn_label`.
    
    Args:
        input_path: Путь к мастер-датасету (``.csv``/``.parquet``) после join-ов.
        output_path: Куда сохранить результат с таргетом.
        customer_col: Имя столбца с идентификатором клиента.
        purchase_ts_col: Имя столбца с датой/временем покупки.
        target_col: Имя создаваемого столбца-таргета.
        horizon_days: Горизонт давности для определения оттока (в днях).
        reference_date: Контрольная дата (``"max"`` или строка вида ``YYYY-MM-DD``).
        filter_status_col: Имя столбца статуса заказа для фильтрации.
        keep_statuses: Список статусов (через запятую), которые оставить.
        force: Пересоздавать таргет даже при наличии столбца ``target_col``.
    
    Returns:
        None
    """
    df = pd.read_parquet(input_path) if input_path.suffix==".parquet" else pd.read_csv(input_path)
    ks = tuple(s.strip() for s in keep_statuses.split(",")) if keep_statuses else None

    df_out = create_churn_label(
        df,
        customer_col=customer_col,
        purchase_ts_col=purchase_ts_col,
        target_col=target_col,
        horizon_days=horizon_days,
        reference_date=reference_date,
        filter_status_col=filter_status_col,
        keep_statuses=ks,
        force=force,
    )

    if output_path.suffix==".parquet":
        df_out.to_parquet(output_path, index=False)
    else:
        df_out.to_csv(output_path, index=False)
    typer.echo(f"Saved with target to: {output_path}")
    
cli = get_command(app)

if __name__ == "__main__":
    app()
