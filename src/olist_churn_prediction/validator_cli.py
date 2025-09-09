"""CLI и ядро валидации датасетов проекта Olist Churn.

Этот модуль содержит команды CLI (Typer) и внутренние утилиты для:
- построения *suite* (профиля) датасета по наблюдаемой структуре и статистикам,
- проверки датасета на расхождения со *suite* (структура, пропуски, выход за
числовые/датовые границы, появление новых категорий),
- пакетной валидации наборов из манифеста.
"""

from __future__ import annotations

import json
import pathlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import yaml
import glob
import os
import typer
from typer.main import get_command
import numpy as np
import pandas as pd
import pathlib
import json
import pandera as pa
from pandera import Column, Check # noqa: F401 (оставлено для обратной совместимости)
from concurrent.futures import ThreadPoolExecutor

app = typer.Typer()


def _quant_bounds(x: pd.Series, qlow=0.01, qhigh=0.99, pad=0.1) -> tuple[float, float]:
    """Вычисляет «мягкие» числовые границы по квантилям с запасом.

    Берёт квантили ``qlow`` и ``qhigh`` (по умолчанию 1% и 99%), вычисляет
    ширину ядра распределения (``hi - lo``) и расширяет границы на ``pad``
    долей этой ширины в обе стороны. Такая эвристика помогает не заваливать
    валидацию редкими, но допустимыми значениями.

    Args:
        x: Числовая серия.
        qlow: Нижний квантиль в диапазоне ``[0, 1]``.
        qhigh: Верхний квантиль в диапазоне ``[0, 1]`` (должен быть > ``qlow``).
        pad: Доля расширения диапазона относительно ``(qhigh - qlow)``.

    Returns:
        Пара ``(low, high)`` расширенных границ как ``float``.

    Examples:
        >>> import pandas as pd
        >>> s = pd.Series([0, 1, 2, 3, 100])
        >>> _quant_bounds(s, 0.01, 0.99, 0.1) # doctest: +SKIP
        (low, high)
    """
    lo, hi = x.quantile(qlow), x.quantile(qhigh)
    span = hi - lo if np.isfinite(hi - lo) else 0
    return float(lo - pad*span), float(hi + pad*span)

    
@app.command()
def profile(input: str, name: str = None, out_dir: str = "validations") -> None:
    """Строит профиль (suite) по датасету и сохраняет в JSON.

    Для каждого столбца сохраняются: тип данных, доля пропусков, для числовых —
    «мягкие» границы, для малочисленных категориальных — перечень категорий.

    Args:
        input: Путь к файлу датасета (``.csv`` или ``.parquet``).
        name: Логическое имя датасета. Если не задано — используется ``stem`` от ``input``.
        out_dir: Корневая папка для сохранения результатов (``validations``).

    Raises:
        OSError: При ошибке чтения/записи файлов.

    Example:
        Запуск из терминала::
    
            python -m olist_churn_prediction.validator_cli profile \
            --input data/processed/customers.parquet \
            --name customers
    """
    if input.endswith(".parquet"):
        df = pd.read_parquet(input)
    else:
        df = pd.read_csv(input)

    suite = {"columns": {}, "row_count": len(df)}
    for c in df.columns:
        s = df[c]
        info = {
            "dtype": str(pd.api.types.infer_dtype(s, skipna=True)),
            "null_rate": float(s.isna().mean())
        }
        if pd.api.types.is_numeric_dtype(s):
            lo, hi = _quant_bounds(s.dropna())
            info["num_bounds"] = {"low": lo, "high": hi}
        elif s.nunique() <= 50:
            info["categories"] = [str(v) for v in s.dropna().unique().tolist()]
        suite["columns"][c] = info

    suites_dir = Path(out_dir) / "suites"
    suites_dir.mkdir(parents=True, exist_ok=True)

    out_name = (name or Path(input).stem) + ".json"
    out_path = suites_dir / out_name
    out_path.write_text(json.dumps(suite, indent=2, ensure_ascii=False))

    typer.echo(f"✅ Профиль сохранён: {out_path}")

    
@app.command()
def validate(
    input: str,
    suite_path: str,
    report_dir: str = "validations/reports",
    null_delta_pp: float = 5.0,
    new_cat_ratio: float = 0.02,
) -> None:
    """Проверяет датасет на расхождения со *suite* и сохраняет отчёт.
    
    Проверяются структура (пропавшие/лишние колонки), рост доли пропусков,
    выходы за «мягкие» числовые границы и появление новых категорий.
    
    Args:
        input: Путь к проверяемому датасету (``.csv``/``.parquet``).
        suite_path: Путь к JSON-профилю (*suite*).
        report_dir: Директория для отчётов проверки.
        null_delta_pp: Допустимый рост доли пропусков в п.п. относительно базовой.
        new_cat_ratio: Допустимая доля строк с новыми категориями.
    
    Raises:
        typer.Exit: С кодом 1, если найдены ошибки валидации.
    
    Example:
        Запуск из терминала::
    
            python -m olist_churn_prediction.validator_cli validate \
            --input data/processed/customers.parquet \
            --suite-path validations/suites/customers.json
    """
    df = pd.read_parquet(input) if input.endswith(".parquet") else pd.read_csv(input)
    suite = json.loads(pathlib.Path(suite_path).read_text())
    errors = []

    # 1. Структура
    expected_cols = set(suite["columns"].keys())
    got_cols = set(df.columns)
    missing = expected_cols - got_cols
    extra = got_cols - expected_cols
    if missing: errors.append(f"Нет колонок: {sorted(missing)}")
    # extra можно допустить, если не strict, иначе:
    if extra: errors.append(f"Лишние колонки: {sorted(extra)}")

    # 2. Колоночные проверки
    for c, spec in suite["columns"].items():
        if c not in df.columns: continue
        s = df[c]
        # null-rate guard
        cur_null = s.isna().mean()*100
        base_null = spec.get("null_rate", 0)*100
        if cur_null - base_null > null_delta_pp:
            errors.append(f"{c}: доля NaN {cur_null:.1f}% > baseline {base_null:.1f}% + {null_delta_pp}п.п.")

        # numeric bounds
        if "num_bounds" in spec and pd.api.types.is_numeric_dtype(s):
            lo, hi = spec["num_bounds"]["low"], spec["num_bounds"]["high"]
            mask = ~s.between(lo, hi)
            ratio = mask.mean()
            if ratio > 0.01:
                errors.append(f"{c}: {ratio:.1%} значений вне [{lo:.3g}, {hi:.3g}]")

        # categories drift
        if "categories" in spec and s.notna().any():
            obs = set(map(str, s.dropna().astype(str).unique()))
            base = set(spec["categories"])
            novel = obs - base
            if novel:
                share = s.astype(str).isin(novel).mean()
                if share > new_cat_ratio:
                    errors.append(f"{c}: новые категории {sorted(novel)} ({share:.1%} строк)")

    # 3) отчёт
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    rep_dir = pathlib.Path(report_dir) / ts
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "summary.json").write_text(json.dumps({"errors": errors, "rows": len(df)}, indent=2, ensure_ascii=False))

    if errors:
        typer.echo("Валидация провалена:")
        for e in errors: typer.echo(f" - {e}")
        raise typer.Exit(code=1)

    typer.echo(f"Валидация пройдена. Отчёт: {rep_dir}/summary.json")


def _load_df(ds: dict) -> pd.DataFrame:
    """Загружает DataFrame из описания датасета в манифесте.

    Приоритет отдаётся пути ``cast.output`` (если он существует); в противном
    случае используется исходный путь ``path``. Поддерживаются ``.csv`` и
    ``.parquet``.
    
    Args:
        ds: Секция датасета из YAML-манифеста.
    
    Returns:
        Загруженный ``pd.DataFrame``.
    
    Raises:
        FileNotFoundError: Если указанный файл не существует.
        ValueError: Если расширение файла не поддерживается.
    """
    # 1) если есть cast.output — читаем его
    out = (ds.get("cast") or {}).get("output")
    if out and Path(out).exists():
        return pd.read_parquet(out) if str(out).endswith(".parquet") else pd.read_csv(out)
    # 2) иначе читаем исходник
    src = ds["path"]
    return pd.read_parquet(src) if str(src).endswith(".parquet") else pd.read_csv(src)


@app.command()
def profile_all(manifest: str = "configs/validation_manifest.yaml") -> None:
    """Строит *suite* для всех датасетов по YAML-манифесту.

    Для каждого датасета загружает данные (с учётом сэмплирования), вызывает
    :func:`profile` и сохраняет профиль в ``validations/suites``.

    Args:
        manifest: Путь к YAML-манифесту валидаций.

    Raises:
        OSError: При ошибках чтения/записи.
        KeyError: Если в манифесте отсутствуют обязательные поля.
    """
    cfg = yaml.safe_load(open(manifest))
    for ds in cfg["datasets"]:
        df = _load_df(ds)
        # опционально: сэмпл
        sample = cfg["defaults"].get("sample") or ds.get("sample")
        if sample: df = df.sample(frac=float(sample), random_state=42)
        # вызываем твою profile-логику:
        name = ds["name"]
        profile(input=ds.get("input", name), name=name)  # можно вынести ядро в функцию


def _validate_df_core(
    df: pd.DataFrame,
    suite_path: str | os.PathLike[str],
    ds: dict | None = None,
    defaults: dict | None = None,
) -> list[str]:
    """Проверяет ``DataFrame`` по сохранённому *suite* и порогам.
    
    Проверяются структура (``missing``/``extra`` columns), рост доли пропусков,
    выход за числовые/датовые границы и новизна категорий, с учётом глобальных
    и пер-колоночных порогов.
    
    Args:
        df: Датафрейм для проверки.
        suite_path: Путь к JSON-профилю (*suite*).
        ds: Переопределения порогов и структуры на уровне датасета.
            Пример структуры::
            
                {
                  "thresholds": {...},
                  "columns": {...},
                  "strict_structure": true
                }

        defaults: Глобальные значения по умолчанию (обычно из ``manifest.defaults.validate``).
    
    Returns:
        Список строк с найденными ошибками. Пустой список означает, что проверка
        пройдена.
    
    Raises:
        Exception: При ошибках чтения *suite* или конвертации типов дат/времени.
    """
    defaults = defaults or {}
    thresholds = {
        "null_delta_pp": defaults.get("null_delta_pp", 5.0),
        "new_cat_ratio": defaults.get("new_cat_ratio", 0.02),
        "oob_ratio":     defaults.get("oob_ratio", 0.01),  # доля значений вне границ
    }
    if ds and isinstance(ds, dict):
        th_ds = ds.get("thresholds", {})
        for k in thresholds:
            thresholds[k] = th_ds.get(k, thresholds[k])

    strict_structure = defaults.get("strict_structure", True)
    if ds and "strict_structure" in ds:
        strict_structure = ds["strict_structure"]

    col_overrides = (ds or {}).get("columns", {})

    suite = json.loads(pathlib.Path(suite_path).read_text())
    errors = []

    # 1) структура
    expected_cols = set(suite["columns"].keys())
    got_cols = set(df.columns)
    missing = expected_cols - got_cols
    extra   = got_cols - expected_cols
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")
    if strict_structure and extra:
        errors.append(f"extra columns: {sorted(extra)}")

    # 2) проверки по колонкам
    for c, spec in suite["columns"].items():
        if c not in df.columns:
            continue
        s = df[c]

        # null-rate delta
        base_null = float(spec.get("null_rate", 0.0)) * 100.0
        cur_null  = float(s.isna().mean()) * 100.0
        null_delta_pp = col_overrides.get(c, {}).get("null_delta_pp", thresholds["null_delta_pp"])
        if cur_null - base_null > null_delta_pp:
            errors.append(f"{c}: nulls {cur_null:.1f}% > baseline {base_null:.1f}% + {null_delta_pp}pp")

        # numeric bounds
        if "num_bounds" in spec and pd.api.types.is_numeric_dtype(s):
            lo, hi = spec["num_bounds"]["low"], spec["num_bounds"]["high"]
            ratio = (~s.between(lo, hi)).mean()
            oob_thr = col_overrides.get(c, {}).get("oob_ratio", thresholds["oob_ratio"])
            if ratio > oob_thr:
                errors.append(f"{c}: {ratio:.1%} out of bounds [{lo}, {hi}] (> {oob_thr:.1%})")

        # categories novelty
        if "categories" in spec:
            base = set(map(str, spec["categories"]))
            obs  = set(map(str, s.dropna().astype(str).unique()))
            novel = obs - base
            if novel:
                share = s.astype(str).isin(novel).mean()
                new_cat_thr = col_overrides.get(c, {}).get("new_cat_ratio", thresholds["new_cat_ratio"])
                if share > new_cat_thr:
                    errors.append(f"{c}: novel categories {sorted(novel)} share={share:.1%} (> {new_cat_thr:.1%})")

        # datetime bounds
        if "datetime_bounds" in spec:
            dt = s
            if not pd.api.types.is_datetime64_any_dtype(dt):
                dt = pd.to_datetime(dt, errors="coerce", infer_datetime_format=True, utc=True)
            else:
                # нормализуем к UTC
                if getattr(dt.dt, "tz", None) is None:
                    dt = dt.dt.tz_localize("UTC")
                else:
                    dt = dt.dt.tz_convert("UTC")
            lo = pd.Timestamp(spec["datetime_bounds"]["low"])
            hi = pd.Timestamp(spec["datetime_bounds"]["high"])
            ratio = ((dt < lo) | (dt > hi)).mean()
            oob_thr = col_overrides.get(c, {}).get("oob_ratio", thresholds["oob_ratio"])
            if ratio > oob_thr:
                errors.append(
                    f"{c}: {ratio:.1%} dates outside [{lo.isoformat()}, {hi.isoformat()}] (> {oob_thr:.1%})"
                )

    return errors


@app.command()
def validate_all(manifest: str = "configs/validation_manifest.yaml") -> None:
    """Пакетно валидирует все датасеты, описанные в YAML-манифесте.

    Для каждого набора выбирает источник данных, ищет путь к *suite*, применяет
    :func:`_validate_df_core` с учётом глобальных и пер-датасетных порогов и
    печатает результат. Если были ошибки — завершает процесс с кодом 1.

    Args:
        manifest: Путь к YAML-манифесту валидаций.

    Raises:
        typer.Exit: Если для любого датасета найдены ошибки валидации.

    Example:
        Запуск из терминала::

            python -m olist_churn_prediction.validator_cli validate-all \
                --manifest configs/validation_manifest.yaml
    """
    cfg = yaml.safe_load(Path(manifest).read_text())
    defaults_validate = (cfg.get("defaults") or {}).get("validate", {})
    errors_total: list[str] = []

    for ds in cfg["datasets"]:
        if ds.get("enabled") is False or ds.get("validate") is False:
            continue

        name = ds["name"]

        # 1) загрузка данных
        try:
            df = _load_df(ds)
        except Exception as e:
            msg = f"load error {e}"
            typer.echo(f"❌ {name}: {msg}")
            errors_total.append(f"{name}: {msg}")
            continue

        # 2) путь к suite
        vcfg = ds.get("validate") or {}
        suite_path = vcfg.get("suite") or ds.get("suite")
        if not suite_path:
            msg = "suite not specified (expected validate.suite)"
            typer.echo(f"❌ {name}: {msg}")
            errors_total.append(f"{name}: {msg}")
            continue
        if not Path(suite_path).exists():
            msg = f"suite not found: {suite_path}"
            typer.echo(f"❌ {name}: {msg}")
            errors_total.append(f"{name}: {msg}")
            continue

        # 3) контекст порогов
        ds_ctx = {
            "thresholds": vcfg.get("thresholds", {}),
            "columns": vcfg.get("columns", {}),
            "strict_structure": vcfg.get("strict_structure", defaults_validate.get("strict_structure", True)),
        }

        # 4) запуск ядра
        try:
            errs = _validate_df_core(df, suite_path, ds_ctx, defaults_validate)
        except Exception as e:
            errs = [f"runtime error {e}"]

        if errs:
            for e in errs:
                typer.echo(f"❌ {name}: {e}")
            errors_total.extend(f"{name}: {e}" for e in errs)
        else:
            typer.echo(f"✅ {name}: OK")

    if errors_total:
        raise typer.Exit(code=1)

    typer.echo("✅ Все таблицы прошли валидацию")
    
cli = get_command(app)

if __name__ == "__main__":
    app()