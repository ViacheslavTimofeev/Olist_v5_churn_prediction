import sys
from pathlib import Path
from typing import Dict, Any

import typer
from typer.main import get_command
import yaml
import pandas as pd

app = typer.Typer(add_completion=False)


def _ensure_columns(df: pd.DataFrame, schema_cols: set, keep_unknown: bool):
    missing = schema_cols - set(df.columns)
    unknown = set(df.columns) - schema_cols
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not keep_unknown and unknown:
        raise ValueError(f"Unknown columns present (set keep_unknown_columns=true to allow): {sorted(unknown)}")

def _cast_string(s: pd.Series) -> pd.Series:
    return s.astype("string")

def _cast_int(s: pd.Series, mode: str, nullable: bool=True) -> pd.Series:
    ser = pd.to_numeric(s, errors=("raise" if mode=="strict" else "coerce"))
    return ser.astype("Int64" if nullable else "int64")

def _cast_float(s: pd.Series, mode: str) -> pd.Series:
    return pd.to_numeric(s, errors=("raise" if mode=="strict" else "coerce")).astype("float64")

def _cast_bool(s: pd.Series, mode: str) -> pd.Series:
    # Универсальный маппинг; при желании вынести в YAML
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    x = s.astype("string").str.strip().str.lower().map(mapping)
    if mode == "strict" and x.isna().any():
        bad = s[x.isna()]
        raise ValueError(f"Boolean cast failed for values: {bad.unique()[:20]}")
    return x.astype("boolean")  # nullable boolean

def _cast_datetime(s: pd.Series, mode: str, fmt: str|None, drop_tz: bool=False) -> pd.Series:
    # Если drop_tz=True, читаем с utc=True и затем убираем TZ.
    errors = "raise" if mode == "strict" else "coerce"
    if drop_tz:
        dt = pd.to_datetime(s, format=fmt, errors=errors, utc=True)
        # Если какие-то значения без TZ, pandas всё равно вернёт tz-aware при utc=True
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        dt = pd.to_datetime(s, format=fmt, errors=errors)
    return dt

def _cast_category(s: pd.Series, cats: list[str]|None, ordered: bool=False) -> pd.Series:
    if cats:
        cat = pd.Categorical(s.astype("string"), categories=cats, ordered=ordered)
        return pd.Series(cat)
    return s.astype("category")

def _cast_col(df: pd.DataFrame, col: str, spec: Dict[str, Any], mode: str) -> pd.Series:
    """Приводит одну колонку к типу по спецификации схемы.

    Поддерживаемые типы: string, int, float, bool, datetime, category.

    Args:
        df: Датафрейм с исходными данными.
        col: Имя колонки.
        spec: Спецификация для колонки (ключ `type`, опции: `nullable`, `format`, `drop_tz`, `categories`, `ordered`).
        mode: Режим ошибок: "strict" (ошибки — исключения) или "coerce" (некорректные значения -> NaN/NaT).

    Returns:
        Преобразованная серия.

    Raises:
        ValueError: Если указан неподдерживаемый тип или приведение в strict-режиме невозможно.
    """
    t = spec["type"]
    if t == "string":
        return _cast_string(df[col])
    if t == "int":
        return _cast_int(df[col], mode, nullable=spec.get("nullable", True))
    if t == "float":
        return _cast_float(df[col], mode)
    if t == "bool":
        return _cast_bool(df[col], mode)
    if t == "datetime":
        return _cast_datetime(df[col], mode, fmt=spec.get("format"), drop_tz=spec.get("drop_tz", False))
    if t == "category":
        return _cast_category(df[col], spec.get("categories"), bool(spec.get("ordered", False)))
    raise ValueError(f"Unsupported type for {col}: {t}")

@app.command()
def cast(
    input_path: Path = typer.Argument(..., help="Входной CSV/Parquet"),
    schema_path: Path = typer.Option(..., "--schema", "-s", help="YAML со схемой типов"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Путь для сохранения"),
    dry_run: bool = typer.Option(False, help="Не сохранять, только отчёт"),
    csv_sep: str = typer.Option(",", help="Разделитель для CSV, если нужно"),
):
    cfg = yaml.safe_load(open(schema_path, "r", encoding="utf-8"))
    spec = cfg["schema"]
    mode = cfg.get("options", {}).get("mode", "strict")
    keep_unknown = cfg.get("options", {}).get("keep_unknown_columns", False)

    # 1) чтение без принудительных dtypes (чтобы не потерять исходные значения)
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path, sep=csv_sep, low_memory=False)
    else:
        df = pd.read_parquet(input_path)

    # 2) структурная проверка колоночного состава
    _ensure_columns(df, set(spec.keys()), keep_unknown)

    # 3) приведение типов + сбор отчёта об ошибках
    report = []
    for col, colspec in spec.items():
        before_na = df[col].isna().sum()
        try:
            df[col] = _cast_col(df, col, colspec, mode)
        except Exception as e:
            typer.secho(f"[ERROR] Column '{col}' cast failed: {e}", fg=typer.colors.RED)
            raise
        after_na = df[col].isna().sum()
        if after_na > before_na:
            report.append({"column": col, "new_missing": int(after_na - before_na)})

    # 4) отчёт
    if report:
        typer.secho("Cast report (new missing due to coercion):", fg=typer.colors.YELLOW)
        for r in report:
            typer.echo(f"  {r['column']}: +{r['new_missing']} NaN/NaT")

    # 5) сохранение
    if not dry_run:
        out = output_path or input_path.with_suffix(".typed.parquet")
        if out.suffix.lower() == ".csv":
            df.to_csv(out, index=False)
        else:
            df.to_parquet(out, index=False)
        typer.secho(f"Saved typed data -> {out}", fg=typer.colors.GREEN)
    else:
        typer.secho("Dry run: nothing saved.", fg=typer.colors.BLUE)


@app.command()
def cast_all(manifest: Path = typer.Argument("validations/validation_manifest.yaml"),
             fail_fast: bool = typer.Option(True, help="Остановиться при первой ошибке")):
    cfg = yaml.safe_load(open(manifest, "r", encoding="utf-8"))
    dfl_cast = (cfg.get("defaults") or {}).get("cast", {})
    errors = []

    for ds in cfg["datasets"]:
        name = ds["name"]
        src = Path(ds["path"])
        cast_cfg = {**dfl_cast, **(ds.get("cast") or {})}
        schema_path = Path(cast_cfg["schema"])

        # склеиваем output по умолчанию, если не указан
        output = cast_cfg.get("output")
        if not output:
            out_dir = Path(cast_cfg.get("output_dir", "data/interim/cli_related/typed"))
            out_dir.mkdir(parents=True, exist_ok=True)
            output = out_dir / f"{name}.typed.parquet"

        try:
            # ВАЖНО: прокинь сюда override'ы, если в твоём cast() они поддерживаются
            cast(
                input_path=src,
                schema_path=schema_path,
                output_path=Path(output),
                dry_run=False,
                csv_sep=cast_cfg.get("csv_sep", ","),
            )
            typer.secho(f"[OK] {name} -> {output}", fg=typer.colors.GREEN)
        except Exception as e:
            msg = f"[FAIL] {name}: {e}"
            errors.append(msg)
            typer.secho(msg, fg=typer.colors.RED)
            if fail_fast:
                raise

    if errors and not fail_fast:
        raise SystemExit(1)
        
cli = get_command(app)

if __name__ == "__main__":
    app()
