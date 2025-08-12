# preproc_cli.py
# CLI для автоматизации предобработки по манифесту (в стиле validator_cli.py)
from __future__ import annotations
import json, glob, os, yaml, typer
from pathlib import Path
from typing import Dict, Any, Union, List, Callable
import pandas as pd

# --- импорт твоих функций предобработки ---
from olist_churn_prediction import feature_processing as fp  # lowercase_categoricals, disambiguate_city_state, group_by_features

app = typer.Typer(help="Preprocessing pipeline CLI (manifest-driven)")

# ======== Загрузка/сохранение — по мотивам validator_cli.py ========

def _load_df(entry: Dict[str, Any]) -> pd.DataFrame:
    """
    Совместимый загрузчик как в validator_cli.py:
      - file: input с glob-маской (берём самый свежий)
      - sql:  query + conn_env (строка подключения в переменной окружения)
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

def _save_df(df: pd.DataFrame, output: str):
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    typer.echo(f"💾 Saved: {out}")

# ======== Ядро применения шагов ========

def _apply_steps(
    df: pd.DataFrame,
    steps: List[Dict[str, Any]],
    defaults: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Применяет последовательность шагов к DataFrame.
    Каждый шаг — dict с ключом 'op' и параметрами. Примеры см. ниже.
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
                # если не передали, используем все object/category (без автоопределения в модуле)
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

        elif op == "rename_columns":
            # утилитарный шаг: {"old":"new", ...}
            mapping = step["mapping"]
            X = X.rename(columns=mapping)

        elif op == "drop_columns":
            cols = step["cols"]
            X = X.drop(columns=cols, errors="ignore")

        else:
            raise ValueError(f"Unknown op '{op}' in step #{i}")

    return X

# ======== Команды CLI ========

@app.command()
def apply(
    input: str = typer.Argument(..., help="Входной файл (.csv/.parquet)"),
    output: str = typer.Argument(..., help="Куда сохранить результат"),
    steps_json: str = typer.Option(None, help="JSON со списком шагов"),
    sample: float = typer.Option(None, help="Доля сэмпла для отладки, напр. 0.1"),
):
    """
    Применить шаги предобработки к одному датасету (без манифеста).
    Пример:
      preproc apply data/raw.csv data/interim/clean.parquet \\
        --steps-json '[{\"op\":\"lowercase_categoricals\",\"cat_cols\":[\"customer_city\",\"customer_state\"]}]'
    """
    df = pd.read_parquet(input) if input.endswith(".parquet") else pd.read_csv(input)
    if sample:
        df = df.sample(frac=float(sample), random_state=42)
    steps = json.loads(steps_json) if steps_json else []
    df_out = _apply_steps(df, steps)
    _save_df(df_out, output)

@app.command()
def run(manifest: str = typer.Argument("preprocessing.yaml", help="Манифест пайплайна")):
    """
    Запустить предобработку для набора датасетов по манифесту YAML.
    Структура манифеста совместима по духу с validator_cli: defaults + datasets[].
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

if __name__ == "__main__":
    app()
