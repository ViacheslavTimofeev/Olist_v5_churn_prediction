import json, pathlib, typer, numpy as np
import pandas as pd, pathlib, json
from pathlib import Path
import pandera as pa
from pandera import Column, Check
from datetime import datetime
import yaml, glob, os
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

app = typer.Typer()

''' функция для вычисления квантилей численных признаков '''
def _quant_bounds(x: pd.Series, qlow=0.01, qhigh=0.99, pad=0.1): 
    lo, hi = x.quantile(qlow), x.quantile(qhigh) # lo - нижняя граница, hi - верхняя
    span = hi - lo if np.isfinite(hi - lo) else 0  # span - ширина ядра распределения
    return float(lo - pad*span), float(hi + pad*span) # возвращает расширенные границы: ниже lo и выше hi на 10% ширины диапазона. Это «мягкие» рамки, чтобы не заваливать валидацию редкими, но нормальными значениями.

''' объявление CLI-команды profile '''
@app.command()
def profile(input: str, name: str = None, out_dir: str = "validations"):
    """
    Собирает профиль (suite) по датасету.
    - input: путь к файлу (csv/parquet)
    - name: (опц.) логическое имя датасета; влияет на имя файла suite.
            Если не задано — берём stem input.
    - out_dir: корневая папка для validations
    """
    # читаем данные
    if input.endswith(".parquet"):
        df = pd.read_parquet(input)
    else:
        df = pd.read_csv(input)

    # строим suite
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

    # куда писать
    suites_dir = Path(out_dir) / "suites"
    suites_dir.mkdir(parents=True, exist_ok=True)

    out_name = (name or Path(input).stem) + ".json"
    out_path = suites_dir / out_name
    out_path.write_text(json.dumps(suite, indent=2, ensure_ascii=False))

    typer.echo(f"✅ Профиль сохранён: {out_path}")

''' объявление CLI-команды validate '''
@app.command()
def validate(input: str, suite_path: str, report_dir: str = "validations/reports",
             null_delta_pp: float = 5.0, new_cat_ratio: float = 0.02):
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

        # dtype (мягко): пытаемся привести к числу/дате по observed типу
        # …

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


def _load_df(entry: Dict[str, Any]) -> pd.DataFrame:
    reader = entry.get("reader", "file")
    if reader == "sql":
        import sqlalchemy as sa
        url = os.environ[entry["conn_env"]]
        engine = sa.create_engine(url)
        return pd.read_sql(sa.text(entry["query"]), engine, params=entry.get("params"))
    else:
        path = sorted(glob.glob(entry["input"]))[-1]  # возьмём самый свежий
        return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

@app.command()
def profile_all(manifest: str = "validations/manifest.yaml"):
    cfg = yaml.safe_load(open(manifest))
    for ds in cfg["datasets"]:
        df = _load_df(ds)
        # опционально: сэмпл
        sample = cfg["defaults"].get("sample") or ds.get("sample")
        if sample: df = df.sample(frac=float(sample), random_state=42)
        # вызываем твою profile-логику:
        name = ds["name"]
        profile(input=ds.get("input", name), name=name)  # можно вынести ядро в функцию

# --- helpers: core validator for one DataFrame based on a saved suite ---
def _validate_df_core(df, suite_path, ds=None, defaults=None):
    """
    Возвращает список ошибок (list[str]) для df по правилам из suite_path.
    Не кидает Exit и ничего не печатает — чистая логика.
    """
    defaults = defaults or {}
    # Базовые пороги
    thresholds = {
        "null_delta_pp": defaults.get("null_delta_pp", 5.0),
        "new_cat_ratio": defaults.get("new_cat_ratio", 0.02),
        "oob_ratio":     defaults.get("oob_ratio", 0.01),  # доля значений вне границ
    }
    # Оверрайды на уровень датасета
    if ds and isinstance(ds, dict):
        th_ds = ds.get("thresholds", {})
        for k in thresholds:
            thresholds[k] = th_ds.get(k, thresholds[k])

    # Строгость структуры
    strict_structure = defaults.get("strict_structure", True)
    if ds and "strict_structure" in ds:
        strict_structure = ds["strict_structure"]

    # Пер-колоночные оверрайды (необяз.)
    col_overrides = (ds or {}).get("columns", {})

    # Загружаем suite
    suite = json.loads(pathlib.Path(suite_path).read_text())
    errors = []

    # --- 1) структура ---
    expected_cols = set(suite["columns"].keys())
    got_cols = set(df.columns)
    missing = expected_cols - got_cols
    extra   = got_cols - expected_cols
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")
    if strict_structure and extra:
        errors.append(f"extra columns: {sorted(extra)}")

    # --- 2) проверки по колонкам ---
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


# --- command: validate-all (без внешних зависимостей на _validate_df_core) ---
@app.command()
def validate_all(manifest: str = "validations/manifest.yaml"):
    """
    Валидирует все датасеты из manifest.yaml, используя внешнюю _load_df(ds).
    Печатает статус по каждому датасету и возвращает exit code 1 при любых ошибках.
    """
    cfg = yaml.safe_load(Path(manifest).read_text())
    defaults = cfg.get("defaults", {})
    errors_total = []

    for ds in cfg["datasets"]:
        # пропускаем выключенные
        if ds.get("enabled") is False or ds.get("validate") is False:
            continue

        name = ds["name"]

        # 1) загрузка данных через внешнюю _load_df(ds)
        try:
            df = _load_df(ds)  # ← тут твоя внешняя функция
        except Exception as e:
            msg = f"load error {e}"
            typer.echo(f"❌ {name}: {msg}")
            errors_total.append(f"{name}: {msg}")
            continue

        # 2) наличие suite
        suite_path = ds["suite"]
        if not Path(suite_path).exists():
            msg = f"suite not found: {suite_path}"
            typer.echo(f"❌ {name}: {msg}")
            errors_total.append(f"{name}: {msg}")
            continue

        # 3) валидация
        try:
            errs = _validate_df_core(df, suite_path, ds, defaults)  # возвращает list[str]
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

'''
@app.command()
def validate_all(manifest: str = "validations/manifest.yaml"):
    cfg = yaml.safe_load(open(manifest))
    defaults = cfg.get("defaults", {})
    errors_total = []
    for ds in cfg["datasets"]:
        df = _load_df(ds)
        suite_path = ds["suite"]
        kw = {
          "null_delta_pp": ds.get("thresholds", {}).get("null_delta_pp", defaults.get("null_delta_pp", 5.0)),
          "new_cat_ratio": ds.get("thresholds", {}).get("new_cat_ratio", defaults.get("new_cat_ratio", 0.02)),
          # передай внутрь и per-column overrides, если реализуешь
        }
        # тут лучше вызвать внутреннюю функцию _validate_df(df, suite, overrides)
        # которая вернёт список ошибок; ниже просто пример:
        try:
            _errors = _validate_df_core(df, suite_path, ds, defaults)  # реализуй как обёртку вокруг твоей validate
        except Exception as e:
            _errors = [f"{ds['name']}: runtime error {e}"]
        if _errors:
            errors_total.extend([f"{ds['name']}: {e}" for e in _errors])

    if errors_total:
        for e in errors_total: typer.echo("❌ " + e)
        raise typer.Exit(code=1)
    typer.echo("✅ Все таблицы прошли валидацию")
'''
if __name__ == "__main__":
    app()