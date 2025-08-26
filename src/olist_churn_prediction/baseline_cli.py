"""Сборка baseline-пайплайна и утилиты для обучения/оценки.

Содержит билдеры препроцессинга и моделей, а также команды Typer (dry_run, cv, fit, predict).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import typer
from typer.main import get_command
import yaml

# MLflow
try:
    import mlflow
    import mlflow.sklearn
except Exception:  # если не установлен — просто продолжим без него
    mlflow = None

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = typer.Typer(add_completion=False, no_args_is_help=True)
# ------------------------------
# Utils
# ------------------------------

def _read_df(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def _infer_feature_types(df: pd.DataFrame, target: str, id_cols: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    id_cols = id_cols or []
    drop = set([target, *id_cols])
    # Категориальные: object, string, category, boolean
    cat_mask = df.drop(columns=list(drop), errors="ignore").select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    # Числовые: number (int/float)
    num_mask = df.drop(columns=list(drop), errors="ignore").select_dtypes(
        include=["number"]
    ).columns.tolist()
    return num_mask, cat_mask


def _choose_scoring(y: pd.Series) -> List[str]:
    classes = y.dropna().unique()
    if len(classes) <= 2:
        return [
            "f1",
            "precision",
            "recall",
            "roc_auc",
        ]
    else:
        # Для мультикласса
        return [
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "roc_auc_ovr_weighted",
        ]


def _build_model(name: str, params: dict):
    name = name.lower()
    if name == "logreg":
        default = dict(max_iter=1000, n_jobs=None, solver="lbfgs")
        # n_jobs игнорируется для lbfgs, но оставлен для совместимости
        cfg = {**default, **(params or {})}
        return LogisticRegression(**cfg)
    elif name == "rf":
        default = dict(n_estimators=300, random_state=42, n_jobs=-1)
        cfg = {**default, **(params or {})}
        return RandomForestClassifier(**cfg)
    else:
        raise ValueError(f"Unknown model: {name}")


def _build_preprocessor(numeric_features: List[str], categorical_features: List[str], *, sparse_ohe: bool = True) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), # заполнение ппропусков для num features
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")), # заполнение пропусков для cat features
        # ВАЖНО: в новых версиях sklearn используем sparse_output
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_ohe, dtype=np.float32)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,  # не обязан, но часто полезно
    )
    return pre


def _ensure_output_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _flatten_params(d: dict, prefix: str = "", sep: str = ".") -> dict:
    """Плоские ключи для логирования в MLflow."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_params(v, key, sep))
        else:
            out[key] = v
    return out


def _mlflow_enabled(cfg: dict) -> bool:
    return bool(cfg.get("mlflow", {}).get("enabled", False) and (mlflow is not None))


def _mlflow_init(cfg: dict):
    if not _mlflow_enabled(cfg):
        return False
    mlf = cfg["mlflow"]
    tracking_uri = mlf.get("tracking_uri", None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    exp_name = mlf.get("experiment", "default")
    mlflow.set_experiment(exp_name)
    return True


def _mlflow_log_config_and_features(cfg: dict, num_cols: list[str], cat_cols: list[str]):
    # логируем конфиг YAML как текст
    try:
        mlflow.log_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), "config.yaml")
    except Exception:
        pass
    # логируем списки колонок отдельным артефактом
    try:
        text = (
            f"numeric_features ({len(num_cols)}): {num_cols}"
            f"categorical_features ({len(cat_cols)}): {cat_cols}"
        )
        mlflow.log_text(text, "features.txt")
    except Exception:
        pass


# ------------------------------
# Core builders
# ------------------------------

def build_pipeline(config: dict, df: Optional[pd.DataFrame] = None) -> Tuple[Pipeline, List[str], List[str]]:
    """Собирает sklearn-пайплайн из YAML-конфига.

    Args:
        config: Конфиг с ключами `target`, `id_cols`, `model`, `numeric_features`, `categorical_features`, `cv`, `random_state`, `output_dir`.
        df: Необязательный сэмпл данных для авто-инференса типов колонок, если списки фич в конфиге не заданы.

    Returns:
        Tuple[pipeline, numeric_features, categorical_features].

    Raises:
        ValueError: Если указана неизвестная модель в `config["model"]["name"]`.
    """
    target = config["target"]
    id_cols = config.get("id_cols", [])

    num_cols = config.get("numeric_features") or []
    cat_cols = config.get("categorical_features") or []

    if (not num_cols or not cat_cols) and df is not None:
        inf_num, inf_cat = _infer_feature_types(df, target, id_cols)
        num_cols = num_cols or inf_num
        cat_cols = cat_cols or inf_cat

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "logreg")
    model_params = model_cfg.get("params", {})

    # Для деревьев предпочтительнее dense ohe (в простом baseline)
    sparse_ohe = False if model_name.lower() == "rf" else True

    pre = _build_preprocessor(num_cols, cat_cols, sparse_ohe=sparse_ohe)
    model = _build_model(model_name, model_params)

    pipe = Pipeline([
        ("prep", pre),
        ("clf", model),
    ])

    return pipe, num_cols, cat_cols


# ------------------------------
# Commands
# ------------------------------

@app.command()
def dry_run(config: str = typer.Option(..., help="Путь к YAML-конфигу")):
    """Показать, как будет собран пайплайн: списки колонок, модель, метрики."""
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    df = _read_df(cfg["data_path"])[:200]  # небольшой фрагмент для инференса типов
    pipe, num_cols, cat_cols = build_pipeline(cfg, df)

    y = df[cfg["target"]]
    scoring = _choose_scoring(y)

    typer.echo("\n[Baseline dry-run]\n" + "-" * 60)
    typer.echo(f"Data sample shape: {df.shape}")
    typer.echo(f"Target: {cfg['target']}")
    typer.echo(f"Numeric features ({len(num_cols)}): {num_cols}")
    typer.echo(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    typer.echo(f"Model: {cfg.get('model', {}).get('name', 'logreg')}")
    typer.echo(f"CV scoring: {scoring}")


@app.command()
def cv(config: str = typer.Option(..., help="Путь к YAML-конфигу")):
    """Кросс-валидация на всём датасете. Сохраняет metrics.json и (при включённом MLflow) логирует результаты."""
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    df = _read_df(cfg["data_path"])  

    target = cfg["target"]
    id_cols = cfg.get("id_cols", [])

    X = df.drop(columns=[target, *id_cols], errors="ignore")
    y = df[target]

    pipe, num_cols, cat_cols = build_pipeline(cfg, df)

    skf = StratifiedKFold(n_splits=cfg.get("cv", {}).get("n_splits", 5), shuffle=True, random_state=cfg.get("random_state", 42))
    scoring = _choose_scoring(y)

    # MLflow
    use_mlflow = _mlflow_init(cfg)
    run_ctx = mlflow.start_run(run_name=cfg.get("mlflow", {}).get("run_name", "cv")) if use_mlflow else None
    try:
        scores = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1, error_score="raise")
        summary = {k.replace("test_", ""): float(np.mean(v)) for k, v in scores.items() if k.startswith("test_")}

        outdir = _ensure_output_dir(cfg.get("output_dir", "artifacts/baseline"))
        (outdir / "metrics_cv.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        typer.echo("CV metrics (mean):")
        typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
        typer.echo(f"Saved: {outdir/'metrics_cv.json'}")

        if use_mlflow:
            # логируем параметры
            mlflow.log_params({
                "model": cfg.get("model", {}).get("name", "logreg"),
                "cv_n_splits": skf.get_n_splits(),
                "random_state": cfg.get("random_state", 42),
                "n_num_features": len(num_cols),
                "n_cat_features": len(cat_cols),
            })
            mlflow.log_params({f"model__{k}": v for k, v in _flatten_params(cfg.get("model", {}).get("params", {})).items()})
            # логируем метрики
            mlflow.log_metrics(summary)
            # логируем конфиг и списки колонок
            _mlflow_log_config_and_features(cfg, num_cols, cat_cols)
            # положим локальный файл с метриками как артефакт
            try:
                mlflow.log_artifact(str(outdir / "metrics_cv.json"))
            except Exception:
                pass
    finally:
        if run_ctx is not None:
            mlflow.end_run()


@app.command()
def fit(
    config: str = typer.Option(..., help="Путь к YAML-конфигу"),
    save_pipeline: bool = typer.Option(True, help="Сохранять pipeline.joblib"),
):
    """Holdout train/test: обучает, оценивает, сохраняет pipeline и метрики. При включённом MLflow всё логируется."""
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    df = _read_df(cfg["data_path"])  

    target = cfg["target"]
    id_cols = cfg.get("id_cols", [])
    test_size = cfg.get("test_size", 0.2)
    random_state = cfg.get("random_state", 42)

    X = df.drop(columns=[target, *id_cols], errors="ignore")
    y = df[target]

    pipe, num_cols, cat_cols = build_pipeline(cfg, df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # MLflow — опционально
    use_mlflow = _mlflow_init(cfg)
    if use_mlflow and cfg.get("mlflow", {}).get("autolog", False):
        try:
            mlflow.sklearn.autolog(log_models=False)  # модель залогируем вручную после fit
        except Exception:
            pass

    run_ctx = mlflow.start_run(run_name=cfg.get("mlflow", {}).get("run_name", "fit")) if use_mlflow else None
    try:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # метрики
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        }
        # ROC-AUC если возможно
        try:
            if len(np.unique(y_test)) <= 2:
                proba = pipe.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            else:
                proba = pipe.predict_proba(X_test)
                metrics["roc_auc_ovr_weighted"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
        except Exception:
            pass

        outdir = _ensure_output_dir(cfg.get("output_dir", "artifacts/baseline"))
        (outdir / "metrics_holdout.json").write_text(json.dumps({**metrics, "report": report}, indent=2, ensure_ascii=False))

        # сохранить pipeline локально
        if save_pipeline:
            joblib.dump(pipe, outdir / "pipeline.joblib")

        # предсказания на тесте (id_cols если есть)
        pred_df = df.loc[y_test.index, id_cols].copy() if id_cols else pd.DataFrame(index=y_test.index)
        pred_df["y_true"] = y_test.values
        pred_df["y_pred"] = y_pred

        try:
            proba = pipe.predict_proba(X_test)
            if proba.ndim == 1:
                pred_df["proba"] = proba
            else:
                classes = pipe.named_steps["clf"].classes_
                for i, cls in enumerate(classes):
                    pred_df[f"proba_{cls}"] = proba[:, i]
        except Exception:
            pass

        pred_path = outdir / "predictions_holdout.csv"
        pred_df.to_csv(pred_path, index=False)

        typer.echo(f"Saved metrics to: {outdir/'metrics_holdout.json'}")
        if save_pipeline:
            typer.echo(f"Saved pipeline to: {outdir/'pipeline.joblib'}")
        typer.echo(f"Saved predictions to: {pred_path}")

        # MLflow логирование
        if use_mlflow:
            mlflow.log_params({
                "model": cfg.get("model", {}).get("name", "logreg"),
                "random_state": random_state,
                "test_size": test_size,
                "n_num_features": len(num_cols),
                "n_cat_features": len(cat_cols),
            })
            mlflow.log_params({f"model__{k}": v for k, v in _flatten_params(cfg.get("model", {}).get("params", {})).items()})
            mlflow.log_metrics(metrics)

            # классовые отчёты как артефакт
            try:
                mlflow.log_text(json.dumps(report, indent=2, ensure_ascii=False), "classification_report.json")
            except Exception:
                pass

            # конфиг и списки колонок
            _mlflow_log_config_and_features(cfg, num_cols, cat_cols)

            # локальные артефакты
            for p in [outdir / "metrics_holdout.json", pred_path]:
                try:
                    mlflow.log_artifact(str(p))
                except Exception:
                    pass

            # логирование самой модели (sklearn-пайплайна)
            if cfg.get("mlflow", {}).get("log_model", True):
                try:
                    mlflow.sklearn.log_model(pipe, artifact_path="model")
                except Exception:
                    # запасной вариант — положим файл joblib
                    if save_pipeline:
                        try:
                            mlflow.log_artifact(str(outdir / "pipeline.joblib"))
                        except Exception:
                            pass
    finally:
        if run_ctx is not None:
            mlflow.end_run()


@app.command()
def predict(
    pipeline_path: str = typer.Option(..., help="Путь к pipeline.joblib"),
    data_path: str = typer.Option(..., help="Путь к данным (csv/parquet) без таргета"),
    id_cols: Optional[str] = typer.Option(None, help="Через запятую: колонки-идентификаторы для вывода"),
    out_path: Optional[str] = typer.Option(None, help="Куда сохранить предсказания .csv"),
):
    """Применить сохранённый pipeline к новым данным."""
    pipe: Pipeline = joblib.load(pipeline_path)
    df = _read_df(data_path)

    id_columns = [c.strip() for c in id_cols.split(",")] if id_cols else []
    X = df.drop(columns=id_columns, errors="ignore")

    preds = pipe.predict(X)
    out = df[id_columns].copy() if id_columns else pd.DataFrame(index=df.index)
    out["prediction"] = preds

    # вероятности если доступны
    try:
        proba = pipe.predict_proba(X)
        if proba.ndim == 1:
            out["proba"] = proba
        else:
            classes = pipe.named_steps["clf"].classes_
            for i, cls in enumerate(classes):
                out[f"proba_{cls}"] = proba[:, i]
    except Exception:
        pass

    out_path = Path(out_path) if out_path else Path(pipeline_path).with_name("predictions.csv")
    out.to_csv(out_path, index=False)
    typer.echo(f"Saved predictions to: {out_path}")
    
cli = get_command(app)

if __name__ == "__main__":
    app()
