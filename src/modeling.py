"""Пайплайны обучения: baseline (LogReg без FE) и сравнение 5 моделей."""
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.config import settings
from src.data import load_raw
from src.metrics import ClassificationReport, evaluate
from src.preprocessing import (
    CATEGORICAL_BASE,
    build_preprocessor,
    clean,
    feature_engineering,
    split_features_target,
)


def make_train_test(use_fe: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = clean(load_raw())
    if use_fe:
        df = feature_engineering(df)
    X, y = split_features_target(df)
    return train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )


def build_baseline_pipeline(X_train: pd.DataFrame) -> Pipeline:
    pre = build_preprocessor(X_train)
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=settings.random_state,
        solver="lbfgs",
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def train_and_eval_baseline(save_path: Path | None = None) -> dict[str, ClassificationReport]:
    X_train, X_test, y_train, y_test = make_train_test(use_fe=False)

    dummy = DummyClassifier(strategy="stratified", random_state=settings.random_state)
    dummy.fit(X_train, y_train)
    dummy_proba = dummy.predict_proba(X_test)[:, 1]
    dummy_pred = dummy.predict(X_test)

    pipe = build_baseline_pipeline(X_train)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)

    reports = {
        "dummy_stratified": evaluate(y_test.values, dummy_pred, dummy_proba),
        "logreg_baseline": evaluate(y_test.values, pred, proba),
    }

    if save_path is None:
        save_path = settings.models_dir / "baseline.joblib"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, save_path)
    return reports


def build_tree_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """OHE для категорий, числовые признаки — passthrough (деревьям scale не нужен)."""
    categorical = [c for c in X.columns if c in CATEGORICAL_BASE or c == "AGE_BUCKET"]
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
    )


def _build_models(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    seed = settings.random_state
    scaled_pre = build_preprocessor(X_train)
    tree_pre = build_tree_preprocessor(X_train)
    return {
        "knn_k25": Pipeline([("pre", scaled_pre), ("clf", KNeighborsClassifier(n_neighbors=25, n_jobs=-1))]),
        "logreg_fe": Pipeline(
            [
                ("pre", scaled_pre),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs"
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("pre", tree_pre),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("pre", tree_pre),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="auc",
                        tree_method="hist",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("pre", tree_pre),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=400,
                        max_depth=-1,
                        num_leaves=63,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }


def run_experiments(save: bool = True) -> dict[str, ClassificationReport]:
    """Сравнивает 5 моделей (LogReg+FE / KNN / RF / XGBoost / LightGBM) на едином сплите."""
    X_train, X_test, y_train, y_test = make_train_test(use_fe=True)
    reports: dict[str, ClassificationReport] = {}
    for name, pipe in _build_models(X_train).items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        reports[name] = evaluate(y_test.values, pred, proba)
        if save:
            path = settings.models_dir / f"{name}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, path)
    return reports


if __name__ == "__main__":
    baseline = train_and_eval_baseline()
    experiments = run_experiments()
    all_reports = {**baseline, **experiments}
    table = pd.DataFrame({k: v.to_dict() for k, v in all_reports.items()}).T
    print(table.to_string())
