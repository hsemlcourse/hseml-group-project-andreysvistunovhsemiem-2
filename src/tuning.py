"""Подбор гиперпараметров для четырёх моделей-кандидатов через RandomizedSearchCV.

Тюнинг идёт на train-выборке через StratifiedKFold; финальная оценка — на отложенном test.
Все сетки и random-state зафиксированы (`settings.random_state`), чтобы прогон был воспроизводимым.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import settings
from src.cv import stratified_cv
from src.metrics import ClassificationReport, evaluate
from src.modeling import build_tree_preprocessor, make_train_test
from src.preprocessing import build_preprocessor


@dataclass(frozen=True)
class TuningResult:
    name: str
    best_params: dict[str, Any]
    best_cv_score: float
    test_report: ClassificationReport

    def summary_row(self) -> dict:
        row = {"name": self.name, "cv_roc_auc": self.best_cv_score}
        row.update(self.test_report.to_dict())
        return row


def _logreg_search(X_train: pd.DataFrame) -> tuple[Pipeline, dict[str, list]]:
    pre = build_preprocessor(X_train)
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=settings.random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    grid = {
        "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    }
    return pipe, grid


def _rf_search(X_train: pd.DataFrame) -> tuple[Pipeline, dict[str, list]]:
    pre = build_tree_preprocessor(X_train)
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=settings.random_state,
                ),
            ),
        ]
    )
    grid = {
        "clf__n_estimators": [300, 500, 800],
        "clf__max_depth": [None, 6, 10, 16],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__max_features": ["sqrt", 0.5, 0.8],
    }
    return pipe, grid


def _xgb_search(X_train: pd.DataFrame) -> tuple[Pipeline, dict[str, list]]:
    pre = build_tree_preprocessor(X_train)
    # вес положительного класса = ratio neg/pos на train; считается в run_tuning.
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                XGBClassifier(
                    eval_metric="auc",
                    tree_method="hist",
                    random_state=settings.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    grid = {
        "clf__n_estimators": [300, 500, 800],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.02, 0.05, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__reg_lambda": [0.5, 1.0, 5.0],
    }
    return pipe, grid


def _lgbm_search(X_train: pd.DataFrame) -> tuple[Pipeline, dict[str, list]]:
    pre = build_tree_preprocessor(X_train)
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LGBMClassifier(
                    class_weight="balanced",
                    random_state=settings.random_state,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )
    grid = {
        "clf__n_estimators": [300, 500, 800],
        "clf__num_leaves": [31, 63, 127],
        "clf__max_depth": [-1, 6, 10],
        "clf__learning_rate": [0.02, 0.05, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__reg_lambda": [0.0, 0.5, 5.0],
    }
    return pipe, grid


SEARCH_BUILDERS = {
    "logreg_fe": _logreg_search,
    "random_forest": _rf_search,
    "xgboost": _xgb_search,
    "lightgbm": _lgbm_search,
}


def tune_one(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int = 25,
) -> TuningResult:
    """Тюнит одну модель и оценивает её на test."""
    pipe, grid = SEARCH_BUILDERS[name](X_train)

    if name == "xgboost":
        # XGB не принимает class_weight; компенсируем дисбаланс через scale_pos_weight.
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        pipe.named_steps["clf"].set_params(scale_pos_weight=neg / max(pos, 1.0))

    search = RandomizedSearchCV(
        pipe,
        grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=stratified_cv(),
        random_state=settings.random_state,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best: Pipeline = search.best_estimator_
    proba = best.predict_proba(X_test)[:, 1]
    pred = best.predict(X_test)
    return TuningResult(
        name=name,
        best_params={k: _to_jsonable(v) for k, v in search.best_params_.items()},
        best_cv_score=float(search.best_score_),
        test_report=evaluate(y_test.values, pred, proba),
    )


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def run_tuning(
    n_iter: int = 25,
    save: bool = True,
) -> dict[str, TuningResult]:
    """Тюнит все 4 модели, сохраняет лучший пайплайн каждой в models/tuned_<name>.joblib."""
    X_train, X_test, y_train, y_test = make_train_test(use_fe=True)
    results: dict[str, TuningResult] = {}
    for name in SEARCH_BUILDERS:
        result = tune_one(name, X_train, y_train, X_test, y_test, n_iter=n_iter)
        results[name] = result
        if save:
            # Перезапускаем фит для полноценного pickle-объекта (RandomizedSearchCV держит refit).
            pipe, _ = SEARCH_BUILDERS[name](X_train)
            if name == "xgboost":
                pos = float((y_train == 1).sum())
                neg = float((y_train == 0).sum())
                pipe.named_steps["clf"].set_params(scale_pos_weight=neg / max(pos, 1.0))
            pipe.set_params(**result.best_params)
            pipe.fit(X_train, y_train)
            path: Path = settings.models_dir / f"tuned_{name}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, path)
    return results


def tuning_results_to_frame(results: dict[str, TuningResult]) -> pd.DataFrame:
    return pd.DataFrame([r.summary_row() for r in results.values()])


if __name__ == "__main__":
    out = run_tuning()
    print(tuning_results_to_frame(out).to_string(index=False))
