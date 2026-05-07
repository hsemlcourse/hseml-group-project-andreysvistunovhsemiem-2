"""Тесты на модули, добавленные в CP2: cv, threshold, dim_reduction, interpret, tuning."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import settings
from src.cv import cv_score_models
from src.dim_reduction import explained_variance_curve, run_pca_experiment
from src.interpret import permutation_feature_importance, tree_feature_importance
from src.preprocessing import (
    BILL_COLS,
    PAY_AMT_COLS,
    PAY_COLS,
    build_preprocessor,
    clean,
    feature_engineering,
    split_features_target,
)
from src.threshold import (
    best_threshold_by_cost,
    best_threshold_by_f1,
    threshold_grid_to_frame,
)


def _synthetic_df(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "ID": np.arange(n),
            "LIMIT_BAL": rng.integers(10_000, 500_000, n).astype(float),
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": rng.integers(0, 7, n),
            "MARRIAGE": rng.integers(0, 4, n),
            "AGE": rng.integers(21, 70, n),
        }
    )
    for c in PAY_COLS:
        df[c] = rng.integers(-2, 4, n)
    for c in BILL_COLS:
        df[c] = rng.integers(-1_000, 100_000, n).astype(float)
    for c in PAY_AMT_COLS:
        df[c] = rng.integers(0, 50_000, n).astype(float)
    # Привяжем таргет к PAY_1 — чтобы у моделей было что искать (иначе AUC≈0.5).
    p = 1.0 / (1.0 + np.exp(-df["PAY_1"].astype(float)))
    df[settings.target_col] = (rng.random(n) < p).astype(int)
    return df


@pytest.fixture(scope="module")
def split():
    df = clean(feature_engineering(_synthetic_df()))
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=settings.random_state)


@pytest.mark.unit
def test_cv_score_models_returns_sorted_results(split):
    X_train, _, y_train, _ = split
    pre = build_preprocessor(X_train)
    models = {
        "logreg": Pipeline(
            [
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=500, random_state=settings.random_state)),
            ]
        )
    }
    results = cv_score_models(models, X_train, y_train)
    assert len(results) == 1
    assert 0.0 <= results[0].mean_roc_auc <= 1.0
    assert len(results[0].folds) == settings.cv_splits


@pytest.mark.unit
def test_threshold_by_cost_minimizes_cost():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500) * 0.4 + y_true * 0.4  # положит. класс смещён вправо
    best = best_threshold_by_cost(y_true, y_proba, fn_cost=5.0, fp_cost=1.0)
    assert 0.0 < best.threshold < 1.0
    # Проверим, что cost у выбранного порога не выше, чем у соседних в сетке.
    grid = threshold_grid_to_frame(y_true, y_proba)
    assert best.cost <= grid["cost"].min() * 1.05  # допуск из-за разной сетки


@pytest.mark.unit
def test_threshold_by_f1_returns_valid_point():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    best = best_threshold_by_f1(y_true, y_proba)
    assert 0.05 <= best.threshold <= 0.95
    assert 0.0 <= best.f1 <= 1.0


@pytest.mark.unit
def test_pca_experiment_runs(split):
    X_train, X_test, y_train, y_test = split
    results = run_pca_experiment(X_train, y_train, X_test, y_test, components=(5, 10))
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r.explained_variance <= 1.0
        assert 0.0 <= r.roc_auc <= 1.0


@pytest.mark.unit
def test_explained_variance_curve_is_monotone(split):
    X_train, *_ = split
    df = explained_variance_curve(X_train, max_components=10)
    diffs = np.diff(df["cumulative_explained_variance"].values)
    assert (diffs >= -1e-9).all(), "кумулятивная дисперсия не должна убывать"
    assert df["cumulative_explained_variance"].iloc[-1] <= 1.0 + 1e-9


@pytest.mark.unit
def test_tree_feature_importance(split):
    X_train, _, y_train, _ = split
    pre = build_preprocessor(X_train)
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=50,
                    random_state=settings.random_state,
                    verbose=-1,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    df = tree_feature_importance(pipe, top_k=5)
    assert list(df.columns) == ["feature", "importance"]
    assert len(df) == 5
    assert (df["importance"].diff().dropna() <= 0).all()


@pytest.mark.unit
def test_permutation_importance_runs(split):
    X_train, X_test, y_train, y_test = split
    pre = build_preprocessor(X_train)
    pipe = Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=500, random_state=settings.random_state)),
        ]
    )
    pipe.fit(X_train, y_train)
    df = permutation_feature_importance(pipe, X_test, y_test, n_repeats=3, top_k=5)
    assert len(df) == 5
    assert {"feature", "importance", "std"}.issubset(df.columns)
