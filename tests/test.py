"""Минимальные тесты пайплайна: анти-leakage и корректность FE."""
import numpy as np
import pandas as pd
import pytest

from src.config import settings
from src.preprocessing import (
    BILL_COLS,
    PAY_AMT_COLS,
    PAY_COLS,
    build_preprocessor,
    clean,
    feature_engineering,
    split_features_target,
)


def _synthetic_df(n: int = 200, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(settings.random_state if seed is None else seed)
    df = pd.DataFrame(
        {
            "ID": np.arange(n),
            "LIMIT_BAL": rng.integers(10_000, 500_000, n).astype(float),
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": rng.integers(0, 7, n),  # включая недокументированные 0/5/6
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
    df[settings.target_col] = rng.integers(0, 2, n)
    return df


def test_clean_collapses_undocumented_categories():
    df = _synthetic_df()
    cleaned = clean(df)
    assert "ID" not in cleaned.columns
    assert set(cleaned["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert 0 not in cleaned["MARRIAGE"].unique()


def test_feature_engineering_adds_expected_columns():
    df = clean(_synthetic_df())
    fe = feature_engineering(df)
    expected = {
        *(f"UTIL_{i}" for i in range(1, 7)),
        *(f"PAY_RATIO_{i}" for i in range(1, 6)),
        "MAX_DELAY",
        "SUM_DELAY",
        "NUM_DELAYS",
        "MEAN_BILL",
        "STD_BILL",
        "MEAN_PAY_AMT",
        "AGE_BUCKET",
    }
    assert expected.issubset(fe.columns)
    assert len(fe) == len(df)
    assert fe.isna().sum().sum() == 0


def test_preprocessor_no_leak_fit_on_train_only():
    """Препроцессор фиттится только на train; transform на test не падает и не
    подсматривает статистики test-выборки (скейлер использует mean/std из train)."""
    df = clean(_synthetic_df(n=300))
    X, _ = split_features_target(df)
    X_train = X.iloc[:200].copy()
    X_test = X.iloc[200:].copy()

    pre = build_preprocessor(X_train)
    pre.fit(X_train)
    Xt_train = pre.transform(X_train)
    Xt_test = pre.transform(X_test)
    assert Xt_train.shape[1] == Xt_test.shape[1]
    assert Xt_train.shape[0] == len(X_train)
    assert Xt_test.shape[0] == len(X_test)

    scaler = pre.named_transformers_["num"]
    # mean, использованный на transform, равен mean из train — не из всего X
    np.testing.assert_allclose(scaler.mean_, X_train[scaler.feature_names_in_].mean().values, rtol=1e-6)


def test_run_experiments_smoke(monkeypatch, tmp_path):
    """Smoke: run_experiments обучает 5 моделей на синтетике и возвращает отчёты."""
    from src import modeling

    df = clean(feature_engineering(_synthetic_df(n=400)))

    def fake_make_train_test(use_fe: bool = False):
        from sklearn.model_selection import train_test_split

        X, y = split_features_target(df)
        return train_test_split(X, y, test_size=0.25, stratify=y, random_state=settings.random_state)

    monkeypatch.setattr(modeling, "make_train_test", fake_make_train_test)
    monkeypatch.setattr(settings, "models_dir", tmp_path)

    reports = modeling.run_experiments(save=False)
    assert set(reports) == {"knn_k25", "logreg_fe", "random_forest", "xgboost", "lightgbm"}
    for rep in reports.values():
        assert 0.0 <= rep.roc_auc <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
