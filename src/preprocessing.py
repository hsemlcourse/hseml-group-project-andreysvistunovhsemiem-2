"""Очистка, feature engineering и препроцессор для моделирования."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

CATEGORICAL_BASE = ["SEX", "EDUCATION", "MARRIAGE", *PAY_COLS]
NUMERIC_BASE = ["LIMIT_BAL", "AGE", *BILL_COLS, *PAY_AMT_COLS]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Типизация, схлопывание редких категорий, удаление дублей/ID."""
    df = df.copy()
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df = df.drop_duplicates().reset_index(drop=True)

    # EDUCATION: 1=graduate, 2=university, 3=high school, 4=others.
    # Значения 0, 5, 6 не документированы -> объединяем с 4 (others).
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}).astype(int)
    # MARRIAGE: 1=married, 2=single, 3=others. 0 не документирован -> 3.
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3}).astype(int)

    for c in CATEGORICAL_BASE:
        if c in df.columns:
            df[c] = df[c].astype(int)
    for c in NUMERIC_BASE:
        if c in df.columns:
            df[c] = df[c].astype(float)
    if settings.target_col in df.columns:
        df[settings.target_col] = df[settings.target_col].astype(int)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    limit = df["LIMIT_BAL"].replace(0, np.nan)
    for i in range(1, 7):
        df[f"UTIL_{i}"] = df[f"BILL_AMT{i}"] / limit
    df[[f"UTIL_{i}" for i in range(1, 7)]] = df[[f"UTIL_{i}" for i in range(1, 7)]].fillna(0.0)

    # PAY_RATIO_i: доля оплаты по отношению к выписке прошлого месяца
    for i in range(1, 6):
        denom = df[f"BILL_AMT{i + 1}"].where(df[f"BILL_AMT{i + 1}"] > 0, np.nan)
        df[f"PAY_RATIO_{i}"] = (df[f"PAY_AMT{i}"] / denom).fillna(0.0).clip(upper=5.0)

    pay_matrix = df[PAY_COLS]
    df["MAX_DELAY"] = pay_matrix.max(axis=1)
    df["SUM_DELAY"] = pay_matrix.clip(lower=0).sum(axis=1)
    df["NUM_DELAYS"] = (pay_matrix > 0).sum(axis=1)

    df["MEAN_BILL"] = df[BILL_COLS].mean(axis=1)
    df["STD_BILL"] = df[BILL_COLS].std(axis=1).fillna(0.0)
    df["MEAN_PAY_AMT"] = df[PAY_AMT_COLS].mean(axis=1)

    bins = [0, 30, 40, 50, 120]
    labels = ["20-30", "30-40", "40-50", "50+"]
    df["AGE_BUCKET"] = pd.cut(df["AGE"], bins=bins, labels=labels, include_lowest=True).astype(str)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[settings.target_col]
    X = df.drop(columns=[settings.target_col])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Собирает ColumnTransformer (OHE для категорий, StandardScaler для чисел)."""
    categorical = [c for c in X.columns if c in CATEGORICAL_BASE or c == "AGE_BUCKET"]
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )
