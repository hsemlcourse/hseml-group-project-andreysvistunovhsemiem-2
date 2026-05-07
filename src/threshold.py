"""Подбор порога классификации и калибровка вероятностей.

Бизнес-постановка: пропустить дефолт (FN) дороже, чем ложно его предсказать (FP).
Базовая логика — выбираем порог, минимизирующий cost = FN_COST * FN + FP_COST * FP.
По умолчанию FN_COST = 5, FP_COST = 1 — дефолт обходится в 5 раз дороже.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline

from src.config import settings
from src.cv import stratified_cv

DEFAULT_FN_COST = 5.0
DEFAULT_FP_COST = 1.0


@dataclass(frozen=True)
class ThresholdScan:
    threshold: float
    f1: float
    precision: float
    recall: float
    cost: float


def best_threshold_by_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> ThresholdScan:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    pos = float(np.sum(y_true == 1))
    neg = float(np.sum(y_true == 0))
    best: ThresholdScan | None = None
    # precision_recall_curve возвращает на 1 элемент больше, чем thresholds.
    for i, t in enumerate(thresholds):
        recall = float(recalls[i])
        precision = float(precisions[i])
        tp = recall * pos
        fn = pos - tp
        fp = (tp / precision - tp) if precision > 0 else neg
        cost = fn * fn_cost + fp * fp_cost
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        cand = ThresholdScan(
            threshold=float(t),
            f1=float(f1),
            precision=precision,
            recall=recall,
            cost=float(cost),
        )
        if best is None or cand.cost < best.cost:
            best = cand
    assert best is not None
    return best


def best_threshold_by_f1(y_true: np.ndarray, y_proba: np.ndarray) -> ThresholdScan:
    grid = np.linspace(0.05, 0.95, 91)
    best: ThresholdScan | None = None
    for t in grid:
        pred = (y_proba >= t).astype(int)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        denom_p = tp + fp
        denom_r = tp + fn
        precision = tp / denom_p if denom_p > 0 else 0.0
        recall = tp / denom_r if denom_r > 0 else 0.0
        cand = ThresholdScan(
            threshold=float(t),
            f1=f1,
            precision=precision,
            recall=recall,
            cost=float(fn * DEFAULT_FN_COST + fp * DEFAULT_FP_COST),
        )
        if best is None or cand.f1 > best.f1:
            best = cand
    assert best is not None
    return best


def calibrate(pipe: Pipeline, method: str = "isotonic") -> CalibratedClassifierCV:
    """Калибрует вероятности на CV-фолдах. Метод: 'isotonic' (нелинейный) или 'sigmoid' (Platt)."""
    return CalibratedClassifierCV(pipe, method=method, cv=stratified_cv(), n_jobs=-1)


@dataclass(frozen=True)
class CalibrationReport:
    method: str
    brier_before: float
    brier_after: float
    roc_auc_before: float
    roc_auc_after: float


def evaluate_calibration(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    method: str = "isotonic",
) -> CalibrationReport:
    from sklearn.metrics import roc_auc_score

    pipe.fit(X_train, y_train)
    proba_before = pipe.predict_proba(X_test)[:, 1]

    cal = calibrate(_clone_pipeline(pipe), method=method)
    cal.fit(X_train, y_train)
    proba_after = cal.predict_proba(X_test)[:, 1]

    return CalibrationReport(
        method=method,
        brier_before=float(brier_score_loss(y_test, proba_before)),
        brier_after=float(brier_score_loss(y_test, proba_after)),
        roc_auc_before=float(roc_auc_score(y_test, proba_before)),
        roc_auc_after=float(roc_auc_score(y_test, proba_after)),
    )


def _clone_pipeline(pipe: Pipeline) -> Pipeline:
    from sklearn.base import clone

    return clone(pipe)


def threshold_grid_to_frame(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    """Сетка порогов 0.05..0.95 — для графика precision/recall/cost от threshold."""
    grid = np.linspace(0.05, 0.95, 91)
    rows = []
    pos = int((y_true == 1).sum())
    for t in grid:
        pred = (y_proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = pos - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / pos if pos > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        cost = fn * DEFAULT_FN_COST + fp * DEFAULT_FP_COST
        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cost": cost,
            }
        )
    return pd.DataFrame(rows)


# Использование settings, чтобы не было unused-import при дальнейшем расширении.
_ = settings
