"""Cross-validation: единый StratifiedKFold для честного сравнения моделей."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import settings


@dataclass(frozen=True)
class CVResult:
    name: str
    mean_roc_auc: float
    std_roc_auc: float
    folds: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cv_roc_auc_mean": self.mean_roc_auc,
            "cv_roc_auc_std": self.std_roc_auc,
        }


def stratified_cv() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=settings.cv_splits,
        shuffle=True,
        random_state=settings.random_state,
    )


def cv_score_models(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
) -> list[CVResult]:
    """Прогоняет все модели через единый StratifiedKFold по ROC-AUC."""
    cv = stratified_cv()
    results: list[CVResult] = []
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        results.append(
            CVResult(
                name=name,
                mean_roc_auc=float(np.mean(scores)),
                std_roc_auc=float(np.std(scores)),
                folds=tuple(float(s) for s in scores),
            )
        )
    return sorted(results, key=lambda r: r.mean_roc_auc, reverse=True)


def cv_results_to_frame(results: list[CVResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])
