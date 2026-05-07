"""Интерпретируемость финальной модели: feature importance + permutation importance.

SHAP-плот добавлять не стал, потому что:
1) у sklearn-пайплайна с OHE имена признаков уже выводятся через `get_feature_names_out`,
2) tree-importance + permutation importance дают понятный картинку: «что в среднем двигает модель».
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.config import settings


@dataclass(frozen=True)
class FeatureImportance:
    feature: str
    importance: float


def _feature_names(pipe: Pipeline) -> list[str]:
    pre = pipe.named_steps["pre"]
    return list(pre.get_feature_names_out())


def tree_feature_importance(pipe: Pipeline, top_k: int = 25) -> pd.DataFrame:
    """Вытаскивает feature_importances_ из tree-моделей (RF, XGB, LGBM)."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise ValueError(f"{type(clf).__name__} не имеет feature_importances_")
    importances = np.asarray(clf.feature_importances_, dtype=float)
    names = _feature_names(pipe)
    if len(names) != len(importances):
        raise ValueError(f"Несоответствие длин: {len(names)} имён vs {len(importances)} важностей")
    df = pd.DataFrame({"feature": names, "importance": importances})
    return df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)


def permutation_feature_importance(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    top_k: int = 25,
) -> pd.DataFrame:
    """Permutation importance — считается на исходных колонках X (до препроцессора)."""
    result = permutation_importance(
        pipe,
        X,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=settings.random_state,
        n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)
