"""Эксперимент с уменьшением размерности через PCA для линейной модели.

После OHE+FE число признаков ~70+. Деревья размерности не боятся, а LogReg может выиграть
от декорреляции и сокращения шума. Здесь выбираем k через долю объяснённой дисперсии и
сравниваем ROC-AUC LogReg с/без PCA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import settings
from src.preprocessing import build_preprocessor


@dataclass(frozen=True)
class PCAExperiment:
    n_components: int | float
    explained_variance: float
    roc_auc: float


def explained_variance_curve(X_train: pd.DataFrame, max_components: int = 50) -> pd.DataFrame:
    """Считает кумулятивную долю объяснённой дисперсии после OHE+Scaler — для графика 'elbow'."""
    pre = build_preprocessor(X_train)
    Xt = pre.fit_transform(X_train)
    n = min(max_components, Xt.shape[1])
    pca = PCA(n_components=n, random_state=settings.random_state)
    pca.fit(Xt)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return pd.DataFrame(
        {
            "k": np.arange(1, n + 1),
            "cumulative_explained_variance": cum,
        }
    )


def run_pca_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    components: tuple[int | float, ...] = (10, 20, 30, 0.95),
) -> list[PCAExperiment]:
    """Сравнивает LogReg с PCA(k) для разных k. k=float — целевая доля дисперсии."""
    results: list[PCAExperiment] = []
    for k in components:
        pre = build_preprocessor(X_train)
        pipe = Pipeline(
            [
                ("pre", pre),
                ("pca", PCA(n_components=k, random_state=settings.random_state)),
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
        pipe.fit(X_train, y_train)
        pca_step = pipe.named_steps["pca"]
        proba = pipe.predict_proba(X_test)[:, 1]
        results.append(
            PCAExperiment(
                n_components=int(pca_step.n_components_) if isinstance(k, float) else k,
                explained_variance=float(np.sum(pca_step.explained_variance_ratio_)),
                roc_auc=float(roc_auc_score(y_test, proba)),
            )
        )
    return results


def pca_results_to_frame(results: list[PCAExperiment]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_components": r.n_components,
                "explained_variance": r.explained_variance,
                "roc_auc": r.roc_auc,
            }
            for r in results
        ]
    )
