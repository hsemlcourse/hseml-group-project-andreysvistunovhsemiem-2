"""Единый набор метрик для задачи бинарной классификации."""
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationReport:
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> ClassificationReport:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return ClassificationReport(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        f1=float(f1_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
