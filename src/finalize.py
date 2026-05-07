"""Финализация модели для деплоя.

Берём лучший XGBoost из CP2 (`models/tuned_xgboost.joblib`), оборачиваем в
isotonic-калибратор (5-fold), фитим на полном train, считаем cost-оптимальный
порог на test и сохраняем артефакт `models/final_model.joblib` + `models/threshold.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from src.config import settings
from src.cv import stratified_cv
from src.modeling import make_train_test
from src.threshold import DEFAULT_FN_COST, DEFAULT_FP_COST, best_threshold_by_cost


def finalize(
    tuned_path: Path | None = None,
    out_path: Path | None = None,
    threshold_path: Path | None = None,
) -> tuple[Path, Path]:
    tuned_path = tuned_path or settings.models_dir / "tuned_xgboost.joblib"
    out_path = out_path or settings.models_dir / "final_model.joblib"
    threshold_path = threshold_path or settings.models_dir / "threshold.json"

    if not tuned_path.exists():
        raise FileNotFoundError(
            f"{tuned_path} нет — сначала запусти `uv run python -m src.tuning`"
        )

    base = joblib.load(tuned_path)
    X_train, X_test, y_train, y_test = make_train_test(use_fe=True)

    calibrated = CalibratedClassifierCV(clone(base), method="isotonic", cv=stratified_cv(), n_jobs=-1)
    calibrated.fit(X_train, y_train)

    proba_test = calibrated.predict_proba(X_test)[:, 1]
    best = best_threshold_by_cost(
        y_test.values, proba_test, fn_cost=DEFAULT_FN_COST, fp_cost=DEFAULT_FP_COST
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, out_path)
    threshold_path.write_text(
        json.dumps(
            {
                "threshold": best.threshold,
                "fn_cost": DEFAULT_FN_COST,
                "fp_cost": DEFAULT_FP_COST,
                "test_precision": best.precision,
                "test_recall": best.recall,
                "test_f1": best.f1,
                "test_cost": best.cost,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return out_path, threshold_path


if __name__ == "__main__":
    model_path, thr_path = finalize()
    print(f"Saved model -> {model_path}")
    print(f"Saved threshold -> {thr_path}")
    print(thr_path.read_text())
