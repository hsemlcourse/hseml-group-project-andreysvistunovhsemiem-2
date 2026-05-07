"""Лёгкая обёртка над финальной моделью для FastAPI и Streamlit.

Загружает один pickle + один JSON (порог) и подаёт скоринг по словарю/Pydantic-схеме.
Применяет тот же `clean + feature_engineering`, что и при обучении, чтобы препроцессор
получил знакомую форму данных.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel, Field

from src.config import settings
from src.preprocessing import clean, feature_engineering


class ClientPayload(BaseModel):
    """Один клиент — все поля из исходного датасета (после переименования PAY_0 → PAY_1)."""

    LIMIT_BAL: float = Field(..., ge=0, description="Кредитный лимит, тайваньский доллар")
    SEX: int = Field(..., ge=1, le=2, description="1 = male, 2 = female")
    EDUCATION: int = Field(..., ge=0, le=6)
    MARRIAGE: int = Field(..., ge=0, le=3)
    AGE: int = Field(..., ge=18, le=100)
    PAY_1: int = Field(..., ge=-2, le=8, description="Статус платежа за месяц 1 (-2..8)")
    PAY_2: int = Field(..., ge=-2, le=8)
    PAY_3: int = Field(..., ge=-2, le=8)
    PAY_4: int = Field(..., ge=-2, le=8)
    PAY_5: int = Field(..., ge=-2, le=8)
    PAY_6: int = Field(..., ge=-2, le=8)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)


class ScoreResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0, description="Вероятность дефолта в след. месяце")
    prediction: int = Field(..., description="1 = ожидаем дефолт, 0 = не ожидаем (по cost-порогу)")
    threshold: float = Field(..., description="Порог решения, под который оптимизирована модель")


@dataclass(frozen=True)
class ScorerArtifacts:
    model_path: Path
    threshold_path: Path


DEFAULT_ARTIFACTS = ScorerArtifacts(
    model_path=settings.models_dir / "final_model.joblib",
    threshold_path=settings.models_dir / "threshold.json",
)


class Scorer:
    """Обёртка модель + препроцессинг + порог. Иммутабельна после load()."""

    def __init__(self, model: Any, threshold: float, threshold_meta: dict[str, Any]):
        self._model = model
        self._threshold = float(threshold)
        self._threshold_meta = dict(threshold_meta)

    @classmethod
    def load(cls, artifacts: ScorerArtifacts | None = None) -> "Scorer":
        artifacts = artifacts or DEFAULT_ARTIFACTS
        if not artifacts.model_path.exists():
            raise FileNotFoundError(
                f"{artifacts.model_path} нет — запусти `uv run python -m src.finalize`"
            )
        if not artifacts.threshold_path.exists():
            raise FileNotFoundError(f"{artifacts.threshold_path} нет — запусти `uv run python -m src.finalize`")
        model = joblib.load(artifacts.model_path)
        meta = json.loads(artifacts.threshold_path.read_text())
        return cls(model=model, threshold=meta["threshold"], threshold_meta=meta)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def threshold_meta(self) -> dict[str, Any]:
        return dict(self._threshold_meta)

    def _prepare(self, payload: ClientPayload) -> pd.DataFrame:
        row = pd.DataFrame([payload.model_dump()])
        return feature_engineering(clean(row))

    def predict(self, payload: ClientPayload) -> ScoreResponse:
        df = self._prepare(payload)
        proba = float(self._model.predict_proba(df)[:, 1][0])
        return ScoreResponse(
            probability=proba,
            prediction=int(proba >= self._threshold),
            threshold=self._threshold,
        )

    def predict_batch(self, payloads: list[ClientPayload]) -> list[ScoreResponse]:
        if not payloads:
            return []
        df = pd.concat([self._prepare(p) for p in payloads], ignore_index=True)
        probas = self._model.predict_proba(df)[:, 1]
        return [
            ScoreResponse(
                probability=float(p),
                prediction=int(p >= self._threshold),
                threshold=self._threshold,
            )
            for p in probas
        ]


@lru_cache(maxsize=1)
def get_scorer() -> Scorer:
    """Singleton-обёртка для FastAPI/Streamlit (загрузка один раз на процесс)."""
    return Scorer.load()
