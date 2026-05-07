"""FastAPI-сервис скоринга.

Запуск:
    uv run uvicorn src.api:app --host 0.0.0.0 --port 8000

Эндпоинты:
    GET  /health         — статус и метаданные модели
    POST /predict        — скор одного клиента
    POST /predict_batch  — скор массива клиентов
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from src.serve import ClientPayload, Scorer, ScoreResponse, get_scorer

app = FastAPI(
    title="Credit Default Scoring API",
    description="ROC-AUC ≈ 0.78, calibrated isotonic XGBoost. Порог решения — cost-минимум.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        scorer = get_scorer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "status": "ok",
        "threshold": scorer.threshold,
        "threshold_meta": scorer.threshold_meta,
    }


@app.post("/predict", response_model=ScoreResponse)
def predict(payload: ClientPayload) -> ScoreResponse:
    scorer: Scorer = get_scorer()
    return scorer.predict(payload)


@app.post("/predict_batch", response_model=list[ScoreResponse])
def predict_batch(payloads: list[ClientPayload]) -> list[ScoreResponse]:
    if not payloads:
        raise HTTPException(status_code=422, detail="Пустой список клиентов")
    scorer: Scorer = get_scorer()
    return scorer.predict_batch(payloads)
