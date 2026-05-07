"""Тесты FastAPI-эндпоинтов через TestClient."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.serve import get_scorer


@pytest.fixture(scope="module")
def client() -> TestClient:
    # Прогреваем scorer на уровне модуля, чтобы упасть быстро, если артефакта нет.
    try:
        get_scorer()
    except FileNotFoundError as e:
        pytest.skip(f"Финальная модель не сохранена: {e}")
    return TestClient(app)


def _sample_payload(default_risk: str = "low") -> dict:
    """high — задержки везде, low — оплаты вовремя."""
    pay = 2 if default_risk == "high" else -1
    base = {
        "LIMIT_BAL": 50000.0,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_1": pay,
        "PAY_2": pay,
        "PAY_3": pay,
        "PAY_4": pay,
        "PAY_5": pay,
        "PAY_6": pay,
    }
    for i in range(1, 7):
        base[f"BILL_AMT{i}"] = 10000.0
        base[f"PAY_AMT{i}"] = 0.0 if default_risk == "high" else 5000.0
    return base


@pytest.mark.integration
def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert 0.0 <= body["threshold"] <= 1.0


@pytest.mark.integration
def test_predict_returns_probability(client: TestClient):
    r = client.post("/predict", json=_sample_payload("low"))
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["prediction"] == int(body["probability"] >= body["threshold"])


@pytest.mark.integration
def test_predict_high_risk_higher_than_low_risk(client: TestClient):
    low = client.post("/predict", json=_sample_payload("low")).json()
    high = client.post("/predict", json=_sample_payload("high")).json()
    assert high["probability"] > low["probability"], "Профиль с задержками должен иметь более высокий риск"


@pytest.mark.integration
def test_predict_validates_input(client: TestClient):
    bad = _sample_payload("low")
    bad["AGE"] = -5  # вне диапазона
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


@pytest.mark.integration
def test_predict_batch(client: TestClient):
    payloads = [_sample_payload("low"), _sample_payload("high")]
    r = client.post("/predict_batch", json=payloads)
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[1]["probability"] > body[0]["probability"]


@pytest.mark.integration
def test_predict_batch_rejects_empty(client: TestClient):
    r = client.post("/predict_batch", json=[])
    assert r.status_code == 422
