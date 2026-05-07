"""Streamlit-UI для скоринга одного клиента.

Запуск:
    uv run streamlit run src/ui.py

Можно работать в двух режимах:
- "Локально (без API)" — модель грузится прямо в процесс Streamlit (быстро, без сети).
- "Через FastAPI" — отправляем POST /predict на указанный URL.
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

from src.serve import ClientPayload, get_scorer

EDUCATION_LABELS = {
    1: "Graduate school",
    2: "University",
    3: "High school",
    4: "Others",
    0: "Unknown (→ Others)",
    5: "Unknown (→ Others)",
    6: "Unknown (→ Others)",
}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Others", 0: "Unknown (→ Others)"}
SEX_LABELS = {1: "Male", 2: "Female"}
PAY_LABELS = {
    -2: "−2 — без счёта",
    -1: "−1 — оплачено вовремя",
    0: "0 — оплата вовремя (с переносом)",
    1: "1 — задержка 1 мес.",
    2: "2 — задержка 2 мес.",
    3: "3 — задержка 3 мес.",
    4: "4 — задержка 4 мес.",
    5: "5+ — длинная просрочка",
}


def _form() -> ClientPayload:
    st.subheader("Демография")
    col1, col2, col3 = st.columns(3)
    with col1:
        sex = st.selectbox("Пол (SEX)", options=list(SEX_LABELS.keys()), format_func=SEX_LABELS.get, index=1)
        age = st.number_input("AGE", min_value=18, max_value=100, value=35, step=1)
    with col2:
        education = st.selectbox(
            "Образование (EDUCATION)",
            options=[1, 2, 3, 4],
            format_func=EDUCATION_LABELS.get,
            index=1,
        )
    with col3:
        marriage = st.selectbox(
            "Семейное положение (MARRIAGE)",
            options=[1, 2, 3],
            format_func=MARRIAGE_LABELS.get,
            index=1,
        )

    st.subheader("Кредит")
    limit_bal = st.number_input("LIMIT_BAL (NT$)", min_value=0.0, value=120000.0, step=1000.0)

    st.subheader("История платежей (PAY_1..6, статус по месяцам)")
    pay_cols = st.columns(6)
    pay_values: dict[str, int] = {}
    for i, col in enumerate(pay_cols, start=1):
        with col:
            pay_values[f"PAY_{i}"] = st.selectbox(
                f"PAY_{i}",
                options=list(PAY_LABELS.keys()),
                format_func=PAY_LABELS.get,
                index=2,
                key=f"pay_{i}",
            )

    st.subheader("Суммы выписок (BILL_AMT 1..6)")
    bill_cols = st.columns(6)
    bills: dict[str, float] = {}
    defaults_bill = [3913.0, 3102.0, 689.0, 0.0, 0.0, 0.0]
    for i, col in enumerate(bill_cols, start=1):
        with col:
            bills[f"BILL_AMT{i}"] = float(
                st.number_input(
                    f"BILL_AMT{i}",
                    value=defaults_bill[i - 1],
                    step=100.0,
                    key=f"bill_{i}",
                )
            )

    st.subheader("Суммы платежей (PAY_AMT 1..6)")
    payamt_cols = st.columns(6)
    payamts: dict[str, float] = {}
    defaults_payamt = [0.0, 689.0, 0.0, 0.0, 0.0, 0.0]
    for i, col in enumerate(payamt_cols, start=1):
        with col:
            payamts[f"PAY_AMT{i}"] = float(
                st.number_input(
                    f"PAY_AMT{i}",
                    min_value=0.0,
                    value=defaults_payamt[i - 1],
                    step=100.0,
                    key=f"payamt_{i}",
                )
            )

    return ClientPayload(
        LIMIT_BAL=float(limit_bal),
        SEX=int(sex),
        EDUCATION=int(education),
        MARRIAGE=int(marriage),
        AGE=int(age),
        **pay_values,
        **bills,
        **payamts,
    )


def _score_local(payload: ClientPayload) -> dict:
    scorer = get_scorer()
    return scorer.predict(payload).model_dump()


def _score_api(payload: ClientPayload, base_url: str) -> dict:
    resp = httpx.post(f"{base_url.rstrip('/')}/predict", json=payload.model_dump(), timeout=10.0)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    st.set_page_config(page_title="Credit Default Scoring", page_icon="💳", layout="wide")
    st.title("💳 Credit Default Scoring")
    st.caption(
        "Бинарный классификатор: вероятность дефолта по платежу в следующем месяце. "
        "Модель — XGBoost (tuned + isotonic calibration), порог по cost-минимуму (FN=5, FP=1)."
    )

    default_url = os.environ.get("API_URL", "http://localhost:8000")
    mode = st.sidebar.radio("Режим скоринга", ["Локально (без API)", "Через FastAPI"])
    api_url = st.sidebar.text_input("API URL", value=default_url) if mode == "Через FastAPI" else ""

    payload = _form()
    if st.button("Оценить риск дефолта", type="primary"):
        try:
            result = _score_local(payload) if mode == "Локально (без API)" else _score_api(payload, api_url)
        except FileNotFoundError as e:
            st.error(f"{e}")
            return
        except httpx.HTTPError as e:
            st.error(f"Ошибка вызова API: {e}")
            return

        prob = result["probability"]
        pred = result["prediction"]
        threshold = result["threshold"]

        st.divider()
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.metric("Вероятность дефолта", f"{prob:.1%}")
            st.metric("Порог решения", f"{threshold:.2f}")
        with col_right:
            if pred == 1:
                st.error("⚠️ Риск дефолта — отправить на ручной разбор / снизить лимит")
            else:
                st.success("✅ Низкий риск — стандартное обслуживание")

        st.progress(prob)


if __name__ == "__main__":
    main()
