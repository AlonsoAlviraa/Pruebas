"""Unit tests for EODHD fundamentals parse (no network)."""
from __future__ import annotations

import pandas as pd

from paper_live.data.eodhd_client import parse_fundamentals_payload


def test_parse_fundamentals_payload_merges_eps_and_revenue():
    payload = {
        "Earnings": {
            "History": {
                "2023-03-31": {"date": "2023-03-31", "epsActual": 1.0, "reportDate": "2023-04-28"},
                "2023-06-30": {"date": "2023-06-30", "epsActual": 1.1, "reportDate": "2023-07-28"},
                "2023-09-30": {"date": "2023-09-30", "epsActual": 1.2, "reportDate": "2023-10-27"},
                "2023-12-31": {"date": "2023-12-31", "epsActual": 1.3, "reportDate": "2024-01-26"},
                "2024-03-31": {"date": "2024-03-31", "epsActual": 1.5, "reportDate": "2024-04-26"},
            }
        },
        "Financials": {
            "Income_Statement": {
                "quarterly": {
                    "2023-03-31": {"totalRevenue": 100, "netIncome": 10},
                    "2023-06-30": {"totalRevenue": 110, "netIncome": 11},
                    "2023-09-30": {"totalRevenue": 120, "netIncome": 12},
                    "2023-12-31": {"totalRevenue": 130, "netIncome": 13},
                    "2024-03-31": {"totalRevenue": 150, "netIncome": 15},
                }
            }
        },
    }
    df = parse_fundamentals_payload(payload, lag_days=45)
    assert len(df) == 5
    assert set(df.columns) >= {"as_of", "eps", "revenue", "net_income", "available_at", "source"}
    assert df["source"].iloc[0] == "eodhd"
    # reportDate used when present
    last = df.iloc[-1]
    assert float(last["eps"]) == 1.5
    assert float(last["revenue"]) == 150
    assert pd.Timestamp(last["available_at"]) >= pd.Timestamp(last["as_of"])


def test_parse_empty_payload():
    df = parse_fundamentals_payload({})
    assert df.empty
    assert "available_at" in df.columns


def test_lag_fallback_without_report_date():
    payload = {
        "Earnings": {
            "History": {
                "2022-06-30": {"date": "2022-06-30", "epsActual": 2.0},
            }
        }
    }
    df = parse_fundamentals_payload(payload, lag_days=45)
    assert len(df) == 1
    delta = (pd.Timestamp(df.iloc[0]["available_at"]) - pd.Timestamp(df.iloc[0]["as_of"])).days
    assert delta == 45
