"""Unit tests for SEC companyfacts parse (offline synthetic)."""
from __future__ import annotations

import pandas as pd

from trad_research.sec_fundamentals import parse_companyfacts_to_quarterly


def _fact_rows(vals):
    return {
        "label": "x",
        "units": {
            "USD": vals,
        },
    }


def test_parse_quarterly_with_filed_pit():
    payload = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "label": "EPS",
                    "units": {
                        "USD/shares": [
                            {
                                "end": "2020-03-31",
                                "val": 1.0,
                                "form": "10-Q",
                                "fp": "Q2",
                                "filed": "2020-05-01",
                                "frame": "CY2020Q1",
                                "start": "2020-01-01",
                            },
                            {
                                "end": "2020-06-30",
                                "val": 1.2,
                                "form": "10-Q",
                                "fp": "Q3",
                                "filed": "2020-08-01",
                                "frame": "CY2020Q2",
                                "start": "2020-04-01",
                            },
                            {
                                "end": "2021-03-31",
                                "val": 1.5,
                                "form": "10-Q",
                                "fp": "Q2",
                                "filed": "2021-05-02",
                                "frame": "CY2021Q1",
                                "start": "2021-01-01",
                            },
                        ]
                    },
                },
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "label": "Rev",
                    "units": {
                        "USD": [
                            {
                                "end": "2020-03-31",
                                "val": 100,
                                "form": "10-Q",
                                "fp": "Q2",
                                "filed": "2020-05-01",
                                "frame": "CY2020Q1",
                                "start": "2020-01-01",
                            },
                            {
                                "end": "2020-06-30",
                                "val": 110,
                                "form": "10-Q",
                                "fp": "Q3",
                                "filed": "2020-08-01",
                                "frame": "CY2020Q2",
                                "start": "2020-04-01",
                            },
                            {
                                "end": "2021-03-31",
                                "val": 150,
                                "form": "10-Q",
                                "fp": "Q2",
                                "filed": "2021-05-02",
                                "frame": "CY2021Q1",
                                "start": "2021-01-01",
                            },
                        ]
                    },
                },
                "NetIncomeLoss": {
                    "label": "NI",
                    "units": {
                        "USD": [
                            {
                                "end": "2020-03-31",
                                "val": 10,
                                "form": "10-Q",
                                "fp": "Q2",
                                "filed": "2020-05-01",
                                "frame": "CY2020Q1",
                                "start": "2020-01-01",
                            },
                            {
                                "end": "2020-06-30",
                                "val": 12,
                                "form": "10-Q",
                                "fp": "Q3",
                                "filed": "2020-08-01",
                                "frame": "CY2020Q2",
                                "start": "2020-04-01",
                            },
                            {
                                "end": "2021-03-31",
                                "val": 20,
                                "form": "10-Q",
                                "fp": "Q2",
                                "filed": "2021-05-02",
                                "frame": "CY2021Q1",
                                "start": "2021-01-01",
                            },
                        ]
                    },
                },
            }
        }
    }
    df = parse_companyfacts_to_quarterly(payload)
    assert len(df) == 3
    assert df.iloc[0]["eps"] == 1.0
    assert df.iloc[-1]["revenue"] == 150
    # available_at is filed, not period end
    assert pd.Timestamp(df.iloc[0]["available_at"]) == pd.Timestamp("2020-05-01", tz="UTC")
    assert pd.Timestamp(df.iloc[0]["as_of"]) == pd.Timestamp("2020-03-31", tz="UTC")


def test_prefer_earlier_filed_for_restatement():
    payload = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {
                                "end": "2019-06-30",
                                "val": 1.0,
                                "form": "10-Q",
                                "fp": "Q2",
                                "filed": "2019-08-01",
                                "frame": "CY2019Q2",
                                "start": "2019-04-01",
                            },
                            {
                                "end": "2019-06-30",
                                "val": 0.9,  # restated later
                                "form": "10-Q/A",
                                "fp": "Q2",
                                "filed": "2019-12-01",
                                "frame": "CY2019Q2",
                                "start": "2019-04-01",
                            },
                        ]
                    }
                },
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "end": "2019-06-30",
                                "val": 100,
                                "form": "10-Q",
                                "fp": "Q2",
                                "filed": "2019-08-01",
                                "frame": "CY2019Q2",
                                "start": "2019-04-01",
                            },
                        ]
                    }
                },
            }
        }
    }
    df = parse_companyfacts_to_quarterly(payload)
    assert len(df) == 1
    assert float(df.iloc[0]["eps"]) == 1.0  # first-filed


def test_empty_payload():
    df = parse_companyfacts_to_quarterly({})
    assert df.empty
