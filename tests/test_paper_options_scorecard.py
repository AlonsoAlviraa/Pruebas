"""Unit tests for options matrix scorecard (promote/watch/kill)."""
from __future__ import annotations

import json
from pathlib import Path

from paper_live.options.scorecard import (
    decide_verdict,
    score_matrix,
    scorecard_to_markdown,
    write_scorecard,
    StrategyScore,
    WindowMetrics,
)


def _fixture_summary() -> dict:
    """Synthetic multi-window summary with clear kill / promote / cash rows."""
    def strat(
        sid,
        kind,
        und,
        ret,
        dd,
        *,
        defined=False,
        vs_spy=-0.05,
        opens=5,
        hard_kill=False,
        cvar=-0.02,
    ):
        return {
            "strategy_id": sid,
            "label": sid,
            "kind": kind,
            "underlying": und,
            "total_return": ret,
            "max_dd": dd,
            "cvar_5pct": cvar,
            "vs_spy_bh": vs_spy,
            "n_opens": opens,
            "n_tp": 2,
            "n_sl": 1,
            "n_time_exit": 0,
            "n_dte_rolls": 0,
            "hard_kill": hard_kill,
            "defined_risk": defined,
        }

    cash = strat("OPT_TA12_cash", "cash", "SPY", 0.0, 0.0, vs_spy=-0.20, opens=0)
    good_pcs = strat(
        "OPT_GOOD_pcs",
        "put_credit_spread",
        "SPY",
        0.04,
        -0.05,
        defined=True,
        vs_spy=-0.16,
        opens=8,
    )
    bad_csp = strat(
        "OPT_BAD_csp",
        "cash_secured_put",
        "SPY",
        -0.12,
        -0.30,
        defined=False,
        vs_spy=-0.35,
        opens=10,
    )
    income = strat(
        "OPT_INCOME_cc",
        "covered_call",
        "SPY",
        0.08,
        -0.10,
        defined=False,
        vs_spy=-0.15,
        opens=6,
    )
    name_pcs = strat(
        "OPT_NAME_AAPL_pcs",
        "put_credit_spread",
        "AAPL",
        0.03,
        -0.06,
        defined=True,
        vs_spy=-0.10,
        opens=3,
    )

    def window(name, ret_scale=1.0):
        def scale(s, scale_r):
            o = dict(s)
            if o["kind"] != "cash":
                o["total_return"] = float(o["total_return"]) * scale_r
            return o

        return {
            "name": name,
            "window": {"start": "2023-01-01", "end": "2023-12-31"},
            "strategies": [
                scale(cash, 1.0),
                scale(good_pcs, ret_scale),
                scale(bad_csp, ret_scale),
                scale(income, ret_scale),
                scale(name_pcs, ret_scale),
            ],
            "benchmarks": {"spy_bh": 0.20},
        }

    return {
        "as_of": "2026-07-22",
        "zoo": "fixture",
        "names_zoo": "fixture_names",
        "windows": [
            window("2022_bear", ret_scale=0.5),
            window("2023", ret_scale=1.0),
            window("2024", ret_scale=1.0),
            window("2025_study", ret_scale=1.0),
        ],
        "stress": {
            "name": "stress_primary",
            "strategies": [
                {**cash, "total_return": 0.0, "max_dd": 0.0},
                {**good_pcs, "total_return": -0.04, "max_dd": -0.08},
                {**bad_csp, "total_return": -0.25, "max_dd": -0.35},
                {**income, "total_return": -0.05, "max_dd": -0.12},
                {**name_pcs, "total_return": -0.03, "max_dd": -0.07},
            ],
        },
    }


def test_score_matrix_verdicts():
    summary = _fixture_summary()
    payload = score_matrix(summary)
    by = {s["strategy_id"]: s for s in payload["strategies"]}

    assert by["OPT_TA12_cash"]["verdict"] == "HOLD"
    assert by["OPT_BAD_csp"]["verdict"] == "KILL"
    # defined-risk that beats cash in calendar windows → PROMOTE_RESEARCH
    assert by["OPT_GOOD_pcs"]["verdict"] == "PROMOTE_RESEARCH"
    assert by["OPT_NAME_AAPL_pcs"]["segment"] == "single_name"
    assert "PROMOTE_RESEARCH" in payload["counts"]
    assert payload["counts"]["KILL"] >= 1


def test_scorecard_markdown_and_write(tmp_path: Path):
    summary = _fixture_summary()
    sp = tmp_path / "summary.json"
    sp.write_text(json.dumps(summary), encoding="utf-8")
    out_md = tmp_path / "SCORECARD.md"
    payload = write_scorecard(sp, out_md=out_md)
    assert out_md.is_file()
    assert out_md.with_suffix(".json").is_file()
    md = out_md.read_text(encoding="utf-8")
    assert "KILL" in md
    assert "PROMOTE_RESEARCH" in md
    assert payload["counts"]["KILL"] >= 1


def test_decide_verdict_max_dd_kill():
    sc = StrategyScore(
        strategy_id="x",
        kind="cash_secured_put",
        underlying="SPY",
        worst_max_dd=-0.40,
        windows={
            "2023": WindowMetrics(name="2023", total_return=-0.01, max_dd=-0.40),
        },
        cash_return_by_window={"2023": 0.0, "stress_primary": 0.0},
        stress_return=-0.10,
        stress_max_dd=-0.40,
    )
    out = decide_verdict(sc)
    assert out.verdict == "KILL"


def test_scorecard_to_markdown_smoke():
    payload = score_matrix(_fixture_summary())
    md = scorecard_to_markdown(payload)
    assert "SCORECARD" in md
    assert "Decision table" in md
