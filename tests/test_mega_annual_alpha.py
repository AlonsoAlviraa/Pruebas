"""Unit tests for mega annual alpha evaluation (synthetic / no network)."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from paper_live.cloud.mega_annual_alpha import (
    DEFAULT_WINDOWS,
    EXCESS_MARGIN,
    YearEval,
    beat_all_indices_by_3pp,
    best_index_return,
    build_tier_tables,
    build_year_eval,
    clamp_calendar_window,
    excess_over_best_index,
    filter_winners,
    load_mega_zoo,
    max_drawdown_from_equity,
    merge_strategy_lists,
    rank_by_mean_excess,
    summarize_strategy_years,
    write_report_pack,
)


def test_beat_all_indices_by_3pp_basic():
    ben = {"spy_bh": 0.10, "qqq_bh": 0.20, "iwm_bh": 0.05}
    # best = 0.20; need >= 0.23
    assert beat_all_indices_by_3pp(0.23, ben) is True
    assert beat_all_indices_by_3pp(0.229, ben) is False
    assert beat_all_indices_by_3pp(0.25, ben) is True
    assert EXCESS_MARGIN == 0.03


def test_beat_ignores_missing_indices():
    ben = {"spy_bh": 0.10, "qqq_bh": None, "iwm_bh": None}
    assert best_index_return(ben) == pytest.approx(0.10)
    assert beat_all_indices_by_3pp(0.13, ben) is True
    assert beat_all_indices_by_3pp(0.12, ben) is False


def test_beat_false_when_no_benchmarks():
    assert beat_all_indices_by_3pp(0.50, {}) is False
    assert beat_all_indices_by_3pp(0.50, {"spy_bh": None}) is False
    assert best_index_return({}) is None
    assert excess_over_best_index(0.1, {}) is None


def test_excess_over_best():
    ben = {"spy_bh": 0.05, "qqq_bh": 0.15, "iwm_bh": 0.08}
    assert excess_over_best_index(0.20, ben) == pytest.approx(0.05)
    assert excess_over_best_index(0.10, ben) == pytest.approx(-0.05)


def test_clamp_calendar_window():
    days = [date(2022, 1, 3) + timedelta(days=i) for i in range(0, 400, 1)]
    # only weekdays-ish synthetic list is fine
    s, e, clamped = clamp_calendar_window(days, date(2022, 1, 1), date(2022, 12, 31))
    assert s == days[0]
    assert e <= date(2022, 12, 31)
    assert clamped is True  # start before first day

    s2, e2, c2 = clamp_calendar_window(days, days[10], days[20])
    assert s2 == days[10]
    assert e2 == days[20]
    assert c2 is False


def test_clamp_empty_raises():
    with pytest.raises(ValueError):
        clamp_calendar_window([], date(2022, 1, 1), date(2022, 12, 31))


def test_max_drawdown_from_equity():
    eq = [100.0, 110.0, 90.0, 95.0]
    mdd = max_drawdown_from_equity(eq)
    assert mdd == pytest.approx(90 / 110 - 1.0)


def test_build_year_eval_flags():
    ye = build_year_eval(
        strategy_id="X",
        year="2023",
        total_return=0.30,
        benchmarks={"spy_bh": 0.20, "qqq_bh": 0.25, "iwm_bh": 0.10},
        max_dd=-0.08,
        n_opens=5,
    )
    assert ye.beat_all_indices_by_3pp is True  # 0.30 >= 0.25+0.03
    assert ye.excess_vs_best == pytest.approx(0.05)
    assert ye.vs_spy == pytest.approx(0.10)
    assert ye.vs_qqq == pytest.approx(0.05)
    d = ye.to_dict()
    assert d["capital_label"] == "VIRTUAL"
    assert d["beat_all_indices_by_3pp"] is True


def test_summarize_and_tiers():
    rows = []
    # strategy A beats all 4 years
    for y, ret in [("2022", 0.05), ("2023", 0.40), ("2024", 0.35), ("2025_study", 0.20)]:
        # benchmarks: 2022 best -0.10, others 0.25/0.30/0.15
        if y == "2022":
            ben = {"spy_bh": -0.18, "qqq_bh": -0.32, "iwm_bh": -0.20}
        elif y == "2023":
            ben = {"spy_bh": 0.24, "qqq_bh": 0.54, "iwm_bh": 0.16}
            ret = 0.60  # beats QQQ 0.54 by >3pp
        elif y == "2024":
            ben = {"spy_bh": 0.24, "qqq_bh": 0.25, "iwm_bh": 0.11}
            ret = 0.30
        else:
            ben = {"spy_bh": 0.05, "qqq_bh": 0.08, "iwm_bh": 0.02}
            ret = 0.15
        rows.append(
            build_year_eval(
                strategy_id="A_WIN",
                year=y,
                total_return=ret,
                benchmarks=ben,
                n_opens=3,
            )
        )
    # strategy B beats exactly 2 years (fails 2023 and 2025)
    for y, ret, best in [
        ("2022", 0.0, -0.18),   # pass: 0 >= -0.15
        ("2023", 0.10, 0.54),   # fail
        ("2024", 0.40, 0.25),   # pass
        ("2025_study", 0.05, 0.08),  # fail: 0.05 < 0.11
    ]:
        ben = {"spy_bh": best, "qqq_bh": best, "iwm_bh": best}
        rows.append(
            build_year_eval(
                strategy_id="B_HALF",
                year=y,
                total_return=ret,
                benchmarks=ben,
                n_opens=2,
            )
        )

    sa = summarize_strategy_years(rows, strategy_id="A_WIN", n_study_years=4)
    sb = summarize_strategy_years(rows, strategy_id="B_HALF", n_study_years=4)
    assert sa.years_passed == 4
    assert sa.tier == "4/4"
    assert sb.years_passed == 2
    assert sb.tier == "2/4"

    summaries = [sa, sb]
    tiers = build_tier_tables(summaries, n_study_years=4)
    assert len(tiers["4/4"]) == 1
    assert tiers["4/4"][0].strategy_id == "A_WIN"
    assert any(s.strategy_id == "B_HALF" for s in tiers["2/4"])

    strict = filter_winners(summaries, min_years_passed=4, n_study_years=4)
    assert len(strict) == 1
    half = filter_winners(summaries, min_years_passed=2, n_study_years=4)
    assert len(half) == 2


def test_rank_by_mean_excess():
    s1 = summarize_strategy_years(
        [
            build_year_eval(
                strategy_id="low",
                year="2023",
                total_return=0.10,
                benchmarks={"spy_bh": 0.05, "qqq_bh": 0.05, "iwm_bh": 0.05},
            )
        ],
        strategy_id="low",
        n_study_years=1,
    )
    s2 = summarize_strategy_years(
        [
            build_year_eval(
                strategy_id="high",
                year="2023",
                total_return=0.30,
                benchmarks={"spy_bh": 0.05, "qqq_bh": 0.05, "iwm_bh": 0.05},
            )
        ],
        strategy_id="high",
        n_study_years=1,
    )
    ranked = rank_by_mean_excess([s1, s2])
    assert ranked[0].strategy_id == "high"


def test_hard_kill_filter():
    ye = build_year_eval(
        strategy_id="K",
        year="2023",
        total_return=0.50,
        benchmarks={"spy_bh": 0.10, "qqq_bh": 0.10, "iwm_bh": 0.10},
        hard_kill=True,
        n_opens=1,
    )
    s = summarize_strategy_years([ye], strategy_id="K", n_study_years=1)
    assert s.hard_kill_years == 1
    filtered = filter_winners(
        [s], min_years_passed=1, n_study_years=1, allow_hard_kill=False
    )
    assert filtered == []
    allowed = filter_winners(
        [s], min_years_passed=1, n_study_years=1, allow_hard_kill=True
    )
    assert len(allowed) == 1


def test_min_opens_filter():
    ye = build_year_eval(
        strategy_id="quiet",
        year="2023",
        total_return=0.50,
        benchmarks={"spy_bh": 0.10, "qqq_bh": 0.10, "iwm_bh": 0.10},
        n_opens=0,
        n_closed_trades=0,
    )
    s = summarize_strategy_years([ye], strategy_id="quiet", n_study_years=1)
    assert (
        filter_winners([s], min_years_passed=1, n_study_years=1, min_opens=1) == []
    )
    assert (
        len(filter_winners([s], min_years_passed=1, n_study_years=1, min_opens=0))
        == 1
    )


def test_merge_strategy_lists_and_zoo():
    zoo = load_mega_zoo()
    assert "equity_strategies" in zoo
    assert "options_strategies" in zoo
    eq = zoo["equity_strategies"]
    opt = zoo["options_strategies"]
    assert len(eq) >= 20
    assert len(opt) >= 5
    ids = [s["id"] for s in eq]
    assert len(ids) == len(set(ids))
    merged = merge_strategy_lists(eq, eq, max_strategies=5)
    assert len(merged) == 5


def test_default_windows():
    names = [w[0] for w in DEFAULT_WINDOWS]
    assert names == ["2022", "2023", "2024", "2025_study"]


def test_write_report_pack(tmp_path: Path):
    rows = [
        build_year_eval(
            strategy_id="S1",
            year="2023",
            total_return=0.40,
            benchmarks={"spy_bh": 0.20, "qqq_bh": 0.25, "iwm_bh": 0.10},
            n_opens=4,
        ),
        build_year_eval(
            strategy_id="S1",
            year="2024",
            total_return=0.10,
            benchmarks={"spy_bh": 0.20, "qqq_bh": 0.25, "iwm_bh": 0.10},
            n_opens=2,
        ),
    ]
    s = summarize_strategy_years(rows, strategy_id="S1", n_study_years=2)
    tiers = build_tier_tables([s], n_study_years=2)
    paths = write_report_pack(
        out_root=tmp_path,
        year_evals=rows,
        summaries=[s],
        tiers=tiers,
        windows_meta=[
            {
                "name": "2023",
                "start": "2023-01-03",
                "end": "2023-12-29",
                "spy_bh": 0.20,
                "qqq_bh": 0.25,
                "iwm_bh": 0.10,
                "clamped": False,
            },
            {
                "name": "2024",
                "start": "2024-01-02",
                "end": "2024-12-31",
                "spy_bh": 0.20,
                "qqq_bh": 0.25,
                "iwm_bh": 0.10,
                "clamped": False,
            },
        ],
        data_sources={"SPY": "synthetic"},
        capital0=100_000.0,
        n_study_years=2,
    )
    assert paths["summary"].is_file()
    assert paths["winners"].is_file()
    text = paths["summary"].read_text(encoding="utf-8")
    assert "Zero strategies" in text or "Tier" in text
    assert "VIRTUAL" in text
    winners = paths["winners"]
    import json

    w = json.loads(winners.read_text(encoding="utf-8"))
    assert "strict_winners" in w
    assert "tiers" in w
    assert (tmp_path / "latest" / "by_year" / "2023.md").is_file()


def test_synthetic_smoke_runner(tmp_path: Path):
    """End-to-end synthetic smoke: tiny zoo subset, short windows on synthetic feed."""
    from paper_live.cloud.mega_annual_alpha import run_mega_annual_alpha_study

    # Short windows keep replay cheap under CI.
    windows = [
        ("Y1", "2020-03-01", "2020-04-30"),
        ("Y2", "2020-05-01", "2020-06-30"),
    ]
    result = run_mega_annual_alpha_study(
        out_root=tmp_path / "mega",
        force_synthetic=True,
        lookback_days=400,
        max_equity=2,
        max_options=1,
        windows=windows,
        min_real_tickers=0,
        skip_options=False,
    )
    assert result.n_equity == 2
    assert result.n_options == 1
    assert len(result.year_evals) == (2 + 1) * 2
    assert (tmp_path / "mega" / "latest" / "SUMMARY.md").is_file()
    assert result.force_synthetic is True
    assert all(str(v).startswith("synthetic") for v in result.data_sources.values())
