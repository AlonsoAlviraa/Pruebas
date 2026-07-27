"""Tests for offline mega annual rescore (no network)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.rescore_mega_annual import aggregate, decide_verdicts, passes


def test_passes_spy_and_best():
    e = {
        "total_return": 0.30,
        "spy_bh": 0.20,
        "qqq_bh": 0.25,
        "iwm_bh": 0.10,
    }
    assert passes(e, "spy", 0.03) is True  # 30 >= 23
    assert passes(e, "best", 0.03) is True  # 30 >= 28
    e2 = dict(e, total_return=0.26)
    assert passes(e2, "best", 0.03) is False  # 26 < 28
    assert passes(e2, "spy", 0.03) is True


def test_aggregate_and_verdicts():
    year_evals = []
    for year, spy, qqq, ret_a, ret_b in [
        ("2022", 0.02, -0.05, 0.10, -0.01),
        ("2023", 0.25, 0.55, 0.52, 0.10),
        ("2024", 0.24, 0.27, 0.40, 0.63),
        ("2025_study", 0.28, 0.38, 0.36, 0.05),
    ]:
        year_evals.append(
            {
                "strategy_id": "A_qqqish",
                "year": year,
                "total_return": ret_a,
                "spy_bh": spy,
                "qqq_bh": qqq,
                "iwm_bh": 0.1,
                "max_dd": -0.12,
                "asset_class": "equity",
                "n_opens": 2,
                "hard_kill": False,
            }
        )
        year_evals.append(
            {
                "strategy_id": "B_2024_alpha",
                "year": year,
                "total_return": ret_b,
                "spy_bh": spy,
                "qqq_bh": qqq,
                "iwm_bh": 0.1,
                "max_dd": -0.20,
                "asset_class": "equity",
                "n_opens": 5,
                "hard_kill": False,
            }
        )
    spy_rows = aggregate(year_evals, mode="spy")
    best_rows = aggregate(year_evals, mode="best")
    by = {r["strategy_id"]: r for r in spy_rows}
    assert by["A_qqqish"]["years_passed"] >= 2
    verdicts = decide_verdicts(spy_rows, best_rows)
    vby = {v["strategy_id"]: v for v in verdicts}
    assert vby["A_qqqish"]["verdict"] == "PROMOTE_RESEARCH"
    assert vby["B_2024_alpha"]["verdict"] in ("WATCH", "PROMOTE_RESEARCH", "HOLD")


def test_cli_on_fixture(tmp_path: Path):
    from scripts.rescore_mega_annual import main
    import sys

    full = {
        "year_evals": [
            {
                "strategy_id": "X",
                "year": "2023",
                "total_return": 0.40,
                "spy_bh": 0.20,
                "qqq_bh": 0.30,
                "iwm_bh": 0.10,
                "max_dd": -0.1,
                "asset_class": "equity",
                "n_opens": 1,
                "hard_kill": False,
            },
            {
                "strategy_id": "X",
                "year": "2024",
                "total_return": 0.35,
                "spy_bh": 0.20,
                "qqq_bh": 0.25,
                "iwm_bh": 0.10,
                "max_dd": -0.1,
                "asset_class": "equity",
                "n_opens": 1,
                "hard_kill": False,
            },
        ],
        "windows": [],
    }
    d = tmp_path / "latest"
    d.mkdir()
    (d / "full_results.json").write_text(json.dumps(full), encoding="utf-8")
    out = tmp_path / "RESCORE.md"
    old = sys.argv
    try:
        sys.argv = ["rescore", "--in", str(d), "--out", str(out)]
        assert main() == 0
    finally:
        sys.argv = old
    assert out.is_file()
    assert out.with_suffix(".json").is_file()
    assert "PROMOTE" in out.read_text(encoding="utf-8") or "WATCH" in out.read_text(
        encoding="utf-8"
    )
