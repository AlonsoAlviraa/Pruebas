"""CLI argparse / exit-code smokes for redesign scripts (no heavy data runs)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_s1_early_missing_data_root_exit_2(tmp_path: Path):
    mod = _load("run_s1_early_window", "scripts/run_s1_early_window.py")
    code = mod.main(
        [
            "--data-root",
            str(tmp_path / "no_such_data"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2


def test_s1_geo_missing_train_root_exit_2(tmp_path: Path):
    mod = _load("run_s1_geo_frozen", "scripts/run_s1_geo_frozen.py")
    code = mod.main(
        [
            "--train-data-root",
            str(tmp_path / "missing"),
            "--out",
            str(tmp_path / "out"),
            "--markets",
            "ES",
        ]
    )
    assert code == 2


def test_redesign_eval_missing_exit_2(tmp_path: Path):
    mod = _load("run_redesign_eval", "scripts/run_redesign_eval.py")
    code = mod.main(
        [
            "--data-root",
            str(tmp_path / "missing"),
            "--ticker-file",
            str(tmp_path / "t.txt"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2


def test_style_clone_gap_smoke_defaults_parse():
    mod = _load("run_style_clone_gap", "scripts/run_style_clone_gap.py")
    assert "same_L0" in mod.NO_LEAK_PROTOCOL


def test_early_parse_args_smoke_presets():
    """parse_args(--smoke) must hit real argparse + apply_smoke_full_defaults."""
    mod = _load("run_s1_early_window", "scripts/run_s1_early_window.py")
    args = mod.parse_args(["--smoke", "--universe-limit", "40"])
    assert args.smoke is True
    assert args.first_oos == 2012
    assert args.last_oos == 2014
    assert args.universe_limit == 15  # capped by smoke


def test_early_parse_args_full_presets():
    mod = _load("run_s1_early_window", "scripts/run_s1_early_window.py")
    args = mod.parse_args(["--full"])
    assert args.full is True
    assert args.first_oos == 2010
    assert args.last_oos == 2014
    assert args.universe_limit <= 40


def test_early_apply_smoke_helper_direct():
    mod = _load("run_s1_early_window", "scripts/run_s1_early_window.py")
    ns = type("NS", (), {"smoke": True, "full": False, "universe_limit": 99, "first_oos": 2005, "last_oos": 2020})()
    out = mod.apply_smoke_full_defaults(ns)
    assert out.first_oos == 2012 and out.last_oos == 2014
    assert out.universe_limit == 15


def test_redesign_parse_args_smoke_presets():
    """parse_args(--smoke) exercises redesign CLI helper, not local constants."""
    mod = _load("run_redesign_eval", "scripts/run_redesign_eval.py")
    args = mod.parse_args(["--smoke", "--universe-limit", "80", "--first-oos", "2018", "--last-oos", "2025"])
    assert args.smoke is True
    assert args.first_oos == 2023
    assert args.last_oos == 2024
    assert args.universe_limit == 15


def test_redesign_apply_smoke_helper_no_smoke_unchanged():
    mod = _load("run_redesign_eval", "scripts/run_redesign_eval.py")
    ns = type("NS", (), {"smoke": False, "universe_limit": 40, "first_oos": 2022, "last_oos": 2024})()
    out = mod.apply_smoke_defaults(ns)
    assert out.first_oos == 2022 and out.universe_limit == 40
