"""Unit tests portfolio meta + grid zoo + marks honesty (no network)."""
from __future__ import annotations

import numpy as np

from paper_live.options.grid_zoo import (
    STRUCTURAL_PROXY_NEGATIVE_KINDS,
    build_grid_zoo,
    filter_zoo_for_marks,
    is_banned_spec,
    is_proxy_short_vol_banned,
)
from paper_live.options.marks_policy import (
    CHAIN_PRICING_ENGINE_AVAILABLE,
    PROXY_META_EXCLUDE_KINDS,
    allow_kind_for_marks,
    filter_sleeve_years_for_marks,
    filter_specs_by_marks_mode,
    is_proxy_marks,
    is_real_marks,
    normalize_marks_mode,
    resolve_study_marks_context,
    short_vol_allowed,
)
from paper_live.portfolio.meta_label_selector import (
    MetaLabelConfig,
    build_feature_row,
    fit_meta,
    keep_one_per_underlying,
    make_meta_label,
    predict_proba,
    rank_sleeves_for_year,
    size_from_proba,
)
from paper_live.portfolio.sleeve_portfolio import (
    PortfolioCaps,
    allocate_weights,
    invested_weight,
    portfolio_vs_spy_cash_blend,
    portfolio_year_return,
    spy_cash_blend_return,
)


def test_grid_zoo_no_leverage_bans():
    zoo = build_grid_zoo(max_strategies=200, include_names=True, marks_mode="proxy_bs")
    assert zoo["n_strategies"] >= 30
    assert zoo["marks_mode"] in ("proxy_bs", "proxy_bs|vix_surface") or is_proxy_marks(
        zoo["marks_mode"]
    )
    for s in zoo["strategies"]:
        assert not is_banned_spec(s)
        assert "2X" not in s["id"].upper()
        assert float((s.get("meta") or {}).get("leverage") or 1) <= 1.01
        # proxy filter must drop structural short-vol pure
        assert s["kind"] not in STRUCTURAL_PROXY_NEGATIVE_KINDS


def test_meta_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 13))
    y = (X[:, 0] + rng.normal(0, 0.5, 80) > 0).astype(float)
    fit = fit_meta(X, y)
    if fit is None:
        return  # sklearn missing
    p = predict_proba(fit, X[:5])
    assert len(p) == 5
    assert size_from_proba(0.9) == 1.0
    assert size_from_proba(0.1) == 0.0


def test_allocate_caps():
    selected = [
        {
            "strategy_id": f"s{i}",
            "underlying": "SPY" if i < 6 else "QQQ",
            "kind": "put_credit_spread",
            "meta_size": 1.0,
            "prior_vol": 0.1,
            "prior_max_dd": -0.1,
        }
        for i in range(10)
    ]
    for s in selected:
        s["meta_proba"] = 0.7
        s["meta_score"] = 1.0
    w = allocate_weights(selected)
    assert sum(w.values()) <= 0.95 + 1e-6
    spy_w = sum(v for k, v in w.items() if k.startswith("s") and int(k[1:]) < 6)
    assert spy_w <= 0.45 + 1e-6 or len(w) < 3
    r = portfolio_year_return(w, {k: 0.1 for k in w})
    assert abs(r - 0.1 * sum(w.values())) < 1e-9


def test_rank_sleeves():
    cands = [
        {"strategy_id": "a", "prior_max_dd": -0.1, "underlying": "SPY", "kind": "iron_condor"},
        {"strategy_id": "b", "prior_max_dd": -0.3, "underlying": "QQQ", "kind": "long_call"},
    ]
    ranked = rank_sleeves_for_year(cands, [0.8, 0.6], top_k=2)
    assert ranked[0]["strategy_id"] == "a"


# --- marks mode gating ---


def test_marks_mode_normalize_and_gate():
    assert is_real_marks("real_chain")
    assert is_real_marks("yahoo_chain")
    assert is_real_marks("eodhd_options_eod")
    assert is_proxy_marks("proxy_bs")
    assert is_proxy_marks("proxy_bs|vix_surface")
    assert is_proxy_marks("vix_surface")
    assert normalize_marks_mode("proxy_bs|vix_surface") in (
        "proxy_bs",
        "proxy_bs|vix_surface",
    )
    # Engine currently unavailable → short_vol never allowed by default
    assert CHAIN_PRICING_ENGINE_AVAILABLE is False
    assert short_vol_allowed("real_chain") is False
    assert short_vol_allowed("proxy_bs") is False
    assert short_vol_allowed("real_chain", chain_engine_available=True) is True
    assert allow_kind_for_marks("iron_condor", "proxy_bs") is False
    assert allow_kind_for_marks("long_call", "proxy_bs") is True
    assert allow_kind_for_marks("iron_condor", "real_chain") is False  # no engine
    assert allow_kind_for_marks(
        "iron_condor", "real_chain", chain_engine_available=True
    ) is True
    assert allow_kind_for_marks("covered_call", "proxy_bs") is True  # equity-linked control


def test_normalize_marks_mode_fail_closed():
    # mixed real+proxy → proxy
    assert is_proxy_marks(normalize_marks_mode("proxy_bs|real_chain"))
    assert is_proxy_marks(normalize_marks_mode("real_chain,proxy_bs"))
    assert is_proxy_marks(normalize_marks_mode("real_chain|proxy_bs"))
    # substring heuristics must not promote unreal/not_real
    assert normalize_marks_mode("unreal") != "real_chain"
    assert normalize_marks_mode("not_real") != "real_chain"
    assert is_proxy_marks("unreal")  # unknown treated as non-real family for honesty
    # exact tokens only
    assert normalize_marks_mode("real_chain") == "real_chain"
    assert normalize_marks_mode("yahoo_chain") == "real_chain"


def test_resolve_study_marks_context_fail_closed():
    ctx = resolve_study_marks_context("real_chain", chain_engine_available=False)
    assert ctx["short_vol_allowed"] is False
    assert ctx["option_marks_label"] == "proxy_bs|vix_surface"
    assert ctx["forced_proxy"] is True
    assert ctx["forced_proxy_reason"]
    assert is_proxy_marks(ctx["effective_mode"])

    ctx2 = resolve_study_marks_context("proxy_bs")
    assert ctx2["short_vol_allowed"] is False
    assert ctx2["option_marks_label"] == "proxy_bs|vix_surface"
    assert ctx2["forced_proxy"] is False

    ctx3 = resolve_study_marks_context(
        "real_chain",
        chain_engine_available=True,
        pricing_backend="real_chain",
    )
    assert ctx3["short_vol_allowed"] is True
    assert ctx3["option_marks_label"] == "real_chain"


def test_filter_specs_proxy_excludes_short_vol():
    specs = [
        {"id": "1", "kind": "iron_condor", "underlying": "SPY"},
        {"id": "2", "kind": "call_credit_spread", "underlying": "QQQ"},
        {"id": "3", "kind": "long_call", "underlying": "AAPL"},
        {"id": "4", "kind": "cash", "underlying": "SPY"},
        {"id": "G_CASH_CTRL", "kind": "cash", "underlying": "SPY"},
        {"id": "5", "kind": "put_credit_spread", "underlying": "IWM"},
        {"id": "6", "kind": "cash_secured_put", "underlying": "MSFT"},
        {"id": "7", "kind": "covered_call", "underlying": "SPY"},
    ]
    kept = filter_specs_by_marks_mode(specs, "proxy_bs")
    kinds = {s["kind"] for s in kept}
    assert "iron_condor" not in kinds
    assert "call_credit_spread" not in kinds
    assert "put_credit_spread" not in kinds
    assert "cash_secured_put" not in kinds
    assert "long_call" in kinds
    assert "covered_call" in kinds
    assert "cash" in kinds

    # real_chain without engine still excludes short-vol pure
    kept_real_no_engine = filter_specs_by_marks_mode(specs, "real_chain")
    assert "iron_condor" not in {s["kind"] for s in kept_real_no_engine}

    # with engine, short-vol allowed
    kept_real = filter_specs_by_marks_mode(
        specs, "real_chain", chain_engine_available=True
    )
    assert len(kept_real) == len(specs)

    # apply_filter=False keeps short-vol (norm violation path)
    kept_nofilter = filter_specs_by_marks_mode(
        specs, "proxy_bs", apply_filter=False
    )
    assert "iron_condor" in {s["kind"] for s in kept_nofilter}


# --- beat_spy labels ---


def test_make_meta_label_modes():
    # positive_ret
    assert make_meta_label(0.05, spy_ret=0.20, label_mode="positive_ret") == 1.0
    assert make_meta_label(-0.01, spy_ret=-0.20, label_mode="positive_ret") == 0.0

    # beat_spy
    assert make_meta_label(0.18, spy_ret=0.15, label_mode="beat_spy") == 1.0
    assert make_meta_label(0.10, spy_ret=0.15, label_mode="beat_spy") == 0.0
    assert make_meta_label(-0.05, spy_ret=-0.10, label_mode="beat_spy") == 1.0
    # no silent fallback — None when spy missing
    assert make_meta_label(0.10, spy_ret=None, label_mode="beat_spy") is None

    # utility_excess: must beat max(spy, 0)
    assert make_meta_label(0.05, spy_ret=-0.10, label_mode="utility_excess") == 1.0
    assert make_meta_label(0.05, spy_ret=0.10, label_mode="utility_excess") == 0.0
    assert make_meta_label(-0.02, spy_ret=-0.10, label_mode="utility_excess") == 0.0
    assert make_meta_label(0.05, spy_ret=None, label_mode="utility_excess") is None

    # default config beat_spy
    cfg = MetaLabelConfig()
    assert cfg.label_mode == "beat_spy"
    assert make_meta_label(0.20, spy_ret=0.10, cfg=cfg) == 1.0


# --- one-per-underlying ---


def test_one_per_underlying_selection():
    cands = [
        {
            "strategy_id": "a1",
            "prior_max_dd": -0.1,
            "underlying": "SPY",
            "kind": "long_call",
        },
        {
            "strategy_id": "a2",
            "prior_max_dd": -0.05,
            "underlying": "SPY",
            "kind": "call_debit_spread",
        },
        {
            "strategy_id": "b1",
            "prior_max_dd": -0.2,
            "underlying": "QQQ",
            "kind": "long_put",
        },
        {
            "strategy_id": "c1",
            "prior_max_dd": -0.15,
            "underlying": "AAPL",
            "kind": "long_call",
        },
    ]
    # high proba for all so size > 0
    proba = [0.9, 0.95, 0.8, 0.7]
    ranked = rank_sleeves_for_year(cands, proba, top_k=8, one_per_underlying=True)
    unds = [r["underlying"] for r in ranked]
    assert len(unds) == len(set(unds))
    # SPY winner should be a2 (higher proba and better dd)
    spy_sel = [r for r in ranked if r["underlying"] == "SPY"]
    assert len(spy_sel) == 1
    assert spy_sel[0]["strategy_id"] == "a2"

    # without gate, can keep both SPY
    ranked_all = rank_sleeves_for_year(cands, proba, top_k=8, one_per_underlying=False)
    spy_n = sum(1 for r in ranked_all if r["underlying"] == "SPY")
    assert spy_n == 2


def test_keep_one_per_underlying_helper():
    rows = [
        {"underlying": "SPY", "meta_score": 1.0, "strategy_id": "x"},
        {"underlying": "SPY", "meta_score": 2.0, "strategy_id": "y"},
        {"underlying": "QQQ", "meta_score": 0.5, "strategy_id": "z"},
    ]
    out = keep_one_per_underlying(rows)
    assert len(out) == 2
    assert out[0]["strategy_id"] == "y"


def test_one_per_und_then_caps_leave_cash():
    """Many same-und candidates collapse; residual cash remains under caps."""
    cands = [
        {
            "strategy_id": f"s{i}",
            "prior_max_dd": -0.1,
            "underlying": und,
            "kind": "long_call",
            "prior_vol": 0.1,
        }
        for i, und in enumerate(["SPY"] * 5 + ["QQQ"] * 5 + ["AAPL"] * 5)
    ]
    proba = [0.9] * len(cands)
    ranked = rank_sleeves_for_year(cands, proba, top_k=8, one_per_underlying=True)
    assert len(ranked) == 3  # one per und
    caps = PortfolioCaps(min_cash=0.10)
    w = allocate_weights(ranked, caps=caps)
    assert sum(w.values()) <= 1.0 - caps.min_cash + 1e-9


# --- spy-cash blend math ---


def test_spy_cash_blend_math():
    assert abs(spy_cash_blend_return(0.4, 0.20) - 0.08) < 1e-12
    assert abs(spy_cash_blend_return(1.0, 0.15) - 0.15) < 1e-12
    assert abs(spy_cash_blend_return(0.0, 0.15) - 0.0) < 1e-12
    assert abs(spy_cash_blend_return(0.5, 0.10, cash_return=0.02) - 0.06) < 1e-12

    w = {"s1": 0.2, "s2": 0.15}
    assert abs(invested_weight(w) - 0.35) < 1e-12
    inv, blend, vs = portfolio_vs_spy_cash_blend(0.05, w, 0.20)
    assert abs(inv - 0.35) < 1e-12
    assert abs(blend - 0.07) < 1e-12
    assert abs(vs - (0.05 - 0.07)) < 1e-12
    inv2, blend2, vs2 = portfolio_vs_spy_cash_blend(0.05, w, None)
    assert blend2 is None and vs2 is None and abs(inv2 - 0.35) < 1e-12


# --- zoo short-vol filter under proxy ---


def test_zoo_short_vol_filter_under_proxy():
    # real_chain zoo may include short-vol pure kinds when filter off
    zoo_real = build_grid_zoo(
        max_strategies=400,
        include_names=True,
        marks_mode="real_chain",
        apply_proxy_short_vol_filter=False,
    )
    kinds_real = {s["kind"] for s in zoo_real["strategies"]}
    assert kinds_real & STRUCTURAL_PROXY_NEGATIVE_KINDS, (
        "real_chain zoo should still generate short-vol kinds for evaluation when subscribed"
    )

    zoo_proxy = build_grid_zoo(
        max_strategies=300, include_names=True, marks_mode="proxy_bs"
    )
    for s in zoo_proxy["strategies"]:
        assert s["kind"] not in STRUCTURAL_PROXY_NEGATIVE_KINDS
        assert not is_proxy_short_vol_banned(s, marks_mode="proxy_bs")
    assert "proxy_marks: exclude short-vol" in " ".join(zoo_proxy.get("ban_rules") or [])

    # post-filter path
    fake = [
        {"id": "ic", "kind": "iron_condor", "underlying": "SPY"},
        {"id": "lc", "kind": "long_call", "underlying": "SPY"},
        {"id": "ccs", "kind": "call_credit_spread", "underlying": "QQQ"},
    ]
    filtered = filter_zoo_for_marks(fake, "proxy_bs")
    assert [s["id"] for s in filtered] == ["lc"]
    assert set(PROXY_META_EXCLUDE_KINDS) >= {
        "iron_condor",
        "call_credit_spread",
        "put_credit_spread",
        "cash_secured_put",
    }


# --- rescore-style sleeve cache filter (kind always; full cache) ---


def test_filter_sleeve_years_for_marks_full_cache_by_kind():
    cache = {
        "ic1": {"2015": {"kind": "iron_condor", "underlying": "SPY", "total_return": -0.1}},
        "lc1": {"2015": {"kind": "long_call", "underlying": "AAPL", "total_return": 0.2}},
        "lc2": {"2015": {"kind": "long_call", "underlying": "MSFT", "total_return": 0.1}},
        "pcs1": {"2015": {"kind": "put_credit_spread", "underlying": "QQQ", "total_return": -0.05}},
        "G_CASH_CTRL": {"2015": {"kind": "cash", "underlying": "SPY", "total_return": 0.0}},
        # ID looks like allowed sample but kind is banned — must drop by kind
        "allowed_id_but_ic": {
            "2015": {"kind": "iron_condor", "underlying": "IWM", "total_return": -0.2}
        },
    }
    # Full cache, proxy filter
    kept = filter_sleeve_years_for_marks(cache, "proxy_bs", apply_filter=True)
    assert "lc1" in kept and "lc2" in kept
    assert "G_CASH_CTRL" in kept
    assert "ic1" not in kept
    assert "pcs1" not in kept
    assert "allowed_id_but_ic" not in kept

    # restrict_to_ids would shrink (study sample path) — rescore uses None
    kept_sub = filter_sleeve_years_for_marks(
        cache, "proxy_bs", apply_filter=True, restrict_to_ids={"lc1"}
    )
    assert set(kept_sub.keys()) == {"lc1", "G_CASH_CTRL"} or set(kept_sub.keys()) == {
        "lc1"
    }
    # cash always allowed if present
    assert "lc2" not in kept_sub

    # no filter: all including short-vol
    kept_all = filter_sleeve_years_for_marks(cache, "proxy_bs", apply_filter=False)
    assert "ic1" in kept_all and "pcs1" in kept_all


def test_specs_from_zoo_normalize_import_smoke(tmp_path):
    """Phase-A zoo load must not NameError on normalize_marks_mode."""
    import importlib
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    mod = importlib.import_module("scripts.run_options_portfolio_meta_study")
    zoo = {
        "capital0": 100_000.0,
        "marks_mode": "proxy_bs",
        "risk": {"max_portfolio_dd": 0.3},
        "strategies": [
            {
                "id": "G_CASH_CTRL",
                "label": "cash",
                "kind": "cash",
                "underlying": "SPY",
            },
            {
                "id": "lc1",
                "label": "long call",
                "kind": "long_call",
                "underlying": "AAPL",
                "dte_days": 30,
                "otm_pct": 0.05,
            },
            {
                "id": "ic1",
                "label": "ic",
                "kind": "iron_condor",
                "underlying": "SPY",
                "dte_days": 30,
                "otm_pct": 0.05,
            },
        ],
    }
    p = tmp_path / "tiny_zoo.json"
    p.write_text(json.dumps(zoo), encoding="utf-8")
    specs, capital0, risk, z = mod._specs_from_zoo(
        p, max_n=10, marks_mode="proxy_bs", apply_proxy_filter=True
    )
    assert capital0 == 100_000.0
    kinds = {s.kind for s in specs}
    assert "iron_condor" not in kinds
    assert "long_call" in kinds or "cash" in kinds
