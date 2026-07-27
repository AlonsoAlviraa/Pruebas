#!/usr/bin/env python3
"""Mega study: crash/oversold entry overlays + win-rate levers on turbo_highvol.

Trains the base highvol model **once per OOS year**, then evaluates many
backtest/overlay configs without retrain (research efficiency).

Windows:
  - Full OOS (default 2018–2025; optional 2016–2025)
  - Crash slices: 2018Q4, 2020 crash, 2022 bear

Research only — not live trading advice. No OPRA/options claims.

Usage (PowerShell, repo root):
  $env:PYTHONPATH = (Get-Location).Path
  python scripts/run_crash_entry_mega_study.py --smoke
  python scripts/run_crash_entry_mega_study.py --grid medium
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.backtest import BacktestConfig, run_portfolio_backtest
from trad_research.crash_entry import (
    CrashEntryConfig,
    WinRateFilterConfig,
    build_crash_entry_map,
    composite_rank_score,
)
from trad_research.features import list_tickers
from trad_research.metrics import equity_metrics
from trad_research.regime import build_all_regime_maps
from trad_research.strategies import HighVolMinAllocStrategy, HighVolTurboStrategy, get_strategy
from trad_research.walk_forward import (
    _build_training_frame,
    _load_panels,
    load_benchmark_equity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crash_mega")

CRASH_WINDOWS: Dict[str, Tuple[str, str]] = {
    "crash_2018q4": ("2018-10-01", "2018-12-31"),
    "crash_2020": ("2020-02-01", "2020-04-30"),
    "crash_2022": ("2022-01-01", "2022-10-31"),
}


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return default


def _window_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    start: str,
    end: str,
    *,
    start_equity_fallback: float = 100_000.0,
) -> Dict[str, float]:
    """Return/DD/entries inside [start, end] using only equity bars in range."""
    if equity is None or equity.empty:
        return {
            "return": float("nan"),
            "max_dd": float("nan"),
            "n_entries": 0.0,
            "win_rate": float("nan"),
        }
    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index, utc=True)
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    sl = eq[(eq.index >= t0) & (eq.index <= t1)]
    if sl.empty or len(sl) < 2:
        return {
            "return": float("nan"),
            "max_dd": float("nan"),
            "n_entries": 0.0,
            "win_rate": float("nan"),
        }
    ret = float(sl.iloc[-1] / sl.iloc[0] - 1.0)
    peak = sl.cummax()
    mdd = float((sl / peak - 1.0).min())
    n_ent = 0
    wr = float("nan")
    if trades is not None and not trades.empty and "entry_date" in trades.columns:
        ed = pd.to_datetime(trades["entry_date"], utc=True)
        mask = (ed >= t0) & (ed <= t1)
        sub = trades.loc[mask]
        n_ent = int(len(sub))
        if n_ent > 0 and "net_profit" in sub.columns:
            wr = float((sub["net_profit"] > 0).mean())
    return {
        "return": ret,
        "max_dd": mdd,
        "n_entries": float(n_ent),
        "win_rate": wr,
    }


def _build_config_grid(grid: str) -> List[Dict[str, Any]]:
    """Named research configs. Smoke = tiny; week = curated 5; medium ~20–30; full ~40+."""
    base_names = ["turbo_highvol", "turbo_highvol_minalloc"]

    def row(
        base: str,
        label: str,
        crash: Optional[CrashEntryConfig] = None,
        wr: Optional[WinRateFilterConfig] = None,
        extra_bt: Optional[Dict[str, Any]] = None,
        breadth: Any = None,
        regime_key: Optional[str] = None,
        peak_mode: str = "continuous",
    ) -> Dict[str, Any]:
        eb = dict(extra_bt or {})
        # Lever may stash peak_mode under _peak_mode
        pm = str(eb.pop("_peak_mode", None) or peak_mode or "continuous")
        return {
            "id": f"{base}__{label}",
            "base": base,
            "label": label,
            "crash": crash,
            "wr": wr,
            "extra_bt": eb,
            "breadth": breadth,
            "regime_key": regime_key,
            "peak_mode": pm,
        }

    configs: List[Dict[str, Any]] = []

    # Week plan: curated highvol80 evidence set (max 5 configs; no dual baselines)
    if grid in ("week", "curated"):
        wr_pack = WinRateFilterConfig(
            hard_stop_cooldown_days=10,
            max_atr_pct_tight=0.16,
            soft_trend_non_crash=True,
        )
        crash_rsi30 = CrashEntryConfig(
            enabled=True,
            mode="rsi",
            rsi_threshold=30.0,
            crash_min_confidence=0.22,
            relax_regime=True,
            crash_relax_trend=True,
        )
        configs.append(row("turbo_highvol_minalloc", "baseline", crash=None, wr=None))
        configs.append(
            row(
                "turbo_highvol_minalloc",
                "crash_rsi30_wr",
                crash=crash_rsi30,
                wr=wr_pack,
            )
        )
        configs.append(row("turbo_highvol", "wr_pack", wr=wr_pack))
        configs.append(
            row(
                "turbo_highvol",
                "crash_dd15",
                crash=CrashEntryConfig(
                    enabled=True,
                    mode="dd",
                    dd_threshold=-0.15,
                    crash_min_confidence=0.22,
                    relax_regime=True,
                    crash_relax_trend=True,
                ),
            )
        )
        configs.append(
            row(
                "turbo_highvol",
                "crash_rsi_or_dd15",
                crash=CrashEntryConfig(
                    enabled=True,
                    mode="rsi_or_dd",
                    rsi_threshold=30.0,
                    dd_threshold=-0.15,
                    crash_min_confidence=0.22,
                    relax_regime=True,
                    crash_relax_trend=True,
                ),
            )
        )
        return configs

    # Phase C risk A/B: minalloc control vs DD circuit (single primary lever)
    if grid in ("week_risk", "risk_ab"):
        from trad_research.risk_levers import WEEK_PRIMARY_LEVER_ID, week_risk_ab_extra_bt

        extras = week_risk_ab_extra_bt()
        configs.append(
            row(
                "turbo_highvol_minalloc",
                "baseline",
                extra_bt=dict(extras["baseline"]),
            )
        )
        configs.append(
            row(
                "turbo_highvol_minalloc",
                WEEK_PRIMARY_LEVER_ID,
                extra_bt=dict(extras[WEEK_PRIMARY_LEVER_ID]),
            )
        )
        return configs

    # Alt-loop MDD attack: baseline + registered combos + breadth + best prior sleeve
    if grid in ("alt_mdd", "alt_loop"):
        from trad_research.breadth_gate import BreadthGateConfig
        from trad_research.risk_levers import (
            ALT_PRIMARY_LEVER_ID,
            alt_mdd_extra_bt_for_strategy,
            alt_mdd_lever_ids,
            get_lever,
        )

        base_name = "turbo_highvol_minalloc"
        strat = _base_strategy(base_name)
        base_ov = strat.backtest_overrides()

        configs.append(
            row(
                base_name,
                "baseline",
                extra_bt=alt_mdd_extra_bt_for_strategy(base_ov, "baseline"),
            )
        )
        for lid in alt_mdd_lever_ids():
            configs.append(
                row(
                    base_name,
                    lid,
                    extra_bt=alt_mdd_extra_bt_for_strategy(base_ov, lid),
                )
            )
        # Breadth gate (≥40% names above SMA50) + primary DD/vol combo
        bcfg = BreadthGateConfig(
            enabled=True,
            sma_period=50,
            min_breadth=0.40,
            min_names=8,
            description="Universe breadth ≥40% above SMA50",
        )
        configs.append(
            row(
                base_name,
                f"breadth40_{ALT_PRIMARY_LEVER_ID}",
                extra_bt=alt_mdd_extra_bt_for_strategy(base_ov, ALT_PRIMARY_LEVER_ID),
                breadth=bcfg,
            )
        )
        # Best prior sleeve (crash RSI30 + WR) + DD circuit only (size unchanged)
        wr_pack = WinRateFilterConfig(
            hard_stop_cooldown_days=10,
            max_atr_pct_tight=0.16,
            soft_trend_non_crash=True,
        )
        crash_rsi30 = CrashEntryConfig(
            enabled=True,
            mode="rsi",
            rsi_threshold=30.0,
            crash_min_confidence=0.22,
            relax_regime=True,
            crash_relax_trend=True,
        )
        configs.append(
            row(
                base_name,
                "crash_rsi30_wr_dd25",
                crash=crash_rsi30,
                wr=wr_pack,
                extra_bt=alt_mdd_extra_bt_for_strategy(base_ov, "dd_circuit_25"),
            )
        )
        # Cap at 6 configs for medium multi-year (baseline + 4 levers + breadth|crash pick)
        # breadth + crash added → 1+4+2 = 7; trim vol-only helpers already out.
        # Prefer primary combo documented in get_lever
        _ = get_lever(ALT_PRIMARY_LEVER_ID)
        return configs

    # Loop2: yearly peak / soft breach / vol-only (after Loop1 permanent-cash diagnosis)
    if grid in ("alt_mdd_v2", "alt_loop_v2"):
        from trad_research.breadth_gate import BreadthGateConfig
        from trad_research.risk_levers import (
            alt_mdd_extra_bt_for_strategy,
            alt_mdd_v2_lever_ids,
        )

        base_name = "turbo_highvol_minalloc"
        strat = _base_strategy(base_name)
        base_ov = strat.backtest_overrides()

        configs.append(
            row(
                base_name,
                "baseline",
                extra_bt=alt_mdd_extra_bt_for_strategy(base_ov, "baseline"),
            )
        )
        for lid in alt_mdd_v2_lever_ids():
            configs.append(
                row(
                    base_name,
                    lid,
                    extra_bt=alt_mdd_extra_bt_for_strategy(base_ov, lid),
                )
            )
        # Breadth + yearly primary
        bcfg = BreadthGateConfig(
            enabled=True,
            sma_period=50,
            min_breadth=0.40,
            min_names=8,
            description="Universe breadth ≥40% above SMA50",
        )
        configs.append(
            row(
                base_name,
                "breadth40_dd25_vt70_yr",
                extra_bt=alt_mdd_extra_bt_for_strategy(base_ov, "dd25_vt70_yr"),
                breadth=bcfg,
            )
        )
        return configs

    # Always include pure baselines (smoke / medium / full)
    for b in base_names:
        configs.append(row(b, "baseline", crash=None, wr=None))

    if grid == "smoke":
        configs.append(
            row(
                "turbo_highvol",
                "crash_rsi30",
                crash=CrashEntryConfig(
                    enabled=True,
                    mode="rsi",
                    rsi_threshold=30.0,
                    crash_min_confidence=0.22,
                    relax_regime=True,
                    crash_relax_trend=True,
                ),
            )
        )
        configs.append(
            row(
                "turbo_highvol",
                "wr_cooldown10",
                wr=WinRateFilterConfig(
                    hard_stop_cooldown_days=10,
                    max_atr_pct_tight=0.16,
                    soft_trend_non_crash=True,
                ),
            )
        )
        configs.append(
            row(
                "turbo_highvol",
                "crash_rsi30_wr",
                crash=CrashEntryConfig(
                    enabled=True,
                    mode="rsi",
                    rsi_threshold=30.0,
                    crash_min_confidence=0.22,
                    relax_regime=True,
                    crash_relax_trend=True,
                ),
                wr=WinRateFilterConfig(
                    hard_stop_cooldown_days=10,
                    max_atr_pct_tight=0.16,
                    soft_trend_non_crash=True,
                ),
            )
        )
        return configs

    # Medium / full grids
    rsi_thrs = [20.0, 25.0, 30.0] if grid == "full" else [25.0, 30.0]
    dd_thrs = [-0.12, -0.20] if grid == "full" else [-0.15]
    confs = [0.20, 0.25] if grid == "full" else [0.22]

    for b in base_names:
        for thr in rsi_thrs:
            for conf in confs[:1]:  # keep conf grid light
                configs.append(
                    row(
                        b,
                        f"crash_rsi{int(thr)}",
                        crash=CrashEntryConfig(
                            enabled=True,
                            mode="rsi",
                            rsi_threshold=thr,
                            crash_min_confidence=conf,
                            relax_regime=True,
                            crash_relax_trend=True,
                            crash_score_boost=1.15,
                        ),
                    )
                )
                configs.append(
                    row(
                        b,
                        f"crash_rsi{int(thr)}_recover",
                        crash=CrashEntryConfig(
                            enabled=True,
                            mode="rsi_recover",
                            rsi_threshold=thr,
                            require_rsi_rising=True,
                            crash_min_confidence=conf,
                            relax_regime=True,
                            crash_relax_trend=True,
                        ),
                    )
                )
        for dd in dd_thrs:
            configs.append(
                row(
                    b,
                    f"crash_dd{int(abs(dd)*100)}",
                    crash=CrashEntryConfig(
                        enabled=True,
                        mode="dd",
                        dd_threshold=dd,
                        crash_min_confidence=0.22,
                        relax_regime=True,
                        crash_relax_trend=True,
                    ),
                )
            )
            configs.append(
                row(
                    b,
                    f"crash_rsi_or_dd{int(abs(dd)*100)}",
                    crash=CrashEntryConfig(
                        enabled=True,
                        mode="rsi_or_dd",
                        rsi_threshold=30.0,
                        dd_threshold=dd,
                        crash_min_confidence=0.22,
                        relax_regime=True,
                        crash_relax_trend=True,
                    ),
                )
            )
        # WR-only and combined packs
        configs.append(
            row(
                b,
                "wr_pack",
                wr=WinRateFilterConfig(
                    hard_stop_cooldown_days=10,
                    max_atr_pct_tight=0.16,
                    soft_trend_non_crash=True,
                ),
            )
        )
        configs.append(
            row(
                b,
                "wr_pack_cd15_atr14",
                wr=WinRateFilterConfig(
                    hard_stop_cooldown_days=15,
                    max_atr_pct_tight=0.14,
                    soft_trend_non_crash=True,
                ),
            )
        )
        configs.append(
            row(
                b,
                "crash_rsi30_wr",
                crash=CrashEntryConfig(
                    enabled=True,
                    mode="rsi",
                    rsi_threshold=30.0,
                    crash_min_confidence=0.22,
                    relax_regime=True,
                    crash_relax_trend=True,
                ),
                wr=WinRateFilterConfig(
                    hard_stop_cooldown_days=10,
                    max_atr_pct_tight=0.16,
                    soft_trend_non_crash=True,
                ),
            )
        )
        configs.append(
            row(
                b,
                "crash_rsi25_wr",
                crash=CrashEntryConfig(
                    enabled=True,
                    mode="rsi",
                    rsi_threshold=25.0,
                    crash_min_confidence=0.22,
                    relax_regime=True,
                    crash_relax_trend=True,
                ),
                wr=WinRateFilterConfig(
                    hard_stop_cooldown_days=10,
                    max_atr_pct_tight=0.16,
                    soft_trend_non_crash=True,
                ),
            )
        )
        if grid == "full":
            configs.append(
                row(
                    b,
                    "crash_rsi30_softreg",
                    crash=CrashEntryConfig(
                        enabled=True,
                        mode="rsi",
                        rsi_threshold=30.0,
                        crash_min_confidence=0.22,
                        relax_regime=True,
                        crash_relax_trend=True,
                    ),
                    extra_bt={"soft_hard_regime": True, "regime_hard_size_scale": 0.40},
                )
            )

    # Dedupe by id
    seen = set()
    out = []
    for c in configs:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        out.append(c)
    return out


def _base_strategy(name: str):
    if name == "turbo_highvol_minalloc":
        return HighVolMinAllocStrategy()
    if name == "turbo_highvol":
        return HighVolTurboStrategy()
    return get_strategy(name)


def _make_bt(
    strategy,
    capital: float,
    regime_map,
    soft_regime,
    crash_cfg: Optional[CrashEntryConfig],
    wr_cfg: Optional[WinRateFilterConfig],
    crash_map: Optional[Dict],
    extra_bt: Dict[str, Any],
    *,
    peak_equity_seed: Optional[float] = None,
) -> BacktestConfig:
    overrides = strategy.backtest_overrides()
    fields = {**BacktestConfig().__dict__, **overrides, **extra_bt}
    # Drop non-BacktestConfig keys
    allowed = set(BacktestConfig.__dataclass_fields__)
    bt = BacktestConfig(**{k: v for k, v in fields.items() if k in allowed})
    bt.initial_capital = capital
    bt.regime_ok = regime_map
    bt.soft_regime_ok = soft_regime
    # Carry peak high-water mark across OOS years for continuous max_portfolio_dd
    if peak_equity_seed is not None and float(peak_equity_seed) > 0:
        bt.peak_equity_seed = float(peak_equity_seed)
    if crash_cfg is not None and crash_cfg.enabled:
        bt.crash_entry_on = crash_map or {}
        bt.crash_entry_cfg = crash_cfg
        bt.crash_relax_regime = bool(crash_cfg.relax_regime)
        bt.crash_score_boost = float(crash_cfg.crash_score_boost or 1.0)
    else:
        bt.crash_entry_on = None
        bt.crash_entry_cfg = None
        bt.crash_relax_regime = False
    if wr_cfg is not None:
        bt.winrate_filter_cfg = wr_cfg
        bt.hard_stop_cooldown_days = int(wr_cfg.hard_stop_cooldown_days or 0)
        if wr_cfg.max_atr_pct_tight is not None:
            bt.max_atr_pct_entry = float(wr_cfg.max_atr_pct_tight)
    return bt


def _resolve_universe_limit(limit: Optional[int]) -> Optional[int]:
    """0 / negative / None → no cap (full ticker file). Positive → first N with data."""
    if limit is None:
        return None
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return n


def run_mega(
    *,
    data_root: Path,
    ticker_file: Path,
    universe_limit: Optional[int],
    first_oos: int,
    last_oos: int,
    grid: str,
    out_dir: Path,
    min_train_rows: int = 3000,
    configs: Optional[List[Dict[str, Any]]] = None,
    regime_key: Optional[str] = None,
    preferred_index: Optional[List[str]] = None,
    market_id: str = "US",
) -> Dict[str, Any]:
    """Run mega overlay study.

    ``configs`` optional override (overnight / custom grids). When provided,
    ``grid`` is stored as a label only.

    Multi-market: pass ``preferred_index`` (e.g. IBEX/DAX) and a market-appropriate
    ``regime_key`` (US: strict_dual_golden; ES: ibex_abs_mom12 or portable_not_deep_bear).
    """
    t0 = time.time()
    configs = list(configs) if configs is not None else _build_config_grid(grid)
    if not configs:
        raise RuntimeError("run_mega: empty configs")
    univ_cap = _resolve_universe_limit(universe_limit)
    logger.info(
        "Grid=%s market=%s configs=%d years=%d-%d univ_limit=%s data=%s",
        grid,
        market_id,
        len(configs),
        first_oos,
        last_oos,
        univ_cap if univ_cap is not None else "full",
        data_root,
    )

    tickers = list_tickers(ticker_file, data_root, limit=univ_cap)
    panels = _load_panels(tickers, data_root)
    if len(panels) < 5:
        raise RuntimeError(f"Too few panels loaded: {len(panels)}")

    pref = tuple(preferred_index) if preferred_index else None
    all_regimes = build_all_regime_maps(data_root, preferred_index=pref)
    rk = str(regime_key or "strict_dual_golden")
    if rk not in all_regimes:
        # Fallbacks by market type
        for cand in (
            rk,
            "portable_not_deep_bear",
            "ibex_abs_mom12",
            "legacy_sma50",
            "none",
        ):
            if cand in all_regimes:
                rk = cand
                break
        else:
            rk = next(iter(all_regimes.keys()))
    regime_map, soft_regime, regime_desc = all_regimes[rk]
    regime_key = rk
    logger.info("Regime: %s — %s (preferred_index=%s)", regime_key, regime_desc, pref)

    # Optional universe breadth map (shared across configs that request it)
    from trad_research.breadth_gate import (
        and_regime_maps,
        build_breadth_risk_on_map,
        closes_from_panels,
    )
    from trad_research.risk_levers import (
        resolve_peak_equity_seed,
        update_peak_equity_state,
    )

    breadth_cache: Dict[str, Tuple[Dict, Dict]] = {}

    def get_breadth_map(bcfg) -> Tuple[Dict, Dict]:
        if bcfg is None or not getattr(bcfg, "enabled", False):
            return {}, {}
        key = json.dumps(bcfg.to_dict(), sort_keys=True)
        if key not in breadth_cache:
            closes = closes_from_panels(panels)
            risk_on, _series, meta = build_breadth_risk_on_map(closes, bcfg)
            breadth_cache[key] = (risk_on, meta)
            logger.info(
                "Breadth gate: min=%.0f%% mean_breadth=%.2f frac_on=%.1f%%",
                100 * float(bcfg.min_breadth),
                float(meta.get("mean_breadth") or float("nan")),
                100 * float(meta.get("frac_risk_on") or float("nan")),
            )
        return breadth_cache[key]

    # Pre-build crash maps per unique CrashEntryConfig signature
    crash_cache: Dict[str, Tuple[Dict, Dict]] = {}

    def get_crash_map(cc: Optional[CrashEntryConfig]):
        if cc is None or not cc.enabled:
            return {}, {}
        key = json.dumps(cc.to_dict(), sort_keys=True)
        if key not in crash_cache:
            crash_cache[key] = build_crash_entry_map(data_root, cc)
        return crash_cache[key]

    # Group configs by base strategy so we train once per (base, year)
    by_base: Dict[str, List[Dict[str, Any]]] = {}
    for c in configs:
        by_base.setdefault(c["base"], []).append(c)

    # Accumulators per config id
    # peak_equity: continuous high-water mark across OOS years (DD circuit seed)
    state: Dict[str, Dict[str, Any]] = {
        c["id"]: {
            "id": c["id"],
            "base": c["base"],
            "label": c["label"],
            "capital": 100_000.0,
            "peak_equity": None,  # filled after first segment; carried thereafter
            "equity_segments": [],
            "trades": [],
            "year_results": [],
            "crash_cfg": c["crash"].to_dict() if c["crash"] else None,
            "wr_cfg": c["wr"].to_dict() if c["wr"] else None,
            "breadth_cfg": (
                c["breadth"].to_dict()
                if c.get("breadth") is not None and hasattr(c["breadth"], "to_dict")
                else None
            ),
        }
        for c in configs
    }

    for base_name, base_cfgs in by_base.items():
        strategy = _base_strategy(base_name)
        logger.info("Base strategy %s (%d variants)", base_name, len(base_cfgs))
        for year in range(first_oos, last_oos + 1):
            test_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            test_end = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
            train_end = test_start
            train_df = _build_training_frame(
                panels,
                train_end=train_end,
                embargo_days=5,
                k_tp=2.5,
                k_sl=1.5,
                label_horizon=20,
            )
            if len(train_df) < min_train_rows:
                logger.warning("%s %s: skip train rows=%d", base_name, year, len(train_df))
                continue
            y = train_df["y_side"]
            hold = train_df[y == 1]
            nonhold = train_df[y != 1]
            if len(hold) > 2 * len(nonhold) and len(nonhold) > 1000:
                hold = hold.sample(n=min(len(hold), 2 * len(nonhold)), random_state=year)
                train_df = pd.concat([nonhold, hold], ignore_index=True)
            logger.info("%s %s: training on %d rows...", base_name, year, len(train_df))
            strategy.train(train_df, year)

            for c in base_cfgs:
                cid = c["id"]
                st = state[cid]
                cmap, _cmeta = get_crash_map(c["crash"])
                # Per-config regime (optional override) AND optional breadth gate
                rk = c.get("regime_key") or regime_key
                if rk in all_regimes:
                    rmap_c, soft_c, _ = all_regimes[rk]
                else:
                    rmap_c, soft_c = regime_map, soft_regime
                bmap, _bmeta = get_breadth_map(c.get("breadth"))
                if bmap:
                    rmap_c = and_regime_maps(rmap_c, bmap)
                # peak_mode: continuous = multi-year HWM; yearly = start peak at capital
                peak_mode = str(c.get("peak_mode") or "continuous")
                seed_peak = resolve_peak_equity_seed(peak_mode, st.get("peak_equity"))
                bt = _make_bt(
                    strategy,
                    st["capital"],
                    rmap_c,
                    soft_c,
                    c["crash"],
                    c["wr"],
                    cmap,
                    c.get("extra_bt") or {},
                    peak_equity_seed=float(seed_peak) if seed_peak is not None else None,
                )
                trades, equity, _ = run_portfolio_backtest(
                    panels, strategy, bt, start=test_start, end=test_end
                )
                if equity.empty:
                    continue
                yrep = equity_metrics(equity, start_equity=st["capital"], trades=trades)
                st["year_results"].append(
                    {
                        "year": year,
                        "year_return": yrep.final_equity / st["capital"] - 1.0,
                        "n_trades": yrep.n_trades,
                        "win_rate": yrep.win_rate,
                        "sharpe": yrep.sharpe,
                        "max_drawdown": yrep.max_drawdown,
                    }
                )
                if not trades.empty:
                    t = trades.copy()
                    t["oos_year"] = year
                    t["config_id"] = cid
                    st["trades"].append(t)
                st["equity_segments"].append(equity)
                st["capital"] = float(equity.iloc[-1])
                # Continuous peak: max(prior peak, segment max equity, ending capital)
                # Yearly mode still tracks for logging but next year re-seeds from capital.
                seg_hi = float(equity.max()) if len(equity) else st["capital"]
                st["peak_equity"] = update_peak_equity_state(
                    peak_mode,
                    st.get("peak_equity"),
                    seg_hi,
                    st["capital"],
                    float(bt.initial_capital),
                )
                logger.info(
                    "  %s %s: ret=%.1f%% trades=%d WR=%.1f%% peak=%.0f",
                    cid,
                    year,
                    st["year_results"][-1]["year_return"] * 100,
                    yrep.n_trades,
                    100 * yrep.win_rate,
                    float(st["peak_equity"] or 0),
                )

    # Aggregate + rank
    rows: List[Dict[str, Any]] = []
    for cid, st in state.items():
        if not st["equity_segments"]:
            rows.append(
                {
                    "id": cid,
                    "base": st["base"],
                    "label": st["label"],
                    "error": "no_equity",
                    "composite": -999.0,
                }
            )
            continue
        full_eq = pd.concat(st["equity_segments"])
        full_eq = full_eq[~full_eq.index.duplicated(keep="last")].sort_index()
        trades_df = (
            pd.concat(st["trades"], ignore_index=True) if st["trades"] else pd.DataFrame()
        )
        start_eq = 100_000.0
        bench = load_benchmark_equity(
            data_root,
            full_eq.index.min(),
            full_eq.index.max(),
            preferred=list(pref) if pref else None,
        )
        rep = equity_metrics(
            full_eq,
            start_equity=start_eq,
            trades=trades_df,
            benchmark=bench,
            positive_year_frac=(
                sum(1 for y in st["year_results"] if y["year_return"] > 0)
                / max(len(st["year_results"]), 1)
            ),
        )
        spy_ret = rep.benchmark_total_return
        excess = None
        if spy_ret is not None:
            excess = rep.total_return - float(spy_ret)

        row: Dict[str, Any] = {
            "id": cid,
            "base": st["base"],
            "label": st["label"],
            "total_return": rep.total_return,
            "cagr": rep.cagr,
            "sharpe": rep.sharpe,
            "sortino": rep.sortino,
            "max_drawdown": rep.max_drawdown,
            "win_rate": rep.win_rate,
            "n_trades": rep.n_trades,
            "profit_factor": rep.profit_factor,
            "calmar": rep.calmar,
            "years": rep.years,
            "spy_total_return": spy_ret,
            "excess_total_vs_spy": excess,
            "excess_cagr": rep.excess_cagr,
            "positive_year_frac": rep.positive_year_frac,
            "crash_cfg": st["crash_cfg"],
            "wr_cfg": st["wr_cfg"],
            "year_results": st["year_results"],
        }
        for wname, (ws, we) in CRASH_WINDOWS.items():
            wm = _window_metrics(full_eq, trades_df, ws, we)
            row[f"{wname}_return"] = wm["return"]
            row[f"{wname}_max_dd"] = wm["max_dd"]
            row[f"{wname}_n_entries"] = wm["n_entries"]
            row[f"{wname}_win_rate"] = wm["win_rate"]
        # Alias for composite scorer
        row["crash_2020_return"] = row.get("crash_2020_return")
        row["composite"] = composite_rank_score(row)
        rows.append(row)

        # Optional per-config equity dump (compact)
        cfg_dir = out_dir / "configs" / cid
        cfg_dir.mkdir(parents=True, exist_ok=True)
        full_eq.to_csv(cfg_dir / "equity.csv", header=["equity"])
        if not trades_df.empty:
            trades_df.to_csv(cfg_dir / "trades.csv", index=False)
        with open(cfg_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in row.items() if k != "year_results"}, f, indent=2, default=str)

    rows_sorted = sorted(rows, key=lambda r: float(r.get("composite") or -999), reverse=True)
    baseline_rows = [r for r in rows if r.get("label") == "baseline"]
    summary = {
        "version": "crash_entry_mega_v1",
        "disclaimer": "Research only. Not financial advice. No guaranteed edge.",
        "grid": grid,
        "market_id": market_id,
        "preferred_index": list(pref) if pref else None,
        "first_oos": first_oos,
        "last_oos": last_oos,
        "universe_limit": univ_cap if univ_cap is not None else 0,
        "universe_n_loaded": len(panels),
        "n_configs": len(rows),
        "elapsed_sec": round(time.time() - t0, 1),
        "regime": regime_key,
        "crash_windows": CRASH_WINDOWS,
        "baselines": baseline_rows,
        "top_by_composite": rows_sorted[:10],
        "top_by_win_rate": sorted(
            [r for r in rows if r.get("n_trades", 0) and r["n_trades"] >= 10],
            key=lambda r: float(r.get("win_rate") or 0),
            reverse=True,
        )[:10],
        "top_by_cagr": sorted(
            rows, key=lambda r: float(r.get("cagr") or -9), reverse=True
        )[:10],
        "all_rows": rows_sorted,
    }
    return summary


def _write_summary_md(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Crash entry mega study — SUMMARY",
        "",
        "> **Research only.** Not financial advice. Past backtests ≠ future results.",
        "",
        f"- Grid: `{summary.get('grid')}`",
        f"- OOS years: {summary.get('first_oos')}–{summary.get('last_oos')}",
        f"- Universe limit: {summary.get('universe_limit')}",
        f"- Configs: {summary.get('n_configs')}",
        f"- Elapsed: {summary.get('elapsed_sec')}s",
        f"- Regime: `{summary.get('regime')}`",
        "",
        "## Recommendation (heuristic)",
        "",
    ]
    top = (summary.get("top_by_composite") or [None])[0]
    bases = summary.get("baselines") or []
    if top and not top.get("error"):
        lines.append(
            f"**Top composite:** `{top.get('id')}` — "
            f"CAGR {_safe_float(top.get('cagr'))*100:.1f}%, "
            f"WR {_safe_float(top.get('win_rate'))*100:.1f}%, "
            f"Sharpe {_safe_float(top.get('sharpe')):.2f}, "
            f"MDD {_safe_float(top.get('max_drawdown'))*100:.1f}%, "
            f"2020 crash ret {_safe_float(top.get('crash_2020_return'))*100:.1f}% "
            f"(entries {int(_safe_float(top.get('crash_2020_n_entries'), 0))})."
        )
        lines.append("")
        lines.append(
            "Prefer a config that **raises win rate vs baseline** and **improves "
            "crash-window entries/returns** without collapsing long-run CAGR/Sharpe. "
            "If none do both, keep baseline for return and use crash overlay only as a "
            "research sleeve, not a promotion claim."
        )
    else:
        lines.append("No ranked configs (smoke/empty).")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("| id | CAGR | WR | Sharpe | MDD | n_trades | excess vs SPY |")
    lines.append("|----|------|-----|--------|-----|----------|---------------|")
    for b in bases:
        lines.append(
            f"| `{b.get('id')}` | {_safe_float(b.get('cagr'))*100:.1f}% | "
            f"{_safe_float(b.get('win_rate'))*100:.1f}% | {_safe_float(b.get('sharpe')):.2f} | "
            f"{_safe_float(b.get('max_drawdown'))*100:.1f}% | {b.get('n_trades')} | "
            f"{_safe_float(b.get('excess_total_vs_spy'))*100:.1f}% |"
        )
    lines.append("")
    lines.append("## Top 10 by composite (WR + excess + crash survival)")
    lines.append("")
    lines.append(
        "| rank | id | CAGR | WR | Sharpe | MDD | 2020 ret | 2020 entries | composite |"
    )
    lines.append(
        "|------|----|------|-----|--------|-----|----------|--------------|-----------|"
    )
    for i, r in enumerate(summary.get("top_by_composite") or [], 1):
        lines.append(
            f"| {i} | `{r.get('id')}` | {_safe_float(r.get('cagr'))*100:.1f}% | "
            f"{_safe_float(r.get('win_rate'))*100:.1f}% | {_safe_float(r.get('sharpe')):.2f} | "
            f"{_safe_float(r.get('max_drawdown'))*100:.1f}% | "
            f"{_safe_float(r.get('crash_2020_return'))*100:.1f}% | "
            f"{int(_safe_float(r.get('crash_2020_n_entries'), 0))} | "
            f"{_safe_float(r.get('composite')):.3f} |"
        )
    lines.append("")
    lines.append("## Top by win rate (n_trades ≥ 10)")
    lines.append("")
    lines.append("| id | WR | CAGR | n_trades | PF |")
    lines.append("|----|-----|------|----------|-----|")
    for r in summary.get("top_by_win_rate") or []:
        lines.append(
            f"| `{r.get('id')}` | {_safe_float(r.get('win_rate'))*100:.1f}% | "
            f"{_safe_float(r.get('cagr'))*100:.1f}% | {r.get('n_trades')} | "
            f"{_safe_float(r.get('profit_factor')):.2f} |"
        )
    lines.append("")
    lines.append("## Crash window definitions")
    lines.append("")
    for k, (a, b) in (summary.get("crash_windows") or {}).items():
        lines.append(f"- `{k}`: {a} → {b}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Crash flags use **causal** index RSI(14)/DD on SPY/QQQ (fail-closed).")
    lines.append("- Base model trained once per year per base strategy; overlays only change entries.")
    lines.append("- Hard-stop cooldown and ATR tight caps target WR without retuning turbo knobs randomly.")
    lines.append("- STYLE-US control (`turbo_highvol_minalloc`) remains the paper baseline unless gates pass.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Crash entry mega study")
    ap.add_argument("--smoke", action="store_true", help="Tiny window/universe for CI (keeps --grid if set)")
    ap.add_argument(
        "--grid",
        choices=(
            "smoke",
            "medium",
            "full",
            "week",
            "curated",
            "week_risk",
            "risk_ab",
            "alt_mdd",
            "alt_loop",
            "alt_mdd_v2",
            "alt_loop_v2",
        ),
        default=None,
        help=(
            "Config grid (week/curated = Phase A 5; week_risk = Phase C A/B; "
            "alt_mdd = MDD combos; alt_mdd_v2 = yearly/soft recovery)"
        ),
    )
    ap.add_argument("--first-oos", type=int, default=None)
    ap.add_argument("--last-oos", type=int, default=None)
    ap.add_argument(
        "--universe-limit",
        type=int,
        default=None,
        help="Max tickers with data; 0 = full file (no cap). Default depends on grid.",
    )
    ap.add_argument(
        "--ticker-file",
        type=Path,
        default=ROOT / "universe_highvol80.txt",
    )
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "crash_entry_mega_study",
    )
    ap.add_argument("--min-train-rows", type=int, default=3000)
    args = ap.parse_args()

    # --smoke alone → smoke grid; with --grid week keep curated configs but shrink window
    if args.smoke and args.grid is None:
        grid = "smoke"
    else:
        grid = args.grid or ("smoke" if args.smoke else "medium")

    weekish = grid in ("week", "curated", "week_risk", "risk_ab")
    altish = grid in ("alt_mdd", "alt_loop", "alt_mdd_v2", "alt_loop_v2")
    if args.smoke:
        first = args.first_oos or 2020
        last = args.last_oos or 2020
        # Explicit 0 means full file even under smoke (caller intent)
        if args.universe_limit is not None:
            univ = args.universe_limit
        else:
            univ = 15
        min_rows = min(args.min_train_rows, 1500)
    elif weekish:
        first = args.first_oos or 2018
        last = args.last_oos or 2025
        # Week plan default: full highvol80 (0 = no cap)
        univ = 0 if args.universe_limit is None else args.universe_limit
        min_rows = args.min_train_rows
    elif altish:
        first = args.first_oos or 2018
        last = args.last_oos or 2025
        # Alt MDD: medium multi-year default n=40 (honest vs smoke; faster than full 80)
        univ = 40 if args.universe_limit is None else args.universe_limit
        min_rows = args.min_train_rows
    else:
        first = args.first_oos or 2018
        last = args.last_oos or 2025
        univ = 40 if args.universe_limit is None else args.universe_limit
        min_rows = args.min_train_rows

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ticker_file = args.ticker_file
    if not ticker_file.is_file():
        ticker_file = ROOT / "good_tickers_wf80.txt"

    summary = run_mega(
        data_root=args.data_root,
        ticker_file=ticker_file,
        universe_limit=univ,
        first_oos=first,
        last_oos=last,
        grid=grid,
        out_dir=out_dir,
        min_train_rows=min_rows,
    )

    # JSON: drop huge year dumps from all_rows for master file? keep compact
    compact = deepcopy(summary)
    for r in compact.get("all_rows") or []:
        r.pop("year_results", None)
    for r in compact.get("top_by_composite") or []:
        r.pop("year_results", None)
    for r in compact.get("baselines") or []:
        r.pop("year_results", None)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, default=str)
    _write_summary_md(summary, out_dir / "SUMMARY.md")
    logger.info("Wrote %s and SUMMARY.md", out_dir / "summary.json")

    top = (summary.get("top_by_composite") or [{}])[0]
    print(
        f"DONE grid={grid} top={top.get('id')} "
        f"CAGR={_safe_float(top.get('cagr'))*100:.1f}% "
        f"WR={_safe_float(top.get('win_rate'))*100:.1f}% "
        f"composite={_safe_float(top.get('composite')):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
