"""ALPHA-PORTABLE v0 evaluation harness (STR-02/05).

Compares a portable L1/L2 path vs style_clone residual on US OOS.

**L1 modes**
* ``rule_rank`` — weighted CS-rank heuristic (ablation control).
* ``residual_train`` — yearly expanding WF logistic on beat_style residual
  labels (invariant ranks only; horizon embargo; train past only).

**Honesty flags (v0):**
* Portable path is a **lightweight CS engine** (ME rebalance, no commissions /
  stops) vs style clone on full ``strategy_runner`` → ``engine_mismatch=True``,
  residual is **diagnostic_only**; R1 promotion ``pass_core`` forced False.
* R1 is **modern-window only** unless both early and modern excesses are passed
  into ``promotion_gates_residual(require_early_and_modern=True)``.
* R2 PIT not evaluated here → incomplete.

Does **not** retune turbo_highvol_* knobs. STYLE-US baseline is control only.

Usage::

    python scripts/run_redesign_eval.py --smoke
    python scripts/run_redesign_eval.py --l1-mode residual_train --first-oos 2022 --last-oos 2024

Research only. Not financial advice.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.alpha_attribution import (  # noqa: E402
    ResidualReport,
    compare_to_benchmark,
    promotion_gates_residual,
)
from trad_research.features import engineer_m2_features, list_tickers, load_history  # noqa: E402
from trad_research.metrics import equity_metrics  # noqa: E402
from trad_research.portable.cs_features import (  # noqa: E402
    INVARIANT_FEATURE_NAMES,
    assert_no_absolute_prices,
)
from trad_research.portable.membership_l0 import (  # noqa: E402
    L0Config,
    rebalance_dates,
    select_members,
)
from trad_research.portable.portfolio_l2 import (  # noqa: E402
    PortfolioL2Config,
    build_weight_panel,
    equity_from_returns,
    portfolio_returns_from_weights,
)
from trad_research.portable.score_l1 import (  # noqa: E402
    ResidualTrainConfig,
    score_panel_l1,
    walk_forward_residual_scores,
)
from trad_research.strategies import RESEARCH_BASELINE_US, get_strategy  # noqa: E402
from trad_research.strategy_runner import run_strategy_walk_forward  # noqa: E402
from trad_research.walk_forward import load_benchmark_equity  # noqa: E402

logger = logging.getLogger("redesign_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _resolve(p: Path) -> Path:
    """Resolve relative paths against repo ROOT (not CWD)."""
    p = Path(p)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _build_long_panel(
    tickers: List[str],
    data_root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load engineered M2 features into a long panel (date, ticker, …)."""
    rows = []
    for t in tickers:
        try:
            raw = load_history(t, data_root)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        try:
            feat = engineer_m2_features(raw)
        except Exception:
            continue
        if feat is None or feat.empty:
            continue
        df = feat.copy()
        if "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df.loc[(df["date"] >= start) & (df["date"] <= end)]
        if df.empty:
            continue
        close = pd.to_numeric(df["close"], errors="coerce")
        ret_1d = close.pct_change().shift(-1)
        piece = df.copy()
        piece["ticker"] = str(t).upper()
        piece["ret_1d"] = ret_1d.to_numpy()
        rows.append(piece)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _me_score_panel(panel: pd.DataFrame, pool: List[str]) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Restrict **scoring** to ME rebalance dates (static L0 pool).

    Returns (me_panel_for_scoring, full_calendar of original panel dates).
    Weights are held across days in ``run_portable_v0`` via ffill.
    """
    if panel.empty:
        return panel, pd.DatetimeIndex([])
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    out["_dnorm"] = out["date"].dt.normalize()
    dates = pd.DatetimeIndex(out["_dnorm"].unique()).sort_values()
    rb = rebalance_dates(dates, freq="ME")
    if len(rb) == 0:
        rb = dates
    cfg = L0Config(require_pit_listed=False, max_names=None)
    allowed: Dict[pd.Timestamp, set] = {}
    for d in rb:
        snap = select_members(pool, d, config=cfg, membership=None)
        allowed[pd.Timestamp(d).normalize()] = set(snap.members)
    rb_norm = {pd.Timestamp(x).normalize() for x in rb}
    me = out.loc[out["_dnorm"].isin(rb_norm)].copy()
    keep = []
    for d, g in me.groupby("_dnorm", sort=False):
        mem = allowed.get(d) or {str(t).upper() for t in pool}
        keep.append(g.loc[g["ticker"].astype(str).str.upper().isin(mem)])
    me = pd.concat(keep, ignore_index=True) if keep else me.iloc[0:0]
    me = me.drop(columns=["_dnorm"], errors="ignore")
    return me, dates


def run_portable_v0(
    panel: pd.DataFrame,
    pool: List[str],
    *,
    top_k: int = 8,
    top_quantile: float = 0.25,
    start_equity: float = 100_000.0,
    use_me_rebalance: bool = True,
    l1_mode: str = "rule_rank",
    first_oos: Optional[int] = None,
    last_oos: Optional[int] = None,
    residual_horizon: int = 20,
    residual_model: str = "logistic",
    full_history_panel: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Portable L1 + top-K L2; ME rebalance **holds** weights between month-ends.

    ``l1_mode``:
      * rule_rank — heuristic CS ranks
      * residual_train — yearly WF logistic on beat_style labels

    Still a lightweight/no-cost diagnostic engine vs strategy_runner.
    """
    from trad_research.portable.portfolio_l2 import hold_weights_across_calendar

    feat_cols = [c for c in INVARIANT_FEATURE_NAMES if c in panel.columns]
    assert_no_absolute_prices(feat_cols)
    full = panel.copy()
    full["date"] = pd.to_datetime(full["date"], utc=True, errors="coerce")
    mode = (l1_mode or "rule_rank").strip().lower()
    l1_meta: Dict[str, Any] = {"l1_mode": mode}
    pool_u = {str(t).upper() for t in pool}
    calendar = pd.DatetimeIndex(full["date"].dt.normalize().unique()).sort_values()

    if mode == "residual_train":
        # CRITICAL: residual WF labels/scores on **daily** (panel-native) bars.
        # Horizon H is bar-count; ME thinning before L1 would make H≈months.
        # ME rebalance is applied only after daily L1 for L2 weights.
        hist = full_history_panel if full_history_panel is not None else full
        hist = hist.copy()
        hist["date"] = pd.to_datetime(hist["date"], utc=True, errors="coerce")
        hist = hist.loc[hist["ticker"].astype(str).str.upper().isin(pool_u)].copy()
        fo = int(first_oos) if first_oos is not None else int(full["date"].dt.year.min())
        lo = int(last_oos) if last_oos is not None else int(full["date"].dt.year.max())
        cfg = ResidualTrainConfig(
            horizon=int(residual_horizon),
            model=str(residual_model),
            top_quantile=float(top_quantile),
        )
        scored_daily, l1_meta = walk_forward_residual_scores(
            hist,
            first_oos_year=fo,
            last_oos_year=lo,
            config=cfg,
            date_col="date",
        )
        l1_meta = dict(l1_meta or {})
        l1_meta["me_applied_before_l1"] = False
        l1_meta["l2_rebalance"] = "ME_hold" if use_me_rebalance else "daily"
        # Keep only OOS window present in original panel
        if not scored_daily.empty and not full.empty:
            d0, d1 = full["date"].min(), full["date"].max()
            scored_daily = scored_daily.loc[
                (scored_daily["date"] >= d0) & (scored_daily["date"] <= d1)
            ].copy()
        # L2: optional ME subsample of **already scored** daily panel
        if use_me_rebalance and not scored_daily.empty:
            score_panel, calendar = _me_score_panel(scored_daily, pool)
            scored = score_panel
        else:
            scored = scored_daily
            calendar = pd.DatetimeIndex(
                scored_daily["date"].dt.normalize().unique()
            ).sort_values() if not scored_daily.empty else calendar
        name = "alpha_portable_v0_residual_train"
        feat_cols = list(l1_meta.get("feature_cols") or feat_cols)
    else:
        if use_me_rebalance:
            score_panel, calendar = _me_score_panel(full, pool)
        else:
            score_panel = full
        scored = score_panel_l1(score_panel, date_col="date", top_quantile=top_quantile)
        name = "alpha_portable_v0_rule_rank"
        l1_meta = {"l1_mode": "rule_rank"}

    l2 = PortfolioL2Config(
        top_k=top_k,
        equal_weight=True,
        max_weight=0.25,
        score_col="l1_score",
        signal_col="l1_signal",
    )
    me_weights = build_weight_panel(scored, config=l2) if not scored.empty else scored
    if use_me_rebalance and isinstance(me_weights, pd.DataFrame) and not me_weights.empty:
        # Hold ME weights across full daily calendar of OOS panel
        calendar = pd.DatetimeIndex(full["date"].dt.normalize().unique()).sort_values()
        weights = hold_weights_across_calendar(me_weights, calendar)
    else:
        weights = me_weights if isinstance(me_weights, pd.DataFrame) else pd.DataFrame()
    # Daily returns from full panel (not ME-only)
    rets_panel = full[["date", "ticker", "ret_1d"]].copy()
    rets_panel["date"] = pd.to_datetime(rets_panel["date"], utc=True, errors="coerce")
    if weights is None or (isinstance(weights, pd.DataFrame) and weights.empty):
        port_rets = pd.Series(dtype=float)
        equity = pd.Series(dtype=float)
        rep = None
    else:
        port_rets = portfolio_returns_from_weights(weights, rets_panel, ret_col="ret_1d")
        port_rets = port_rets.replace([np.inf, -np.inf], np.nan).dropna()
        equity = equity_from_returns(port_rets, start_equity=start_equity)
        rep = equity_metrics(equity, start_equity=start_equity) if len(equity) > 5 else None
    return {
        "name": name,
        "equity": equity,
        "port_rets": port_rets,
        "report": rep.to_dict() if rep is not None else {},
        "n_weight_rows": len(weights) if isinstance(weights, pd.DataFrame) else 0,
        "n_me_weight_rows": len(me_weights) if isinstance(me_weights, pd.DataFrame) else 0,
        "feature_cols": feat_cols,
        "top_k": top_k,
        "top_quantile": top_quantile,
        "l1_meta": l1_meta,
        "l0": {
            "rebalance": "ME_hold" if use_me_rebalance else "daily_panel",
            "require_pit_listed": False,
            "pool_n": len(pool),
            "weights_held_between_me": bool(use_me_rebalance),
        },
        "engine": "cs_lightweight_no_costs",
    }


def run_style_control(
    name: str,
    *,
    data_root: Path,
    ticker_file: Path,
    universe_limit: int,
    first_oos: int,
    last_oos: int,
) -> Dict[str, Any]:
    strat = get_strategy(name)
    if hasattr(strat, "universe_source_file"):
        strat.universe_source_file = str(ticker_file)
    res = run_strategy_walk_forward(
        strat,
        data_root=data_root,
        ticker_file=ticker_file,
        universe_limit=universe_limit,
        first_oos_year=first_oos,
        last_oos_year=last_oos,
        use_pit_membership=False,
    )
    eq = res.get("equity")
    if eq is not None and not isinstance(eq, pd.Series):
        eq = pd.Series(eq)
    rep = res.get("report")
    rep_d = rep.to_dict() if hasattr(rep, "to_dict") else (rep or {})
    return {"name": name, "equity": eq, "report": rep_d, "engine": "strategy_runner"}


def apply_smoke_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply --smoke preset (unit-testable; used by parse_args/main)."""
    if getattr(args, "smoke", False):
        args.universe_limit = min(int(args.universe_limit), 15)
        args.first_oos = 2023
        args.last_oos = 2024
    return args


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ALPHA-PORTABLE v0 redesign eval")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--ticker-file", type=Path, default=ROOT / "universe_highvol80.txt")
    p.add_argument("--universe-limit", type=int, default=40)
    p.add_argument("--first-oos", type=int, default=2022)
    p.add_argument("--last-oos", type=int, default=2024)
    p.add_argument("--style-clone", type=str, default="style_ew_hv")
    p.add_argument("--baseline-control", type=str, default=RESEARCH_BASELINE_US)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--top-quantile", type=float, default=0.25)
    p.add_argument(
        "--l1-mode",
        type=str,
        default="rule_rank",
        choices=["rule_rank", "residual_train"],
        help="L1 scoring: rule_rank (control) or residual_train (WF beat_style)",
    )
    p.add_argument(
        "--residual-horizon",
        type=int,
        default=20,
        help="Forward horizon H for residual / beat_style labels",
    )
    p.add_argument(
        "--residual-model",
        type=str,
        default="logistic",
        choices=["logistic", "thin_ml", "xgb"],
        help="Model for residual_train mode",
    )
    p.add_argument("--run-baseline", action="store_true", help="Also run STYLE-US control (slow)")
    p.add_argument("--out", type=Path, default=ROOT / "reports/redesign/S2_portable_v0")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    args.data_root = _resolve(args.data_root)
    args.ticker_file = _resolve(args.ticker_file)
    args.out = _resolve(args.out)
    return apply_smoke_defaults(args)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.ticker_file.is_file() or not args.data_root.is_dir():
        print("ERROR: data root or ticker file missing", flush=True)
        return 2

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = list_tickers(args.ticker_file, args.data_root, limit=args.universe_limit)
    start = pd.Timestamp(f"{args.first_oos}-01-01", tz="UTC")
    end = pd.Timestamp(f"{args.last_oos}-12-31 23:59:59", tz="UTC")
    # residual_train needs multi-year past for expanding WF; rule_rank only needs warm-up
    if str(args.l1_mode).lower() == "residual_train":
        load_start = start - pd.Timedelta(days=365 * 4 + 60)
    else:
        load_start = start - pd.Timedelta(days=400)

    meta = {
        "product": "ALPHA-PORTABLE_v0",
        "l1": args.l1_mode,
        "l1_mode": args.l1_mode,
        "residual_horizon": int(args.residual_horizon),
        "residual_model": args.residual_model,
        "l2": f"top_k={args.top_k}_ew",
        "ticker_file": str(args.ticker_file),
        "universe_limit": args.universe_limit,
        "n_tickers": len(tickers),
        "first_oos": args.first_oos,
        "last_oos": args.last_oos,
        "style_clone": args.style_clone,
        "smoke": bool(args.smoke),
        "no_turbo_retune": True,
        "engine_mismatch": True,
        "diagnostic_only": True,
        "r1_scope": "modern_window_only_provisional",
        "l0": "ME rebalance on static pool; require_pit_listed=False",
        "note": (
            "Portable CS engine has no commissions/stops; residual vs style_runner "
            "is diagnostic_only — pass_core forced False."
        ),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[redesign_eval] meta={meta}", flush=True)

    print("[redesign_eval] building long panel…", flush=True)
    panel = _build_long_panel(tickers, args.data_root, start=load_start, end=end)
    if panel.empty:
        print("ERROR: empty panel", flush=True)
        return 3
    panel_oos = panel.loc[(panel["date"] >= start) & (panel["date"] <= end)].copy()
    print(f"  panel rows={len(panel_oos)} tickers~{panel_oos['ticker'].nunique()}", flush=True)

    # residual_train needs pre-OOS history for expanding labels/features
    portable = run_portable_v0(
        panel_oos,
        tickers,
        top_k=args.top_k,
        top_quantile=args.top_quantile,
        l1_mode=args.l1_mode,
        first_oos=args.first_oos,
        last_oos=args.last_oos,
        residual_horizon=args.residual_horizon,
        residual_model=args.residual_model,
        full_history_panel=panel if args.l1_mode == "residual_train" else None,
    )
    print(
        f"  portable CAGR={portable['report'].get('cagr')} "
        f"Sharpe={portable['report'].get('sharpe')} mode={args.l1_mode}",
        flush=True,
    )

    print(f"[redesign_eval] style control {args.style_clone}…", flush=True)
    style = run_style_control(
        args.style_clone,
        data_root=args.data_root,
        ticker_file=args.ticker_file,
        universe_limit=args.universe_limit,
        first_oos=args.first_oos,
        last_oos=args.last_oos,
    )
    print(
        f"  style CAGR={style['report'].get('cagr')} Sharpe={style['report'].get('sharpe')}",
        flush=True,
    )

    residual = None
    if portable.get("equity") is not None and style.get("equity") is not None:
        try:
            residual = compare_to_benchmark(
                portable["equity"],
                style["equity"],
                label="portable_vs_style_DIAGNOSTIC",
            ).to_dict()
            residual["diagnostic_only"] = True
            residual["engine_mismatch"] = True
        except Exception as exc:  # noqa: BLE001
            residual = {"error": str(exc)}

    gates = None
    if residual and "excess_cagr" in residual:
        rr = ResidualReport(
            strategy_cagr=float(residual.get("strategy_cagr") or 0),
            strategy_sharpe=float(residual.get("strategy_sharpe") or 0),
            strategy_mdd=float(residual.get("strategy_mdd") or 0),
            bench_cagr=float(residual.get("bench_cagr") or 0),
            bench_sharpe=float(residual.get("bench_sharpe") or 0),
            bench_mdd=float(residual.get("bench_mdd") or 0),
            excess_cagr=float(residual.get("excess_cagr") or 0),
            residual_sharpe=float(residual.get("residual_sharpe") or 0),
            label="portable_vs_style_DIAGNOSTIC",
        )
        gates = promotion_gates_residual(
            rr,
            residual_vs_pit_ew=None,
            engine_matched=False,
            diagnostic_only=True,
            require_early_and_modern=False,
        )
        assert gates["pass_core"] is False
        assert gates["R2_status"] == "not_evaluated"

    baseline_block = None
    if args.run_baseline:
        print(f"[redesign_eval] STYLE-US control {args.baseline_control}…", flush=True)
        baseline_block = run_style_control(
            args.baseline_control,
            data_root=args.data_root,
            ticker_file=args.ticker_file,
            universe_limit=args.universe_limit,
            first_oos=args.first_oos,
            last_oos=args.last_oos,
        )

    spy_block: Dict[str, Any] = {}
    peq = portable.get("equity")
    if peq is not None and len(peq) > 5:
        spy = load_benchmark_equity(
            args.data_root, peq.index.min(), peq.index.max(), preferred=["SPY"]
        )
        if spy is not None and not spy.empty:
            try:
                spy_block = compare_to_benchmark(peq, spy, label="portable_vs_spy").to_dict()
            except Exception as exc:  # noqa: BLE001
                spy_block = {"error": str(exc)}

    summary = {
        "meta": meta,
        "portable": {
            "name": portable["name"],
            "report": portable.get("report"),
            "n_weight_rows": portable.get("n_weight_rows"),
            "feature_cols": portable.get("feature_cols"),
            "l0": portable.get("l0"),
            "l1_meta": portable.get("l1_meta"),
            "engine": portable.get("engine"),
        },
        "style_clone": {
            "name": style["name"],
            "report": style.get("report"),
            "engine": style.get("engine"),
        },
        "residual_vs_style": residual,
        "promotion_gates": gates,
        "spy": spy_block,
        "baseline_control": None
        if baseline_block is None
        else {"name": baseline_block["name"], "report": baseline_block.get("report")},
        "l0_note": (
            "ME rebalance on static ticker pool (select_members, require_pit=False). "
            "Not full PIT membership. Cross-engine residual is diagnostic_only."
        ),
        "honesty": {
            "engine_mismatch": True,
            "diagnostic_only": True,
            "R2_status": (gates or {}).get("R2_status", "not_evaluated"),
            "pass_core": (gates or {}).get("pass_core", False),
        },
    }

    eq_dir = out_dir / "equity"
    eq_dir.mkdir(exist_ok=True)
    if peq is not None and len(peq):
        peq.rename("equity").to_csv(eq_dir / "alpha_portable_v0.csv", header=True)
    if style.get("equity") is not None:
        style["equity"].rename("equity").to_csv(
            eq_dir / f"{style['name']}.csv", header=True
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        "# S2 ALPHA-PORTABLE v0 eval (diagnostic)",
        "",
        f"**L1:** `{args.l1_mode}` · **L2:** top-{args.top_k} EW · **L0:** ME rebalance static pool",
        f"**Window:** {args.first_oos}–{args.last_oos} · L0 `{args.ticker_file}` n={len(tickers)}",
        "",
        "**engine_mismatch / diagnostic_only:** portable CS path ≠ strategy_runner costs.",
        "**R1 scope:** modern window only (provisional); dual-window not enforced here.",
        "**pass_core:** forced False until cost-matched engine + PIT R2.",
        "",
        f"- Portable CAGR/Sharpe: {portable['report'].get('cagr')} / {portable['report'].get('sharpe')}",
        f"- Style `{args.style_clone}` CAGR/Sharpe: {style['report'].get('cagr')} / {style['report'].get('sharpe')}",
        f"- Residual vs style (diagnostic): {residual}",
        f"- Promotion gates: {gates}",
        f"- L1 meta: {portable.get('l1_meta')}",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[redesign_eval] wrote {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
