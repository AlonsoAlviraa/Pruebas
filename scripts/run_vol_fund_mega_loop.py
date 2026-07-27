"""Multi-hour vol∩fund mega loop: screen 2018–21 → freeze → confirm 2022–25.

Research only. Resume via PROGRESS.json. Does not touch paper freeze.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.backtest import BacktestConfig  # noqa: E402
from trad_research.growth_universe import GrowthGateConfig  # noqa: E402
from trad_research.metrics import equity_metrics  # noqa: E402
from trad_research.risk_levers import LEVERS, RiskMddLever  # noqa: E402
from trad_research.risk_metrics import extended_risk_from_equity  # noqa: E402
from trad_research.strategies import get_strategy  # noqa: E402
from trad_research.strategy_runner import run_strategy_walk_forward  # noqa: E402
from trad_research.universe import write_ticker_file  # noqa: E402
from trad_research.vol_fund_l0 import (  # noqa: E402
    growth_l0_from_pool,
    highvol_pool_asof,
    write_year_l0,
)
from trad_research.walk_forward import load_benchmark_equity  # noqa: E402

COMMISSION = 0.001
SLIPPAGE = 0.0005


@dataclass
class GridConfig:
    config_id: str
    strategy: str
    growth_hard: bool = True
    growth_top_k: int = 40
    lever_id: str = "baseline"
    vol_pool_n: int = 200
    vol_only_top: int = 80  # when growth_hard=False
    label: str = ""
    # Loop F: soft-ban tickers (excluded from L0; top-up from remainder of pool)
    exclude_tickers: Tuple[str, ...] = field(default_factory=tuple)
    # Optional extra backtest overrides merged after risk lever
    extra_bt: Dict[str, Any] = field(default_factory=dict)


def loop_d_grid() -> List[GridConfig]:
    """Loop D: only turbo_highvol_minalloc + risk levers, NO growth gate.

    Confirm winner family from vol_fund_mega: vol-only minalloc k80.
    """
    base = "turbo_highvol_minalloc"
    # Prefer yearly peak + milder packs (alt_mdd_v2 lessons); include baseline control
    lever_ids = [
        "baseline",
        "vt60_only",
        "vol_target_tight_70",
        "dd_circuit_25",
        "dd25_vt70",
        "dd25_vt70_yr",
        "dd25_vt70_soft",
        "dd20_vt60",
        "dd18_vt70_pos75",
        "dd35_vt80_yr",
    ]
    # Also k60 / k100 vol-only tops as mild L0 sensitivity (still no growth)
    tops = (60, 80, 100)
    out: List[GridConfig] = []
    for top in tops:
        for lid in lever_ids:
            if lid not in LEVERS:
                continue
            # only full lever set on k80 (confirm default); thinner set on k60/k100
            if top != 80 and lid not in (
                "baseline",
                "dd35_vt80_yr",
                "dd25_vt70_yr",
                "vt60_only",
                "dd18_vt70_pos75",
            ):
                continue
            out.append(
                GridConfig(
                    config_id=f"{base}__volonly_k{top}_{lid}",
                    strategy=base,
                    growth_hard=False,
                    growth_top_k=top,
                    lever_id=lid,
                    vol_only_top=top,
                    vol_pool_n=200,
                    label="loop_d_minalloc_risk",
                )
            )
    return out


def core_grid() -> List[GridConfig]:
    """~50–70 configs: bases × growth_k × few levers (not 452 clones)."""
    out: List[GridConfig] = []
    bases = [
        "turbo_highvol_minalloc",
        "turbo_highvol",
        "growth_ew",
        "growth_trend_mom",
        "growth_turbo_minalloc",
    ]
    # Loop A style: vol-only controls
    for base in ("turbo_highvol_minalloc", "turbo_highvol"):
        out.append(
            GridConfig(
                config_id=f"{base}__volonly_k80_baseline",
                strategy=base,
                growth_hard=False,
                growth_top_k=80,
                lever_id="baseline",
                vol_only_top=80,
                label="vol_only",
            )
        )
    # Loop B: growth hard
    levers = ["baseline", "dd_circuit_25", "dd25_vt70"]
    # yearly soft from LEVERS if present
    for lid in list(LEVERS.keys()):
        if "yearly" in lid or lid in ("dd35_vt80_yr",):
            levers.append(lid)
    levers = list(dict.fromkeys(levers))[:6]

    for base in bases:
        for k in (40, 60, 80):
            for lid in levers:
                # skip turbo_highvol without minalloc × heavy levers to save time
                if base == "turbo_highvol" and lid not in ("baseline", "dd_circuit_25"):
                    continue
                if base == "growth_ew" and lid not in ("baseline", "dd_circuit_25", "dd25_vt70"):
                    continue
                cid = f"{base}__gh1_k{k}_{lid}"
                out.append(
                    GridConfig(
                        config_id=cid,
                        strategy=base,
                        growth_hard=True,
                        growth_top_k=k,
                        lever_id=lid if lid in LEVERS else "baseline",
                        label="growth_hard",
                    )
                )
    # de-dupe
    seen = set()
    uniq = []
    for c in out:
        if c.config_id in seen:
            continue
        seen.add(c.config_id)
        uniq.append(c)
    return uniq


def _apply_lever(overrides: Dict[str, Any], lever_id: str) -> Dict[str, Any]:
    """Apply risk lever; prefer shared apply_risk_mdd_lever when available."""
    try:
        from trad_research.risk_levers import apply_risk_mdd_lever

        o = apply_risk_mdd_lever(overrides, lever_id)
    except Exception:
        o = dict(overrides)
        lev = LEVERS.get(lever_id) or LEVERS["baseline"]
        assert isinstance(lev, RiskMddLever)
        o["max_portfolio_dd"] = float(lev.max_portfolio_dd)
        o["dd_soft_scale"] = float(lev.dd_soft_scale)
        if lev.dd_breach_size_scale is not None:
            o["dd_breach_size_scale"] = float(lev.dd_breach_size_scale)
        o["dd_peak_mode"] = str(lev.peak_mode)
        vt = float(o.get("volatility_target_pct") or 0.04)
        o["volatility_target_pct"] = vt * float(lev.vol_target_scale)
        mp = float(o.get("max_position_pct") or 0.22)
        o["max_position_pct"] = mp * float(lev.max_position_scale)
        if lev.risk_off_scale is not None:
            o["risk_off_scale"] = float(lev.risk_off_scale)
    # peak_mode field name used by backtest if present
    lev2 = LEVERS.get(lever_id) or LEVERS["baseline"]
    o["dd_peak_mode"] = str(getattr(lev2, "peak_mode", "continuous"))
    o["peak_mode"] = o["dd_peak_mode"]
    o["commission"] = COMMISSION
    o["slippage"] = SLIPPAGE
    return o


def _stitch(segments: List[pd.Series]) -> pd.Series:
    if not segments:
        return pd.Series(dtype=float)
    parts = []
    prev_end = None
    for seg in segments:
        s = seg.dropna().astype(float)
        if s.empty:
            continue
        if prev_end is not None and float(s.iloc[0]) != 0:
            s = s * (prev_end / float(s.iloc[0]))
        parts.append(s)
        prev_end = float(s.iloc[-1])
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")].sort_index()


def _spy_excess(eq: pd.Series, data_root: Path) -> Optional[float]:
    try:
        b = load_benchmark_equity(
            data_root, eq.index.min(), eq.index.max(), preferred=["SPY"]
        )
        if b is None or b.empty:
            return None
        eq2 = eq.copy()
        eq2.index = pd.to_datetime(eq2.index, utc=True).normalize()
        eq2 = eq2[~eq2.index.duplicated(keep="last")]
        b = b.copy()
        b.index = pd.to_datetime(b.index, utc=True).normalize()
        b = b[~b.index.duplicated(keep="last")]
        j = pd.concat([eq2.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
        if len(j) < 5:
            return None
        st = float(j["s"].iloc[-1] / j["s"].iloc[0] - 1.0)
        bt = float(j["b"].iloc[-1] / j["b"].iloc[0] - 1.0)
        return st - bt
    except Exception:
        return None


def _honest_score(row: Dict[str, Any]) -> float:
    resid = float(row.get("residual_cagr_vs_style") or 0.0)
    sortino = float(row.get("sortino") or 0.0)
    xs = float(row.get("excess_spy_total") or 0.0)
    # rough total→cagr soft: use total excess capped
    mdd = float(row.get("max_drawdown") or -1.0)
    score = 3.0 * resid + 2.0 * sortino + 1.0 * min(xs, 0.5)
    if mdd < -0.50:
        score -= 2.0 * ((-0.50) - mdd)
    return float(score)


def run_config_years(
    cfg: GridConfig,
    *,
    years: Sequence[int],
    data_root: Path,
    panel_file: Path,
    l0_cache: Path,
    static_pool: List[str],
    min_train_rows: int,
    use_dynamic_vol: bool,
) -> Dict[str, Any]:
    strat = get_strategy(cfg.strategy)
    segments: List[pd.Series] = []
    trades_all: List[pd.DataFrame] = []
    year_meta = []
    style_segments: List[pd.Series] = []

    for y in years:
        as_of = f"{int(y) - 1}-12-31"
        if use_dynamic_vol:
            pool = highvol_pool_asof(
                data_root, panel_file, as_of, n=int(cfg.vol_pool_n)
            )
        else:
            pool = list(static_pool)

        if cfg.growth_hard:
            l0, diag = growth_l0_from_pool(
                pool,
                data_root,
                as_of,
                top_k=int(cfg.growth_top_k),
            )
        else:
            # vol-only: top vol_only_top of pool (already vol-ranked if static highvol file order)
            # re-score as-of for honesty when dynamic; else first N of static
            if use_dynamic_vol:
                l0 = pool[: int(cfg.vol_only_top)]
            else:
                l0 = pool[: int(cfg.vol_only_top)]
            diag = {"pool": len(pool), "l0": len(l0), "pass_all": len(l0)}

        # Loop F soft-ban: drop excluded names and top-up from remaining pool order
        ban = {str(t).upper() for t in (cfg.exclude_tickers or ()) if t}
        if ban:
            keep = [t for t in l0 if str(t).upper() not in ban]
            need = int(cfg.vol_only_top if not cfg.growth_hard else cfg.growth_top_k)
            if len(keep) < need:
                for t in pool:
                    tu = str(t).upper()
                    if tu in ban or tu in {x.upper() for x in keep}:
                        continue
                    keep.append(t)
                    if len(keep) >= need:
                        break
            diag = {
                **diag,
                "l0_pre_ban": int(diag.get("l0", len(l0))),
                "banned": sorted(ban),
                "l0": len(keep),
            }
            l0 = keep

        if len(l0) < 8:
            year_meta.append({"year": y, "error": "l0_small", **diag})
            continue

        yfile = write_year_l0(l0_cache, y, l0, tag=cfg.config_id.replace("/", "_")[:40])

        base_ov = strat.backtest_overrides() if hasattr(strat, "backtest_overrides") else {}
        merged = _apply_lever(base_ov, cfg.lever_id)
        if cfg.extra_bt:
            merged.update(dict(cfg.extra_bt))

        def _ov() -> Dict[str, Any]:
            return dict(merged)

        orig = getattr(strat, "backtest_overrides", None)
        if orig is not None:
            strat.backtest_overrides = _ov  # type: ignore[method-assign]
        try:
            res = run_strategy_walk_forward(
                strat,
                data_root=data_root,
                ticker_file=yfile,
                universe_limit=max(len(l0), 10),
                first_oos_year=int(y),
                last_oos_year=int(y),
                min_train_rows=min_train_rows,
                preferred_index=["SPY", "QQQ"],
                base_bt=BacktestConfig(commission=COMMISSION, slippage=SLIPPAGE),
            )
            # style EW same L0
            style = get_strategy("growth_ew")
            if hasattr(style, "backtest_overrides"):
                s_ov = _apply_lever(style.backtest_overrides(), "baseline")

                def _sov() -> Dict[str, Any]:
                    return dict(s_ov)

                s_orig = style.backtest_overrides
                style.backtest_overrides = _sov  # type: ignore[method-assign]
                try:
                    sres = run_strategy_walk_forward(
                        style,
                        data_root=data_root,
                        ticker_file=yfile,
                        universe_limit=max(len(l0), 10),
                        first_oos_year=int(y),
                        last_oos_year=int(y),
                        min_train_rows=min_train_rows,
                        preferred_index=["SPY"],
                        base_bt=BacktestConfig(commission=COMMISSION, slippage=SLIPPAGE),
                    )
                finally:
                    style.backtest_overrides = s_orig  # type: ignore[method-assign]
            else:
                sres = {}
        finally:
            if orig is not None:
                strat.backtest_overrides = orig  # type: ignore[method-assign]

        eq = res.get("equity")
        if eq is None or (hasattr(eq, "empty") and eq.empty):
            year_meta.append({"year": y, "error": "empty", **diag})
            continue
        eq = eq.dropna().astype(float)
        segments.append(eq)
        tr = res.get("trades")
        if isinstance(tr, pd.DataFrame) and not tr.empty:
            t2 = tr.copy()
            t2["oos_year"] = y
            trades_all.append(t2)
        yret = float(eq.iloc[-1] / float(eq.iloc[0]) - 1.0)
        year_meta.append({"year": y, "year_return": yret, "n_trades": int(len(tr) if tr is not None else 0), **diag})

        seq = sres.get("equity") if isinstance(sres, dict) else None
        if seq is not None and not (hasattr(seq, "empty") and seq.empty):
            style_segments.append(seq.dropna().astype(float))

    eq_all = _stitch(segments)
    if eq_all.empty:
        return {
            "config_id": cfg.config_id,
            "error": "no_equity",
            "years": year_meta,
            "strategy": cfg.strategy,
        }

    tdf = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    start = float(eq_all.iloc[0])
    rep = equity_metrics(eq_all, start_equity=start, trades=tdf if not tdf.empty else None)
    risk = extended_risk_from_equity(
        eq_all.to_numpy(),
        trade_pnls=tdf["net_profit"].to_numpy()
        if not tdf.empty and "net_profit" in tdf.columns
        else None,
    )
    total = float(eq_all.iloc[-1] / start - 1.0)
    xs = _spy_excess(eq_all, data_root)

    style_eq = _stitch(style_segments)
    residual = None
    if not style_eq.empty:
        s_start = float(style_eq.iloc[0])
        s_rep = equity_metrics(style_eq, start_equity=s_start)
        residual = float(rep.cagr) - float(s_rep.cagr)

    row = {
        "config_id": cfg.config_id,
        "strategy": cfg.strategy,
        "growth_hard": cfg.growth_hard,
        "growth_top_k": cfg.growth_top_k,
        "lever_id": cfg.lever_id,
        "exclude_tickers": list(cfg.exclude_tickers or ()),
        "extra_bt": dict(cfg.extra_bt or {}),
        "cagr": rep.cagr,
        "sharpe": rep.sharpe,
        "sortino": risk.sortino,
        "max_drawdown": rep.max_drawdown,
        "n_trades": rep.n_trades,
        "win_rate": rep.win_rate,
        "total_return": total,
        "excess_spy_total": xs,
        "residual_cagr_vs_style": residual,
        "years": year_meta,
    }
    row["honest_score"] = _honest_score(row)
    return {
        **row,
        "equity": eq_all,
        "trades": tdf,
    }


def save_progress(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=ROOT / "universe_highvol200.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--first-screen", type=int, default=2018)
    ap.add_argument("--last-screen", type=int, default=2021)
    ap.add_argument("--first-confirm", type=int, default=2022)
    ap.add_argument("--last-confirm", type=int, default=2025)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--min-train-rows", type=int, default=2500)
    ap.add_argument("--dynamic-vol", action="store_true")
    ap.add_argument("--top-freeze", type=int, default=15)
    ap.add_argument("--phase", choices=["screen", "confirm", "all"], default="all")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "vol_fund_mega",
    )
    ap.add_argument("--max-configs", type=int, default=0, help="0=all core grid")
    ap.add_argument(
        "--grid",
        choices=["core", "loop_d"],
        default="core",
        help="core=vol+growth mega; loop_d=minalloc vol-only + risk levers only",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    prog_path = out / "PROGRESS.json"
    screen_dir = out / "screen"
    confirm_dir = out / "confirm"
    l0_cache = out / "l0_cache"
    screen_dir.mkdir(exist_ok=True)
    confirm_dir.mkdir(exist_ok=True)

    panel_file = Path(args.panel)
    static_pool = [
        ln.strip().upper()
        for ln in panel_file.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    print(f"Panel {panel_file} n={len(static_pool)}", flush=True)

    configs = loop_d_grid() if args.grid == "loop_d" else core_grid()
    if int(args.max_configs) > 0:
        configs = configs[: int(args.max_configs)]
    print(f"Grid mode={args.grid} configs={len(configs)}", flush=True)

    t0 = time.time()
    deadline = t0 + float(args.hours) * 3600.0

    state: Dict[str, Any] = {
        "started": datetime.now(timezone.utc).isoformat(),
        "panel": str(panel_file),
        "n_panel": len(static_pool),
        "n_configs": len(configs),
        "screen_done": [],
        "confirm_done": [],
        "screen_rows": [],
        "confirm_rows": [],
        "stop_reason": None,
    }
    if prog_path.is_file():
        try:
            prev = json.loads(prog_path.read_text(encoding="utf-8"))
            state["screen_done"] = list(prev.get("screen_done") or [])
            state["confirm_done"] = list(prev.get("confirm_done") or [])
            state["screen_rows"] = list(prev.get("screen_rows") or [])
            state["confirm_rows"] = list(prev.get("confirm_rows") or [])
            print(
                f"Resume: screen_done={len(state['screen_done'])} confirm_done={len(state['confirm_done'])}",
                flush=True,
            )
        except Exception as e:
            print(f"PROGRESS read fail: {e}", flush=True)

    screen_years = list(range(int(args.first_screen), int(args.last_screen) + 1))
    confirm_years = list(range(int(args.first_confirm), int(args.last_confirm) + 1))

    # --- SCREEN ---
    if args.phase in ("screen", "all"):
        for i, cfg in enumerate(configs, 1):
            if time.time() > deadline:
                state["stop_reason"] = "hours_budget_screen"
                break
            if cfg.config_id in state["screen_done"]:
                continue
            print(f"[screen {i}/{len(configs)}] {cfg.config_id}", flush=True)
            r = run_config_years(
                cfg,
                years=screen_years,
                data_root=Path(args.data_root),
                panel_file=panel_file,
                l0_cache=l0_cache,
                static_pool=static_pool,
                min_train_rows=int(args.min_train_rows),
                use_dynamic_vol=bool(args.dynamic_vol),
            )
            # persist equity
            safe = cfg.config_id.replace("/", "_")
            if isinstance(r.get("equity"), pd.Series):
                cfg_dir = screen_dir / "configs" / safe
                cfg_dir.mkdir(parents=True, exist_ok=True)
                r["equity"].to_csv(cfg_dir / "equity.csv", header=["equity"])
                (cfg_dir / "metrics.json").write_text(
                    json.dumps({k: v for k, v in r.items() if k not in ("equity", "trades")}, indent=2, default=str),
                    encoding="utf-8",
                )
            row = {k: v for k, v in r.items() if k not in ("equity", "trades")}
            state["screen_rows"].append(row)
            state["screen_done"].append(cfg.config_id)
            save_progress(prog_path, state)
            print(
                f"  cagr={row.get('cagr')} sortino={row.get('sortino')} mdd={row.get('max_drawdown')} "
                f"resid={row.get('residual_cagr_vs_style')} score={row.get('honest_score')}",
                flush=True,
            )

    # freeze top
    screen_ok = [r for r in state["screen_rows"] if not r.get("error")]
    screen_ok.sort(key=lambda x: float(x.get("honest_score") or -9e9), reverse=True)
    freeze = screen_ok[: int(args.top_freeze)]
    freeze_ids = [r["config_id"] for r in freeze]
    (out / "screen_rank.json").write_text(json.dumps(screen_ok, indent=2, default=str), encoding="utf-8")
    (out / "freeze_top.json").write_text(json.dumps(freeze, indent=2, default=str), encoding="utf-8")
    print(f"Freeze top {len(freeze)}: {freeze_ids[:5]}…", flush=True)

    # --- CONFIRM ---
    if args.phase in ("confirm", "all") and freeze:
        cfg_by_id = {c.config_id: c for c in configs}
        for i, fid in enumerate(freeze_ids, 1):
            if time.time() > deadline:
                state["stop_reason"] = state.get("stop_reason") or "hours_budget_confirm"
                break
            if fid in state["confirm_done"]:
                continue
            cfg = cfg_by_id.get(fid)
            if cfg is None:
                continue
            print(f"[confirm {i}/{len(freeze_ids)}] {fid}", flush=True)
            r = run_config_years(
                cfg,
                years=confirm_years,
                data_root=Path(args.data_root),
                panel_file=panel_file,
                l0_cache=l0_cache,
                static_pool=static_pool,
                min_train_rows=int(args.min_train_rows),
                use_dynamic_vol=bool(args.dynamic_vol),
            )
            safe = fid.replace("/", "_")
            if isinstance(r.get("equity"), pd.Series):
                cfg_dir = confirm_dir / "configs" / safe
                cfg_dir.mkdir(parents=True, exist_ok=True)
                r["equity"].to_csv(cfg_dir / "equity.csv", header=["equity"])
                (cfg_dir / "metrics.json").write_text(
                    json.dumps({k: v for k, v in r.items() if k not in ("equity", "trades")}, indent=2, default=str),
                    encoding="utf-8",
                )
            row = {k: v for k, v in r.items() if k not in ("equity", "trades")}
            state["confirm_rows"].append(row)
            state["confirm_done"].append(fid)
            save_progress(prog_path, state)
            print(
                f"  CONFIRM cagr={row.get('cagr')} resid={row.get('residual_cagr_vs_style')} mdd={row.get('max_drawdown')}",
                flush=True,
            )

    if not state.get("stop_reason"):
        state["stop_reason"] = "completed"
    state["finished"] = datetime.now(timezone.utc).isoformat()
    state["elapsed_sec"] = time.time() - t0
    save_progress(prog_path, state)

    confirm_ok = [r for r in state["confirm_rows"] if not r.get("error")]
    confirm_ok.sort(key=lambda x: float(x.get("honest_score") or -9e9), reverse=True)
    (out / "confirm_rank.json").write_text(json.dumps(confirm_ok, indent=2, default=str), encoding="utf-8")

    # SUMMARY
    lines = [
        "# Vol∩Fund mega loop SUMMARY",
        "",
        "> **Research only.** Not financial advice.",
        "",
        f"- Panel: `{panel_file.name}` n={len(static_pool)}",
        f"- Screen: {args.first_screen}–{args.last_screen} · Confirm: {args.first_confirm}–{args.last_confirm}",
        f"- Configs grid: {len(configs)} · stop: `{state['stop_reason']}` · elapsed_h={state['elapsed_sec']/3600:.2f}",
        "",
        "## Screen top 10 (honest_score)",
        "",
        "| rank | config | CAGR | Sortino | MDD | resid | score |",
        "|------|--------|------|---------|-----|-------|-------|",
    ]
    for i, r in enumerate(screen_ok[:10], 1):
        lines.append(
            f"| {i} | `{r.get('config_id')}` | {100*float(r.get('cagr') or 0):.1f}% | "
            f"{float(r.get('sortino') or 0):.2f} | {100*float(r.get('max_drawdown') or 0):.1f}% | "
            f"{100*float(r.get('residual_cagr_vs_style') or 0):.1f}pp | {float(r.get('honest_score') or 0):.2f} |"
        )
    lines += [
        "",
        "## Confirm ranking",
        "",
        "| rank | config | CAGR | Sortino | MDD | resid | score |",
        "|------|--------|------|---------|-----|-------|-------|",
    ]
    for i, r in enumerate(confirm_ok[:15], 1):
        lines.append(
            f"| {i} | `{r.get('config_id')}` | {100*float(r.get('cagr') or 0):.1f}% | "
            f"{float(r.get('sortino') or 0):.2f} | {100*float(r.get('max_drawdown') or 0):.1f}% | "
            f"{100*float(r.get('residual_cagr_vs_style') or 0):.1f}pp | {float(r.get('honest_score') or 0):.2f} |"
        )
    lines += [
        "",
        "## Decision gates",
        "",
        "- Success B: residual>0 or excess SPY>0, MDD better than −55%, n_trades≥200, confirm not pathology.",
        "- Paper freeze: **unchanged** without human ADVANCE.",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'SUMMARY.md'} stop={state['stop_reason']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
