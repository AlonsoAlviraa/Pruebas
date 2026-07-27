"""Run S1–S5 growth strategies on yearly PIT growth L0; report financial metrics.

Rebuilds growth universe each year (as-of prior year-end) when --rebuild-universe.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.backtest import BacktestConfig  # noqa: E402
from trad_research.growth_universe import GrowthGateConfig, build_growth_universe  # noqa: E402
from trad_research.metrics import equity_metrics  # noqa: E402
from trad_research.risk_metrics import extended_risk_from_equity  # noqa: E402
from trad_research.strategies import get_strategy  # noqa: E402
from trad_research.strategy_runner import run_strategy_walk_forward  # noqa: E402
from trad_research.universe import write_ticker_file  # noqa: E402
from trad_research.walk_forward import load_benchmark_equity  # noqa: E402

GROWTH_STRATS = [
    "growth_ew",
    "growth_trend_mom",
    "growth_cs_mom",
    "growth_turbo_minalloc",
    "growth_quality_strict",
]

COMMISSION = 0.001
SLIPPAGE = 0.0005


def _bench_total(equity: pd.Series, data_root: Path, name: str) -> Optional[float]:
    try:
        b = load_benchmark_equity(
            data_root, equity.index.min(), equity.index.max(), preferred=[name]
        )
        if b is None or b.empty:
            return None
        b = b.copy()
        b.index = pd.to_datetime(b.index, utc=True).normalize()
        eq = equity.copy()
        eq.index = pd.to_datetime(eq.index, utc=True).normalize()
        j = pd.concat([eq.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
        if len(j) < 3:
            return None
        return float(j["b"].iloc[-1] / j["b"].iloc[0] - 1.0)
    except Exception:
        return None


def _stitch(segments: List[pd.Series]) -> pd.Series:
    if not segments:
        return pd.Series(dtype=float)
    parts = []
    cap_scale = 1.0
    prev_end = None
    for seg in segments:
        s = seg.dropna().astype(float)
        if s.empty:
            continue
        if prev_end is not None and float(s.iloc[0]) != 0:
            # scale so continuity of capital
            cap_scale = prev_end / float(s.iloc[0])
        s = s * cap_scale
        parts.append(s)
        prev_end = float(s.iloc[-1])
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def run_strategy_yearly_growth(
    name: str,
    *,
    years: List[int],
    data_root: Path,
    scan_file: Path,
    cfg: GrowthGateConfig,
    univ_dir: Path,
    min_train_rows: int,
    rebuild: bool,
    limit_scan: Optional[int],
) -> Dict[str, Any]:
    strat = get_strategy(name)
    segments: List[pd.Series] = []
    all_trades: List[pd.DataFrame] = []
    year_meta: List[Dict[str, Any]] = []
    l0_sizes: Dict[int, int] = {}

    for y in years:
        as_of = f"{y - 1}-12-31"
        yfile = univ_dir / f"universe_growth_top{cfg.top_n}_{y}.txt"
        if rebuild or not yfile.is_file():
            top, rows = build_growth_universe(
                data_root,
                scan_file,
                as_of,
                cfg=cfg,
                limit_scan=limit_scan,
            )
            write_ticker_file(yfile, top)
            n_pass = sum(1 for r in rows if r.pass_all)
        else:
            top = [
                ln.strip().upper()
                for ln in yfile.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
            n_pass = len(top)
        l0_sizes[y] = len(top)
        if len(top) < 10:
            year_meta.append({"year": y, "error": "l0_too_small", "n": len(top), "n_pass": n_pass})
            continue

        if hasattr(strat, "universe_source_file"):
            strat.universe_source_file = str(yfile)
        base_ov = strat.backtest_overrides() if hasattr(strat, "backtest_overrides") else {}
        merged = {**base_ov, "commission": COMMISSION, "slippage": SLIPPAGE}

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
                universe_limit=max(len(top), int(cfg.top_n)),
                first_oos_year=y,
                last_oos_year=y,
                min_train_rows=min_train_rows,
                preferred_index=["SPY", "QQQ"],
                base_bt=BacktestConfig(commission=COMMISSION, slippage=SLIPPAGE),
            )
        finally:
            if orig is not None:
                strat.backtest_overrides = orig  # type: ignore[method-assign]

        eq = res.get("equity")
        tr = res.get("trades")
        if eq is None or (hasattr(eq, "empty") and eq.empty):
            year_meta.append({"year": y, "error": "empty_equity", "n_l0": len(top)})
            continue
        eq = eq.dropna().astype(float)
        segments.append(eq)
        if isinstance(tr, pd.DataFrame) and not tr.empty:
            t2 = tr.copy()
            t2["oos_year"] = y
            all_trades.append(t2)
        start_eq = float(eq.iloc[0])
        yret = float(eq.iloc[-1] / start_eq - 1.0) if start_eq else float("nan")
        year_meta.append(
            {
                "year": y,
                "n_l0": len(top),
                "n_pass_scan": n_pass,
                "year_return": yret,
                "n_trades": int(len(tr)) if isinstance(tr, pd.DataFrame) else 0,
            }
        )
        print(
            f"  {name} {y}: L0={len(top)} ret={yret:.1%} trades={year_meta[-1]['n_trades']}",
            flush=True,
        )

    eq_all = _stitch(segments)
    if eq_all.empty:
        return {"strategy": name, "error": "no_equity", "years": year_meta, "l0_sizes": l0_sizes}

    tdf = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    start_eq = float(eq_all.iloc[0])
    rep = equity_metrics(eq_all, start_equity=start_eq, trades=tdf if not tdf.empty else None)
    risk = extended_risk_from_equity(
        eq_all.to_numpy(),
        trade_pnls=tdf["net_profit"].to_numpy()
        if not tdf.empty and "net_profit" in tdf.columns
        else None,
    )
    total = float(eq_all.iloc[-1] / start_eq - 1.0)
    spy = _bench_total(eq_all, data_root, "SPY")
    qqq = _bench_total(eq_all, data_root, "QQQ")
    return {
        "strategy": name,
        "total_return": total,
        "cagr": rep.cagr,
        "sharpe": rep.sharpe,
        "sortino": risk.sortino,
        "max_drawdown": rep.max_drawdown,
        "calmar": risk.calmar,
        "n_trades": rep.n_trades,
        "win_rate": rep.win_rate,
        "profit_factor": rep.profit_factor,
        "excess_spy": (total - spy) if spy is not None else None,
        "excess_qqq": (total - qqq) if qqq is not None else None,
        "spy_total": spy,
        "qqq_total": qqq,
        "years": year_meta,
        "l0_sizes": l0_sizes,
        "equity": eq_all,
        "trades": tdf,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-oos", type=int, default=2022)
    ap.add_argument("--last-oos", type=int, default=2025)
    ap.add_argument("--strategies", type=str, default=",".join(GROWTH_STRATS))
    ap.add_argument("--ticker-file", type=Path, default=ROOT / "good_tickers_filtrados.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--top-n", type=int, default=80)
    ap.add_argument("--limit-scan", type=int, default=0)
    ap.add_argument("--min-train-rows", type=int, default=2000)
    ap.add_argument("--rebuild-universe", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "growth_eodhd_battery",
    )
    args = ap.parse_args(argv)

    years = list(range(int(args.first_oos), int(args.last_oos) + 1))
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    univ_dir = out / "universes"
    univ_dir.mkdir(exist_ok=True)

    cfg = GrowthGateConfig(top_n=int(args.top_n))
    limit = None if int(args.limit_scan) <= 0 else int(args.limit_scan)
    names = [s.strip() for s in args.strategies.split(",") if s.strip()]

    results: List[Dict[str, Any]] = []
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name} years={years[0]}-{years[-1]}", flush=True)
        r = run_strategy_yearly_growth(
            name,
            years=years,
            data_root=Path(args.data_root),
            scan_file=Path(args.ticker_file),
            cfg=cfg,
            univ_dir=univ_dir,
            min_train_rows=int(args.min_train_rows),
            rebuild=bool(args.rebuild_universe),
            limit_scan=limit,
        )
        # residual vs growth_ew if present later
        results.append(r)
        safe = name.replace("/", "_")
        if isinstance(r.get("equity"), pd.Series):
            r["equity"].to_csv(out / f"equity_{safe}.csv", header=["equity"])
        if isinstance(r.get("trades"), pd.DataFrame) and not r["trades"].empty:
            r["trades"].to_csv(out / f"trades_{safe}.csv", index=False)
        print(
            f"  → cagr={r.get('cagr')} sharpe={r.get('sharpe')} mdd={r.get('max_drawdown')} "
            f"vsSPY={r.get('excess_spy')}",
            flush=True,
        )

    # residual vs S1
    s1 = next((x for x in results if x.get("strategy") == "growth_ew" and "cagr" in x), None)
    for r in results:
        if s1 and "cagr" in r and r.get("cagr") is not None and s1.get("cagr") is not None:
            r["residual_cagr_vs_growth_ew"] = float(r["cagr"]) - float(s1["cagr"])
        else:
            r["residual_cagr_vs_growth_ew"] = None

    # serializable summary
    ser = []
    for r in results:
        ser.append(
            {
                k: v
                for k, v in r.items()
                if k not in ("equity", "trades") and not isinstance(v, (pd.Series, pd.DataFrame))
            }
        )
    # rank by sortino then residual
    ok = [x for x in ser if "error" not in x]
    ranked = sorted(
        ok,
        key=lambda x: (float(x.get("sortino") or -9), float(x.get("residual_cagr_vs_growth_ew") or -9)),
        reverse=True,
    )
    for i, r in enumerate(ranked, 1):
        r["rank"] = i

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "first_oos": years[0],
        "last_oos": years[-1],
        "top_n": cfg.top_n,
        "gates": {"min_eps_q_yoy": cfg.min_eps_q_yoy, "min_eps_ttm_yoy": cfg.min_eps_ttm_yoy},
        "commission": COMMISSION,
        "slippage": SLIPPAGE,
        "disclaimer": "Research only. Not financial advice.",
        "strategies": ranked + [x for x in ser if "error" in x],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Growth EODHD strategy battery",
        "",
        "> **Research only.** Not financial advice.",
        "",
        f"- OOS: **{years[0]}–{years[-1]}**",
        f"- Gates: Q EPS YoY ≥ {cfg.min_eps_q_yoy:.0%}, annual ≥ {cfg.min_eps_ttm_yoy:.0%} (EPS TTM / rev fallback)",
        f"- Top-N: **{cfg.top_n}** · costs {COMMISSION:.2%}+{SLIPPAGE:.2%}",
        "",
        "| Rank | Strategy | CAGR | Sharpe | Sortino | MDD | vs SPY | Resid vs EW | Trades | WR |",
        "|------|----------|------|--------|---------|-----|--------|-------------|--------|-----|",
    ]
    for r in ranked:
        lines.append(
            f"| {r.get('rank')} | `{r['strategy']}` | "
            f"{100 * float(r.get('cagr') or 0):.1f}% | "
            f"{float(r.get('sharpe') or 0):.2f} | "
            f"{float(r.get('sortino') or 0):.2f} | "
            f"{100 * float(r.get('max_drawdown') or 0):.1f}% | "
            f"{100 * float(r.get('excess_spy') or 0):.1f}% | "
            f"{100 * float(r.get('residual_cagr_vs_growth_ew') or 0):.1f}pp | "
            f"{r.get('n_trades')} | "
            f"{100 * float(r.get('win_rate') or 0):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Residual vs `growth_ew` (S1) = selection/timing alpha over the growth filter itself.",
            "- Paper freeze stays `turbo_highvol_minalloc` until promotion ADVANCE.",
            "- If fund depth is short, start OOS later (see coverage report).",
            "",
            "Research only. Not financial advice.",
            "",
        ]
    )
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'SUMMARY.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
