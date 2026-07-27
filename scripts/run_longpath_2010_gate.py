"""Single path longhist OOS 2010–2025 with gates CAGR>10% and MDD≥−65%.

Primary: turbo_highvol_minalloc. Multi-year table + LOYO on same equity.
Research only. Does not change paper freeze.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.backtest import BacktestConfig  # noqa: E402
from trad_research.metrics import equity_metrics  # noqa: E402
from trad_research.risk_metrics import extended_risk_from_equity  # noqa: E402
from trad_research.strategies import get_strategy  # noqa: E402
from trad_research.strategy_runner import run_strategy_walk_forward  # noqa: E402
from trad_research.walk_forward import load_benchmark_equity  # noqa: E402

COMMISSION = 0.001
SLIPPAGE = 0.0005
GATE_CAGR = 0.10
GATE_MDD = -0.65  # max drawdown must be >= this (not deeper)
GATE_TRADES = 100


def _eq_norm(s: pd.Series) -> pd.Series:
    out = s.dropna().astype(float)
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.duplicated(keep="last")].dropna().sort_index()
    return out


def _metrics(eq: pd.Series, trades: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    eq = _eq_norm(eq)
    if eq.empty:
        return {"error": "empty"}
    start = float(eq.iloc[0])
    tdf = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame()
    rep = equity_metrics(eq, start_equity=start, trades=tdf if not tdf.empty else None)
    risk = extended_risk_from_equity(
        eq.to_numpy(),
        trade_pnls=tdf["net_profit"].to_numpy()
        if not tdf.empty and "net_profit" in tdf.columns
        else None,
    )
    return {
        "cagr": float(rep.cagr),
        "sharpe": float(rep.sharpe),
        "sortino": float(risk.sortino),
        "max_drawdown": float(rep.max_drawdown),
        "n_trades": int(rep.n_trades),
        "win_rate": float(rep.win_rate) if rep.win_rate is not None else None,
        "total_return": float(eq.iloc[-1] / start - 1.0),
        "start": str(eq.index.min()),
        "end": str(eq.index.max()),
        "n_bars": int(len(eq)),
    }


def _spy_excess(eq: pd.Series, data_root: Path) -> Optional[float]:
    try:
        b = load_benchmark_equity(
            data_root, eq.index.min(), eq.index.max(), preferred=["SPY"]
        )
        if b is None or b.empty:
            return None
        eq2 = _eq_norm(eq)
        b = _eq_norm(b)
        j = pd.concat([eq2.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
        if len(j) < 5:
            return None
        return float(j["s"].iloc[-1] / j["s"].iloc[0] - j["b"].iloc[-1] / j["b"].iloc[0])
    except Exception:
        return None


def _year_table(eq: pd.Series, trades: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    eq = _eq_norm(eq)
    rows = []
    for y, seg in eq.groupby(eq.index.year):
        if len(seg) < 3:
            continue
        ret = float(seg.iloc[-1] / float(seg.iloc[0]) - 1.0)
        peak = seg.cummax()
        dd = float((seg / peak - 1.0).min())
        n_tr = 0
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            t = trades.copy()
            if "oos_year" in t.columns:
                n_tr = int((pd.to_numeric(t["oos_year"], errors="coerce") == int(y)).sum())
            elif "entry_date" in t.columns:
                ed = pd.to_datetime(t["entry_date"], utc=True, errors="coerce")
                n_tr = int((ed.dt.year == int(y)).sum())
        rows.append(
            {
                "year": int(y),
                "return": ret,
                "max_drawdown": dd,
                "n_trades": n_tr,
                "green": ret > 0,
            }
        )
    return rows


def _slice_metrics(eq: pd.Series, y0: int, y1: int) -> Dict[str, Any]:
    eq = _eq_norm(eq)
    years = eq.index.year
    seg = eq[(years >= y0) & (years <= y1)]
    m = _metrics(seg)
    m["window"] = f"{y0}-{y1}"
    return m


def _loyo(eq: pd.Series, drop_year: int) -> Dict[str, Any]:
    eq = _eq_norm(eq)
    seg = eq[eq.index.year != int(drop_year)]
    m = _metrics(seg)
    m["drop_year"] = int(drop_year)
    return m


def _gates(m: Dict[str, Any]) -> Dict[str, Any]:
    cagr = float(m.get("cagr") or 0.0)
    mdd = float(m.get("max_drawdown") or -1.0)
    n = int(m.get("n_trades") or 0)
    g_cagr = cagr > GATE_CAGR
    g_mdd = mdd >= GATE_MDD
    g_tr = n >= GATE_TRADES
    return {
        "cagr_ok": g_cagr,
        "mdd_ok": g_mdd,
        "trades_ok": g_tr,
        "pass": bool(g_cagr and g_mdd and g_tr),
        "thresholds": {
            "cagr_gt": GATE_CAGR,
            "mdd_ge": GATE_MDD,
            "n_trades_ge": GATE_TRADES,
        },
    }


def run_path(
    name: str,
    *,
    first: int,
    last: int,
    data_root: Path,
    ticker_file: Path,
    universe_limit: int,
    min_train_rows: int,
) -> Dict[str, Any]:
    strat = get_strategy(name)
    if hasattr(strat, "universe_source_file"):
        strat.universe_source_file = str(ticker_file)
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
            ticker_file=ticker_file,
            universe_limit=universe_limit,
            first_oos_year=int(first),
            last_oos_year=int(last),
            min_train_rows=int(min_train_rows),
            preferred_index=["SPY", "QQQ"],
            base_bt=BacktestConfig(commission=COMMISSION, slippage=SLIPPAGE),
        )
    finally:
        if orig is not None:
            strat.backtest_overrides = orig  # type: ignore[method-assign]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", type=str, default="turbo_highvol_minalloc")
    ap.add_argument(
        "--extra-strategies",
        type=str,
        default="",
        help="Comma strategies if primary fails or always-run controls",
    )
    ap.add_argument("--always-controls", action="store_true")
    ap.add_argument("--first", type=int, default=2010)
    ap.add_argument("--last", type=int, default=2025)
    ap.add_argument(
        "--ticker-file",
        type=Path,
        default=ROOT / "universe_longhist100.txt",
    )
    ap.add_argument("--universe-limit", type=int, default=80)
    ap.add_argument("--min-train-rows", type=int, default=1500)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "longpath_2010",
    )
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    ticker_file = Path(args.ticker_file)
    # Prefer longhist 2010-passers only when the user asked for the longhist100 file
    # (do not override an explicit highvol / custom universe path).
    default_longhist = (ROOT / "universe_longhist100.txt").resolve()
    passers = ROOT / "universe_longhist2010_pass.txt"
    try:
        tf_res = ticker_file.resolve()
    except Exception:
        tf_res = ticker_file
    if tf_res == default_longhist and passers.is_file():
        n_pass = sum(
            1
            for ln in passers.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        )
        if n_pass >= 40:
            ticker_file = passers
            print(f"Using longhist 2010 passers n={n_pass}: {ticker_file}", flush=True)

    names = [args.strategy.strip()]
    extras = [s.strip() for s in str(args.extra_strategies).split(",") if s.strip()]
    if args.always_controls:
        for s in extras:
            if s not in names:
                names.append(s)

    results: List[Dict[str, Any]] = []
    primary_pass = False

    for name in names:
        print(
            f"[path] {name} OOS {args.first}-{args.last} "
            f"univ={ticker_file.name} limit={args.universe_limit} …",
            flush=True,
        )
        res = run_path(
            name,
            first=int(args.first),
            last=int(args.last),
            data_root=data_root,
            ticker_file=ticker_file,
            universe_limit=int(args.universe_limit),
            min_train_rows=int(args.min_train_rows),
        )
        eq = res.get("equity")
        tr = res.get("trades")
        if not isinstance(eq, pd.Series) or eq.empty:
            print(f"  EMPTY equity for {name}", flush=True)
            results.append({"name": name, "error": "empty_equity"})
            continue
        eq = _eq_norm(eq)
        tdf = tr if isinstance(tr, pd.DataFrame) else pd.DataFrame()
        # tag oos year if missing
        if not tdf.empty and "oos_year" not in tdf.columns and "entry_date" in tdf.columns:
            tdf = tdf.copy()
            tdf["oos_year"] = pd.to_datetime(
                tdf["entry_date"], utc=True, errors="coerce"
            ).dt.year

        m = _metrics(eq, tdf)
        m["excess_spy_total"] = _spy_excess(eq, data_root)
        gates = _gates(m)
        years = _year_table(eq, tdf)
        green_frac = (
            float(np.mean([1.0 if y["green"] else 0.0 for y in years])) if years else 0.0
        )
        multi = {
            "by_year": years,
            "green_frac": green_frac,
            "sub_2010_2017": _slice_metrics(eq, 2010, 2017),
            "sub_2018_2025": _slice_metrics(eq, 2018, 2025),
            "loyo_drop_2020": _loyo(eq, 2020),
            "loyo_drop_2022": _loyo(eq, 2022),
        }
        multi_ok = green_frac >= 0.55 or float(
            multi["loyo_drop_2020"].get("cagr") or 0
        ) > 0.05

        eq.to_csv(out / f"equity_{name}.csv", header=["equity"])
        if not tdf.empty:
            tdf.to_csv(out / f"trades_{name}.csv", index=False)
        row = {
            "name": name,
            "metrics": m,
            "gates": gates,
            "multi_year": multi,
            "multi_year_soft_ok": multi_ok,
            "verdict": "PASS" if gates["pass"] else "FAIL",
        }
        (out / f"metrics_{name}.json").write_text(
            json.dumps(row, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"  CAGR={m.get('cagr'):.2%} MDD={m.get('max_drawdown'):.2%} "
            f"n={m.get('n_trades')} gates={gates['pass']} "
            f"green={green_frac:.0%}",
            flush=True,
        )
        results.append(row)
        if name == args.strategy.strip():
            primary_pass = bool(gates["pass"])
            # if primary fails and controls not already scheduled, run extras
            if not primary_pass and extras and not args.always_controls:
                for s in extras:
                    if s not in names:
                        names.append(s)

    # SUMMARY
    lines = [
        "# Long path 2010–2025 — gate CAGR>10% · MDD≥−65%",
        "",
        "> **Research only.** Not financial advice. Paper freeze unchanged.",
        "",
        f"- OOS: **{args.first}–{args.last}**",
        f"- Universe: `{ticker_file}` limit={args.universe_limit}",
        f"- Costs: commission {COMMISSION} + slippage {SLIPPAGE}",
        f"- Primary: `{args.strategy}`",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Path results",
        "",
        "| strategy | CAGR | Sharpe | MDD | n_trades | excess SPY | gates | multi soft |",
        "|----------|------|--------|-----|----------|------------|-------|------------|",
    ]
    for r in results:
        if r.get("error"):
            lines.append(f"| `{r.get('name')}` | ERROR | — | — | — | — | FAIL | — |")
            continue
        m = r["metrics"]
        xs = m.get("excess_spy_total")
        xs_s = f"{100*float(xs):.0f}%" if xs is not None else "n/a"
        lines.append(
            f"| `{r['name']}` | {100*float(m.get('cagr') or 0):.1f}% | "
            f"{float(m.get('sharpe') or 0):.2f} | {100*float(m.get('max_drawdown') or 0):.1f}% | "
            f"{m.get('n_trades')} | {xs_s} | "
            f"{'**PASS**' if r['gates']['pass'] else 'FAIL'} | "
            f"{'OK' if r.get('multi_year_soft_ok') else 'warn'} |"
        )

    # primary detail
    prim = next((r for r in results if r.get("name") == args.strategy.strip()), None)
    if prim and not prim.get("error"):
        lines += ["", f"## Multi-year — `{args.strategy}`", "", "| year | return | MDD | n | green |", "|------|--------|-----|---|-------|"]
        for y in prim["multi_year"]["by_year"]:
            lines.append(
                f"| {y['year']} | {100*y['return']:.1f}% | {100*y['max_drawdown']:.1f}% | "
                f"{y['n_trades']} | {'Y' if y['green'] else 'N'} |"
            )
        my = prim["multi_year"]
        lines += [
            "",
            f"- Green frac: **{100*float(my['green_frac']):.0f}%**",
            f"- Sub 2010–17 CAGR: **{100*float(my['sub_2010_2017'].get('cagr') or 0):.1f}%** "
            f"MDD {100*float(my['sub_2010_2017'].get('max_drawdown') or 0):.1f}%",
            f"- Sub 2018–25 CAGR: **{100*float(my['sub_2018_2025'].get('cagr') or 0):.1f}%** "
            f"MDD {100*float(my['sub_2018_2025'].get('max_drawdown') or 0):.1f}%",
            f"- LOYO drop 2020 CAGR: **{100*float(my['loyo_drop_2020'].get('cagr') or 0):.1f}%**",
            f"- LOYO drop 2022 CAGR: **{100*float(my['loyo_drop_2022'].get('cagr') or 0):.1f}%**",
        ]

    lines += [
        "",
        "## Decision",
        "",
        f"- Primary gates: **{'PASS' if primary_pass else 'FAIL'}**",
        "- Paper freeze: **turbo_highvol_minalloc** unchanged",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    decision = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "primary": args.strategy,
        "primary_pass": primary_pass,
        "paper_freeze": "turbo_highvol_minalloc",
        "results": results,
        "disclaimer": "Research only. Not financial advice.",
    }
    (out / "summary.json").write_text(
        json.dumps(decision, indent=2, default=str), encoding="utf-8"
    )

    # DECISION short
    dlines = [
        "# Long path 2010 — Decision",
        "",
        f"**Primary:** `{args.strategy}` → **{'PASS' if primary_pass else 'FAIL'}**",
        "",
    ]
    if prim and not prim.get("error"):
        m = prim["metrics"]
        dlines += [
            f"- CAGR: **{100*float(m.get('cagr') or 0):.1f}%** (gate >10%)",
            f"- MDD: **{100*float(m.get('max_drawdown') or 0):.1f}%** (gate ≥ −65%)",
            f"- Trades: **{m.get('n_trades')}**",
            f"- Multi soft: **{'OK' if prim.get('multi_year_soft_ok') else 'warn'}**",
            "",
        ]
    dlines += [
        "**Paper freeze unchanged.**",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "DECISION.md").write_text("\n".join(dlines), encoding="utf-8")
    print(f"Wrote {out / 'SUMMARY.md'} primary_pass={primary_pass}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
