"""YTD bake-off of named strategies (refresh EODHD then WF one OOS year).

Default: calendar year = current year (or --year), universe highvol80.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.data.eodhd_client import fetch_eod, get_token
from trad_research.metrics import equity_metrics
from trad_research.risk_metrics import extended_risk_from_equity, sortino_ratio
from trad_research.strategies import get_strategy
from trad_research.strategy_runner import run_strategy_walk_forward
from trad_research.walk_forward import load_benchmark_equity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ytd_bakeoff")

DEFAULT_STRATS = [
    "turbo_highvol_minalloc",
    "turbo_highvol_minalloc_sector_rot",
    "turbo_highvol_minalloc_softreg",
    "turbo_strict_adaptive",
    "champion_ml",
    "turbo_highvol",
    "turbo_strict",
    "aggressive_turbo",
]


def _merge_history(path: Path, new_df: pd.DataFrame) -> int:
    """Merge new EOD into existing history CSV; return bars after merge."""
    new_df = new_df.copy()
    new_df["date"] = pd.to_datetime(new_df["date"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        if c in new_df.columns:
            new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
    new_df = new_df.dropna(subset=["date", "close"]).sort_values("date")
    cols = ["date", "open", "high", "low", "close", "volume"]
    new_df = new_df[[c for c in cols if c in new_df.columns]]
    if path.is_file():
        old = pd.read_csv(path)
        old.columns = [c.lower().strip() for c in old.columns]
        old["date"] = pd.to_datetime(old["date"], utc=True, errors="coerce")
        for c in ("open", "high", "low", "close", "volume"):
            if c in old.columns:
                old[c] = pd.to_numeric(old[c], errors="coerce")
        old = old.dropna(subset=["date", "close"])
        both = pd.concat([old[cols], new_df[cols]], ignore_index=True)
    else:
        both = new_df[cols]
    both = both.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    path.parent.mkdir(parents=True, exist_ok=True)
    both.to_csv(path, index=False)
    return len(both)


def refresh_tickers(
    tickers: List[str],
    *,
    data_root: Path,
    start: str,
    end: str,
    sleep_s: float = 0.12,
) -> Dict[str, Any]:
    import time

    tok = get_token()
    ok = fail = 0
    details = {}
    for t in tickers:
        key = t.upper().strip()
        path = data_root / f"{key}_history.csv"
        try:
            df = fetch_eod(key, start=start, end=end, token=tok)
            if df.empty:
                fail += 1
                details[key] = "empty"
                logger.warning("empty %s", key)
            else:
                n = _merge_history(path, df)
                ok += 1
                details[key] = f"ok bars={n} max={df['date'].max()}"
                logger.info("refreshed %s -> %s", key, details[key])
        except Exception as e:  # noqa: BLE001
            fail += 1
            details[key] = f"error:{e}"
            logger.warning("fail %s: %s", key, e)
        time.sleep(sleep_s)
    return {"ok": ok, "fail": fail, "details": details}


def _bench_total(equity: pd.Series, data_root: Path, name: str) -> Optional[float]:
    try:
        b = load_benchmark_equity(
            data_root, equity.index.min(), equity.index.max(), preferred=[name]
        )
        if b is None or b.empty:
            return None
        b.index = pd.to_datetime(b.index, utc=True).normalize()
        eq = equity.copy()
        eq.index = pd.to_datetime(eq.index, utc=True).normalize()
        j = pd.concat([eq.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
        if len(j) < 3:
            return None
        return float(j["b"].iloc[-1] / j["b"].iloc[0] - 1.0)
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=date.today().year)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--ticker-file", type=Path, default=ROOT / "universe_highvol80.txt")
    ap.add_argument("--universe-limit", type=int, default=80)
    ap.add_argument("--strategies", type=str, default=",".join(DEFAULT_STRATS))
    ap.add_argument("--skip-refresh", action="store_true")
    ap.add_argument("--refresh-start", type=str, default="2024-01-01")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "ytd_model_bakeoff",
    )
    ap.add_argument("--min-train-rows", type=int, default=3000)
    args = ap.parse_args(argv)

    year = int(args.year)
    today = date.today().isoformat()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    ticker_file = Path(args.ticker_file)
    if not ticker_file.is_absolute():
        ticker_file = ROOT / ticker_file

    tickers = [
        ln.strip().upper()
        for ln in ticker_file.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ][: int(args.universe_limit)]
    # indices for benches + regime
    extra = ["SPY", "QQQ"]
    all_dl = list(dict.fromkeys(tickers + extra))

    refresh_meta: Dict[str, Any] = {"skipped": True}
    if not args.skip_refresh:
        logger.info("Refreshing %d symbols %s → %s", len(all_dl), args.refresh_start, today)
        refresh_meta = refresh_tickers(
            all_dl, data_root=data_root, start=args.refresh_start, end=today
        )
        refresh_meta["skipped"] = False

    # data coverage after refresh
    max_dates = []
    for t in tickers:
        p = data_root / f"{t}_history.csv"
        if not p.is_file():
            continue
        d = pd.read_csv(p, usecols=lambda c: str(c).lower() == "date")
        col = [c for c in d.columns if str(c).lower() == "date"][0]
        mx = pd.to_datetime(d[col], utc=True, errors="coerce").max()
        if pd.notna(mx):
            max_dates.append(mx)
    coverage = {
        "n_tickers_with_data": len(max_dates),
        "max_bar": str(max(max_dates)) if max_dates else None,
        "min_of_max_bar": str(min(max_dates)) if max_dates else None,
        "n_with_year": sum(1 for d in max_dates if d.year >= year),
    }
    logger.info("coverage %s", coverage)

    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    rows: List[Dict[str, Any]] = []
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    eq_dir = out / "equity"
    eq_dir.mkdir(exist_ok=True)

    for name in names:
        logger.info("=== running %s OOS %s ===", name, year)
        try:
            strat = get_strategy(name)
            # force same universe file
            if hasattr(strat, "universe_source_file"):
                strat.universe_source_file = str(ticker_file)
            res = run_strategy_walk_forward(
                strat,
                data_root=data_root,
                ticker_file=ticker_file,
                universe_limit=int(args.universe_limit),
                first_oos_year=year,
                last_oos_year=year,
                min_train_rows=int(args.min_train_rows),
                preferred_index=["SPY", "QQQ"],
            )
            eq = res.get("equity")
            if eq is None or (hasattr(eq, "empty") and eq.empty):
                rows.append({"name": name, "error": "empty equity", "ok": False})
                continue
            eq = eq.dropna().astype(float)
            eq.to_csv(eq_dir / f"{name}.csv", header=["equity"])
            start_eq = float(eq.iloc[0])
            rep = equity_metrics(eq, start_equity=start_eq, trades=res.get("trades"))
            rets = eq.pct_change().dropna()
            risk = extended_risk_from_equity(eq.to_numpy())
            spy_tot = _bench_total(eq, data_root, "SPY")
            qqq_tot = _bench_total(eq, data_root, "QQQ")
            total = float(eq.iloc[-1] / start_eq - 1.0)
            row = {
                "name": name,
                "ok": True,
                "year": year,
                "start": str(eq.index.min()),
                "end": str(eq.index.max()),
                "n_days": int(len(eq)),
                "total_return": total,
                "cagr": rep.cagr,
                "sharpe": rep.sharpe,
                "sortino": risk.sortino,
                "max_drawdown": rep.max_drawdown,
                "calmar": risk.calmar,
                "n_trades": rep.n_trades,
                "win_rate": rep.win_rate,
                "profit_factor": rep.profit_factor,
                "spy_total_return": spy_tot,
                "qqq_total_return": qqq_tot,
                "excess_vs_spy": (total - spy_tot) if spy_tot is not None else None,
                "excess_vs_qqq": (total - qqq_tot) if qqq_tot is not None else None,
                "year_results": res.get("year_results"),
            }
            rows.append(row)
            logger.info(
                "%s total=%.1f%% sharpe=%.2f mdd=%.1f%% vsSPY=%s trades=%s",
                name,
                total * 100,
                rep.sharpe,
                rep.max_drawdown * 100,
                row["excess_vs_spy"],
                rep.n_trades,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("fail %s", name)
            rows.append({"name": name, "ok": False, "error": str(e)})

    # rank ok rows by total return
    ok_rows = [r for r in rows if r.get("ok")]
    ok_rows.sort(key=lambda r: float(r.get("total_return") or -9e9), reverse=True)

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "year": year,
        "asof_today": today,
        "ticker_file": str(ticker_file),
        "universe_limit": args.universe_limit,
        "coverage": coverage,
        "refresh": {k: refresh_meta[k] for k in ("ok", "fail", "skipped") if k in refresh_meta},
        "ranking": ok_rows,
        "all": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# YTD model bake-off — {year}",
        "",
        f"**As-of (calendar):** {today} · **Data max bar (universe):** {coverage.get('max_bar')}",
        f"**Universe:** `{ticker_file.name}` limit={args.universe_limit} · n with year {year} data: {coverage.get('n_with_year')}",
        f"**Refresh EODHD:** ok={refresh_meta.get('ok')} fail={refresh_meta.get('fail')} skipped={refresh_meta.get('skipped')}",
        "",
        "Walk-forward: train ends at year-start; OOS = calendar year bars available in cache.",
        "",
        "| Rank | Strategy | Total ret | Sharpe | Sortino | MDD | Trades | vs SPY | vs QQQ | Window |",
        "|------|----------|-----------|--------|---------|-----|--------|--------|--------|--------|",
    ]
    for i, r in enumerate(ok_rows, 1):
        lines.append(
            f"| {i} | `{r['name']}` | **{r['total_return']:.2%}** | {r['sharpe']:.2f} | "
            f"{r['sortino']:.2f} | {r['max_drawdown']:.2%} | {r['n_trades']} | "
            f"{(r['excess_vs_spy'] if r['excess_vs_spy'] is not None else float('nan')):.2%} | "
            f"{(r['excess_vs_qqq'] if r['excess_vs_qqq'] is not None else float('nan')):.2%} | "
            f"{str(r['start'])[:10]}→{str(r['end'])[:10]} |"
        )
    fails = [r for r in rows if not r.get("ok")]
    if fails:
        lines += ["", "## Failures", ""]
        for r in fails:
            lines.append(f"- `{r['name']}`: {r.get('error')}")
    lines += [
        "",
        "## Notes",
        "",
        "- Total return is over the OOS equity span (not annualized if partial year).",
        "- vs SPY/QQQ = strategy total − index total on overlapping dates.",
        "- Research only. Not financial advice.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out / 'SUMMARY.md'}")
    return 0 if ok_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
