"""Sistema A falsification: orb_htf_daily_proxy (EOD) kill-test.

Design: docs/design/2026-07-27_orb_htf_falsification.md
Research only. Does not change paper freeze.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.metrics import equity_metrics  # noqa: E402
from trad_research.monte_carlo import mc_bootstrap_trades, mc_shuffle_trades  # noqa: E402
from trad_research.risk_metrics import extended_risk_from_equity  # noqa: E402
from trad_research.strategies import OrbHtfDailyProxyStrategy, get_strategy  # noqa: E402
from trad_research.strategy_runner import run_strategy_walk_forward  # noqa: E402
from trad_research.walk_forward import load_benchmark_equity  # noqa: E402

COMMISSION = 0.001
SLIPPAGE = 0.0005
GATE_CAGR = 0.10
GATE_MDD = -0.65
GATE_TRADES = 80


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
        "start": str(eq.index.min().date()),
        "end": str(eq.index.max().date()),
        "n_bars": int(len(eq)),
    }


def _slice_eq(eq: pd.Series, start: str, end: str) -> pd.Series:
    eq = _eq_norm(eq)
    a = pd.Timestamp(start, tz="UTC")
    b = pd.Timestamp(end, tz="UTC")
    return eq[(eq.index >= a) & (eq.index <= b)]


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


def _mean_invested_weight(trades: pd.DataFrame, eq: pd.Series) -> float:
    """Rough: 1 - cash proxy from idle — use 1.0 if unknown; optional from capital."""
    # Without daily cash series, approximate exposure from open capital not available.
    # Return NaN-safe default: if many trades, assume partial; report as 1.0 placeholder.
    if eq is None or eq.empty:
        return 1.0
    return 1.0  # cash-aware blend skipped without exposure series


def _window_pack(
    eq: pd.Series, trades: pd.DataFrame, data_root: Path, start: str, end: str
) -> Dict[str, Any]:
    seg = _slice_eq(eq, start, end)
    if len(seg) < 10:
        return {"error": "short", "start": start, "end": end}
    tdf = trades
    if isinstance(trades, pd.DataFrame) and not trades.empty and "exit_date" in trades.columns:
        t = trades.copy()
        t["exit_date"] = pd.to_datetime(t["exit_date"], utc=True, errors="coerce")
        a = pd.Timestamp(start, tz="UTC")
        b = pd.Timestamp(end, tz="UTC")
        tdf = t[(t["exit_date"] >= a) & (t["exit_date"] <= b)]
    m = _metrics(seg, tdf)
    m["excess_spy_total"] = _spy_excess(seg, data_root)
    m["window"] = f"{start[:4]}-{end[:4]}"
    return m


def _mc_pack(trades: pd.DataFrame, n_sims: int = 2000, seed: int = 42) -> Dict[str, Any]:
    if trades is None or trades.empty or "net_profit" not in trades.columns:
        return {"error": "no_trades"}
    pnls = trades["net_profit"].to_numpy(dtype=float)
    pnls = pnls[np.isfinite(pnls)]
    if len(pnls) < 10:
        return {"error": "few_trades", "n": int(len(pnls))}
    sh = mc_shuffle_trades(pnls, n_sims=n_sims, seed=seed)
    bs = mc_bootstrap_trades(pnls, n_sims=n_sims, seed=seed + 1)
    return {
        "n_trades": int(len(pnls)),
        "shuffle_sortino_p5": float(sh.sortino_p5),
        "shuffle_mdd_p95": float(sh.mdd_p95),
        "bootstrap_sortino_p5": float(bs.sortino_p5),
        "bootstrap_mdd_p95": float(bs.mdd_p95),
        "prob_mdd_worse_60_bootstrap": float(
            bs.prob_mdd_worse_than.get("0.60", bs.prob_mdd_worse_than.get(0.6, float("nan")))
            if isinstance(bs.prob_mdd_worse_than, dict)
            else float("nan")
        ),
    }


def _decide(primary: Dict[str, Any], early: Dict[str, Any], modern: Dict[str, Any], mc: Dict[str, Any]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    label = "HOLD"

    def fget(d: Dict[str, Any], k: str, default=None):
        if not d or d.get("error"):
            return default
        return d.get(k, default)

    cagr = fget(primary, "cagr")
    mdd = fget(primary, "max_drawdown")
    ntr = fget(primary, "n_trades", 0) or 0
    sortino = fget(primary, "sortino")
    excess = fget(primary, "excess_spy_total")

    if ntr < GATE_TRADES:
        reasons.append(f"n_trades={ntr}<{GATE_TRADES}")
        label = "KILL"
    if mdd is not None and mdd < GATE_MDD:
        reasons.append(f"mdd={mdd:.3f}<{GATE_MDD}")
        label = "KILL"
    if cagr is not None and cagr <= 0:
        reasons.append(f"cagr={cagr:.3f}<=0")
        label = "KILL"
    if cagr is not None and cagr <= GATE_CAGR:
        reasons.append(f"cagr={cagr:.3f}<={GATE_CAGR}")
        label = "KILL"
    if excess is not None and excess <= 0:
        reasons.append(f"excess_spy={excess:.3f}<=0")
        label = "KILL"
    if excess is not None and excess <= 0 and sortino is not None and sortino < 0.4:
        reasons.append(f"excess<=0 and sortino={sortino:.2f}<0.4")
        label = "KILL"

    e_cagr = fget(early, "cagr")
    e_ex = fget(early, "excess_spy_total")
    m_cagr = fget(modern, "cagr")
    if m_cagr is not None and m_cagr > GATE_CAGR:
        if e_cagr is not None and e_cagr < 0:
            reasons.append("window_cherry: modern OK early CAGR<0")
            label = "KILL"
        if e_ex is not None and e_ex < -0.05:
            reasons.append("window_cherry: modern OK early excess_spy<-5pp")
            label = "KILL"
        if e_cagr is not None and e_cagr < 0.05 and (e_ex is not None and e_ex < 0):
            reasons.append("early_weak: CAGR<5% and excess_spy<0")
            # HOLD or KILL — plan says early fail + modern only → KILL if early CAGR<0
            # for CAGR 0-5% with excess<0 → treat as KILL under full gate already if full fails

    mc_s = mc.get("bootstrap_sortino_p5") if mc else None
    if mc_s is not None and mc_s < 0.1:
        reasons.append(f"mc_bootstrap_sortino_p5={mc_s:.3f}<0.1")
        label = "KILL"

    if not reasons and label == "HOLD":
        reasons.append("no_hard_kill_trigger")
    if label == "HOLD" and reasons == ["no_hard_kill_trigger"]:
        # still HOLD only if full passed soft bars
        if cagr is not None and excess is not None and cagr > GATE_CAGR and excess > 0:
            reasons = ["passes_full_gates_no_advance"]
        else:
            label = "KILL"
            reasons = ["failed_implicit_full_gates"]

    # ADVANCE forbidden
    return label, reasons


def run_one(
    strategy_name: str,
    *,
    data_root: Path,
    ticker_file: Path,
    universe_limit: int,
    first_year: int,
    last_year: int,
    risk_pct: Optional[float] = None,
) -> Dict[str, Any]:
    strat = get_strategy(strategy_name)
    if risk_pct is not None and isinstance(strat, OrbHtfDailyProxyStrategy):
        strat.risk_per_trade_pct = float(risk_pct)
    res = run_strategy_walk_forward(
        strat,
        data_root=data_root,
        ticker_file=ticker_file,
        universe_limit=universe_limit,
        first_oos_year=first_year,
        last_oos_year=last_year,
        preferred_index=["SPY", "QQQ"],
    )
    eq = res.get("equity")
    if eq is None:
        eq = res.get("full_equity")
    if not isinstance(eq, pd.Series):
        # strategy_runner returns full equity how?
        raise RuntimeError(f"no equity in result keys={list(res.keys())}")
    trades = res.get("trades")
    if not isinstance(trades, pd.DataFrame):
        trades = res.get("trades_df", pd.DataFrame())
    return {"result": res, "equity": eq, "trades": trades, "strategy": strategy_name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--universe", type=str, default="universe_longhist100.txt")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--first-year", type=int, default=2010)
    ap.add_argument("--last-year", type=int, default=2025)
    ap.add_argument("--out", type=str, default="reports/redesign/orb_htf_falsification_v1")
    ap.add_argument("--mc-sims", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true", help="limit years 2020-2022 only")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    first, last = args.first_year, args.last_year
    if args.smoke:
        first, last = 2020, 2022

    # Write one-ticker control files
    for t in ("SPY", "QQQ"):
        p = out / f"universe_{t}_only.txt"
        p.write_text(t + "\n", encoding="utf-8")

    runs: List[Dict[str, Any]] = []
    configs = [
        ("orb_htf_daily_proxy", args.universe, args.limit, "A0_base_longhist50"),
        ("orb_htf_daily_proxy_a1", args.universe, args.limit, "A1_sma200_longhist50"),
        ("orb_htf_daily_proxy", str(out / "universe_SPY_only.txt"), 1, "A0_SPY_only"),
        ("orb_htf_daily_proxy", str(out / "universe_QQQ_only.txt"), 1, "A0_QQQ_only"),
    ]
    # sizing MC stress only on primary name
    risk_stress = [0.005, 0.0075, 0.01]

    primary_pack: Optional[Dict[str, Any]] = None

    for sname, univ, lim, tag in configs:
        print(f"=== RUN {tag} {sname} univ={univ} limit={lim} ===")
        try:
            pack = run_one(
                sname,
                data_root=data_root,
                ticker_file=Path(univ),
                universe_limit=lim,
                first_year=first,
                last_year=last,
            )
        except Exception as e:
            runs.append({"tag": tag, "error": str(e)})
            print("ERROR", tag, e)
            continue
        eq = pack["equity"]
        trades = pack["trades"]
        full = _metrics(eq, trades)
        full["excess_spy_total"] = _spy_excess(eq, data_root)
        early = _window_pack(eq, trades, data_root, f"{first}-01-01", "2017-12-31")
        modern = _window_pack(eq, trades, data_root, "2018-01-01", f"{last}-12-31")
        stress = _window_pack(eq, trades, data_root, "2022-01-01", "2022-12-31")
        mc = _mc_pack(trades, n_sims=args.mc_sims if not args.smoke else 200)
        row = {
            "tag": tag,
            "strategy": sname,
            "data_label": "eod_proxy",
            "execution_mode": "daily_close_research",
            "universe": univ,
            "limit": lim,
            "full": full,
            "early": early,
            "modern": modern,
            "stress_2022": stress,
            "mc": mc,
        }
        if tag == "A0_base_longhist50":
            decision, reasons = _decide(full, early, modern, mc)
            row["decision"] = decision
            row["decision_reasons"] = reasons
            primary_pack = row
            # export equity/trades
            eq.to_csv(out / "equity_A0_base.csv", header=["equity"])
            if isinstance(trades, pd.DataFrame) and not trades.empty:
                trades.to_csv(out / "trades_A0_base.csv", index=False)
        runs.append(row)
        print(
            f"  CAGR={full.get('cagr')} MDD={full.get('max_drawdown')} "
            f"n={full.get('n_trades')} excess={full.get('excess_spy_total')}"
        )

    # Risk stress on A0 only (re-run light if not smoke-heavy)
    risk_rows = []
    if not args.smoke:
        for rp in risk_stress:
            tag = f"A0_risk_{rp}"
            print(f"=== RISK STRESS {tag} ===")
            try:
                pack = run_one(
                    "orb_htf_daily_proxy",
                    data_root=data_root,
                    ticker_file=Path(args.universe),
                    universe_limit=args.limit,
                    first_year=first,
                    last_year=last,
                    risk_pct=rp,
                )
                full = _metrics(pack["equity"], pack["trades"])
                full["excess_spy_total"] = _spy_excess(pack["equity"], data_root)
                risk_rows.append({"tag": tag, "risk_per_trade_pct": rp, "full": full})
                print(f"  risk={rp} CAGR={full.get('cagr')} MDD={full.get('max_drawdown')}")
            except Exception as e:
                risk_rows.append({"tag": tag, "error": str(e)})

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": "docs/design/2026-07-27_orb_htf_falsification.md",
        "data_label": "eod_proxy",
        "paper_freeze": "turbo_highvol_minalloc_unchanged",
        "advance_forbidden": True,
        "runs": runs,
        "risk_stress": risk_rows,
        "primary_decision": (primary_pack or {}).get("decision"),
        "primary_reasons": (primary_pack or {}).get("decision_reasons"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # scorecard csv
    lines = ["tag,strategy,cagr,sharpe,sortino,max_dd,n_trades,excess_spy,decision"]
    for r in runs:
        if r.get("error"):
            lines.append(f"{r.get('tag')},ERROR,,,,,,,")
            continue
        f = r.get("full") or {}
        lines.append(
            f"{r.get('tag')},{r.get('strategy')},{f.get('cagr')},{f.get('sharpe')},"
            f"{f.get('sortino')},{f.get('max_drawdown')},{f.get('n_trades')},"
            f"{f.get('excess_spy_total')},{r.get('decision','')}"
        )
    (out / "scorecard.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    dec = (primary_pack or {}).get("decision", "KILL")
    reasons = (primary_pack or {}).get("decision_reasons", ["no_primary_run"])
    decision_md = [
        "# DECISION — Sistema A ORB+HTF daily proxy",
        "",
        f"**Verdict:** `{dec}`",
        "",
        f"**data_label:** `eod_proxy` (not 15m session ORB)",
        f"**Paper freeze:** unchanged (`turbo_highvol_minalloc`)",
        f"**ADVANCE:** forbidden this cycle",
        "",
        "## Reasons",
        "",
    ]
    for x in reasons:
        decision_md.append(f"- {x}")
    decision_md.extend(
        [
            "",
            "## Primary full metrics",
            "",
            "```json",
            json.dumps((primary_pack or {}).get("full"), indent=2, default=str),
            "```",
            "",
            "Research only. Not financial advice.",
            "",
        ]
    )
    (out / "DECISION.md").write_text("\n".join(decision_md), encoding="utf-8")

    # SUMMARY.md
    sm = [
        "# SUMMARY — ORB+HTF daily proxy falsification v1",
        "",
        f"Generated: {summary['created_at']}",
        "",
        f"**Primary decision:** `{dec}`",
        "",
        "| Tag | CAGR | Sharpe | Sortino | MaxDD | Trades | Excess SPY |",
        "|-----|------|--------|---------|-------|--------|------------|",
    ]
    for r in runs:
        if r.get("error"):
            sm.append(f"| {r.get('tag')} | ERROR | | | | | |")
            continue
        f = r.get("full") or {}
        sm.append(
            f"| {r.get('tag')} | {f.get('cagr')} | {f.get('sharpe')} | {f.get('sortino')} | "
            f"{f.get('max_drawdown')} | {f.get('n_trades')} | {f.get('excess_spy_total')} |"
        )
    sm.extend(["", "See DECISION.md and summary.json.", ""])
    (out / "SUMMARY.md").write_text("\n".join(sm), encoding="utf-8")

    print("DONE", out, "decision=", dec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
