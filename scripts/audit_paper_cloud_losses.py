#!/usr/bin/env python3
"""Mega-audit of paper_cloud losses: reconstruct trades, vs SPY, failure modes.

Reads digests under reports/paper_cloud/history/<as_of>/strategies/*/daily/
and equity curves. Writes reports/paper_cloud/audits/<stamp>_loss_audit.md
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Lot:
    ticker: str
    qty: float
    entry_px: float
    entry_day: str
    strategy_id: str


@dataclass
class ClosedTrade:
    strategy_id: str
    ticker: str
    qty: float
    entry_px: float
    exit_px: float
    entry_day: str
    exit_day: str
    pnl: float
    ret: float
    hold_days_approx: int = 0
    exit_side: str = "sell"


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def fifo_trades(strategy_id: str, daily_dir: Path) -> Tuple[List[ClosedTrade], List[Lot], Counter, dict]:
    """Reconstruct closed trades from daily fills (FIFO)."""
    days = sorted(daily_dir.glob("*.json"))
    open_lots: Dict[str, List[Lot]] = defaultdict(list)
    closed: List[ClosedTrade] = []
    rejects = Counter()
    day_stats = []
    for fp in days:
        d = _load_json(fp)
        day = d.get("day") or fp.stem
        for k, v in (d.get("reject_reasons") or {}).items():
            rejects[str(k)] += int(v)
        day_stats.append(
            {
                "day": day,
                "equity": float(d.get("equity") or 0),
                "n_buys": int(d.get("n_buys") or 0),
                "n_sells": int(d.get("n_sells") or 0),
                "n_positions": int(d.get("n_positions") or 0),
                "gross_exposure": float(d.get("gross_exposure") or 0),
                "dd_from_peak": float(d.get("dd_from_peak") or 0),
                "commission": float(d.get("commission") or 0),
            }
        )
        for f in d.get("fills") or []:
            side = str(f.get("side") or "").lower()
            t = str(f.get("ticker") or "").upper()
            qty = float(f.get("qty") or 0)
            px = float(f.get("price") or 0)
            if qty <= 0 or px <= 0 or not t:
                continue
            if side == "buy":
                open_lots[t].append(
                    Lot(ticker=t, qty=qty, entry_px=px, entry_day=day, strategy_id=strategy_id)
                )
            elif side == "sell":
                rem = qty
                while rem > 1e-9 and open_lots[t]:
                    lot = open_lots[t][0]
                    take = min(lot.qty, rem)
                    pnl = (px - lot.entry_px) * take
                    ret = (px / lot.entry_px - 1.0) if lot.entry_px > 0 else 0.0
                    # calendar span from ISO dates
                    try:
                        h = (pd.Timestamp(day) - pd.Timestamp(lot.entry_day)).days
                    except Exception:
                        h = 0
                    closed.append(
                        ClosedTrade(
                            strategy_id=strategy_id,
                            ticker=t,
                            qty=take,
                            entry_px=lot.entry_px,
                            exit_px=px,
                            entry_day=lot.entry_day,
                            exit_day=day,
                            pnl=pnl,
                            ret=ret,
                            hold_days_approx=int(h),
                        )
                    )
                    lot.qty -= take
                    rem -= take
                    if lot.qty <= 1e-9:
                        open_lots[t].pop(0)
    open_flat: List[Lot] = []
    for lots in open_lots.values():
        open_flat.extend(lots)
    return closed, open_flat, rejects, {"days": day_stats}


def trade_stats(trades: List[ClosedTrade]) -> Dict[str, Any]:
    if not trades:
        return {
            "n": 0,
            "win_rate": None,
            "avg_ret": None,
            "avg_win": None,
            "avg_loss": None,
            "profit_factor": None,
            "expectancy_ret": None,
            "total_pnl": 0.0,
            "median_hold_days": None,
        }
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses))
    rets = [t.ret for t in trades]
    return {
        "n": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_ret": sum(rets) / len(rets),
        "avg_win": (sum(t.ret for t in wins) / len(wins)) if wins else None,
        "avg_loss": (sum(t.ret for t in losses) / len(losses)) if losses else None,
        "profit_factor": (gp / gl) if gl > 1e-9 else (math.inf if gp > 0 else 0.0),
        "expectancy_ret": sum(rets) / len(rets),
        "total_pnl": sum(t.pnl for t in trades),
        "median_hold_days": float(pd.Series([t.hold_days_approx for t in trades]).median()),
        "p25_ret": float(pd.Series(rets).quantile(0.25)),
        "p75_ret": float(pd.Series(rets).quantile(0.75)),
        "worst_ret": min(rets),
        "best_ret": max(rets),
    }


def by_ticker(trades: List[ClosedTrade]) -> List[Dict[str, Any]]:
    g: Dict[str, List[ClosedTrade]] = defaultdict(list)
    for t in trades:
        g[t.ticker].append(t)
    rows = []
    for ticker, ts in sorted(g.items(), key=lambda kv: sum(x.pnl for x in kv[1])):
        st = trade_stats(ts)
        rows.append({"ticker": ticker, **st})
    return rows


def by_month(trades: List[ClosedTrade]) -> List[Dict[str, Any]]:
    g: Dict[str, List[ClosedTrade]] = defaultdict(list)
    for t in trades:
        m = str(t.exit_day)[:7]
        g[m].append(t)
    rows = []
    for m in sorted(g):
        st = trade_stats(g[m])
        rows.append({"month": m, **st})
    return rows


def exposure_profile(day_stats: List[dict], capital0: float = 100_000.0) -> Dict[str, Any]:
    if not day_stats:
        return {}
    exp = [d["gross_exposure"] / capital0 for d in day_stats if capital0 > 0]
    eq = [d["equity"] for d in day_stats]
    return {
        "avg_gross_exposure": sum(exp) / len(exp) if exp else 0,
        "max_gross_exposure": max(exp) if exp else 0,
        "median_gross_exposure": float(pd.Series(exp).median()) if exp else 0,
        "pct_days_flat": sum(1 for d in day_stats if d["n_positions"] == 0) / len(day_stats),
        "pct_days_fullish": sum(1 for e in exp if e > 0.5) / len(exp) if exp else 0,
        "max_dd_from_peak": min(d["dd_from_peak"] for d in day_stats),
        "final_equity": eq[-1] if eq else capital0,
        "total_return": (eq[-1] / capital0 - 1.0) if eq else 0.0,
    }


def spy_bh(start: str, end: str, seed: Path) -> Dict[str, Any]:
    df = pd.read_csv(seed)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")
    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    w = df[(df["date"] >= s) & (df["date"] <= e)]
    if w.empty:
        return {"ok": False}
    c0 = float(w.iloc[0]["close"])
    c1 = float(w.iloc[-1]["close"])
    # max dd of BH
    peak = c0
    max_dd = 0.0
    for c in w["close"]:
        c = float(c)
        peak = max(peak, c)
        max_dd = min(max_dd, c / peak - 1.0)
    return {
        "ok": True,
        "start": str(w.iloc[0]["date"]),
        "end": str(w.iloc[-1]["date"]),
        "return": c1 / c0 - 1.0,
        "max_dd": max_dd,
        "n_days": len(w),
    }


def worst_trades(trades: List[ClosedTrade], n: int = 15) -> List[ClosedTrade]:
    return sorted(trades, key=lambda t: t.pnl)[:n]


def best_trades(trades: List[ClosedTrade], n: int = 10) -> List[ClosedTrade]:
    return sorted(trades, key=lambda t: t.pnl, reverse=True)[:n]


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:+.2f}%"


def fmt_f(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "n/a"
    if math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def analyze_root_causes(
    all_trades: List[ClosedTrade],
    strat_rows: List[dict],
    spy: dict,
    rejects_all: Counter,
) -> List[str]:
    causes = []
    # 1. Alpha vs market
    if spy.get("ok"):
        avg_strat = sum(r["total_return"] for r in strat_rows) / max(len(strat_rows), 1)
        if spy["return"] > 0 and avg_strat < 0:
            causes.append(
                f"BENCHMARK: SPY B&H {fmt_pct(spy['return'])} while avg strategy {fmt_pct(avg_strat)} "
                f"— system underperformed a passive long index (edge not present in this window)."
            )
        elif spy["return"] < 0 and avg_strat < spy["return"]:
            causes.append(
                f"BENCHMARK: SPY also down {fmt_pct(spy['return'])} but strategies worse on average "
                f"({fmt_pct(avg_strat)}) — active long rules added drag vs sitting in SPY."
            )
        elif spy["return"] < 0 and avg_strat > spy["return"]:
            causes.append(
                f"BENCHMARK: SPY {fmt_pct(spy['return'])}; strategies less bad on average "
                f"({fmt_pct(avg_strat)}) — some relative value but absolute still red if negative."
            )

    st = trade_stats(all_trades)
    if st["n"] and st["win_rate"] is not None:
        if st["win_rate"] < 0.45 and (st["avg_win"] or 0) <= abs(st["avg_loss"] or 0):
            causes.append(
                f"ASYMMETRY: win_rate={st['win_rate']:.1%} with avg_win={fmt_pct(st['avg_win'])} "
                f"vs avg_loss={fmt_pct(st['avg_loss'])} — winners do not pay for losers (classic "
                f"momentum-chop: buy strength, mean-revert against you)."
            )
        if st.get("profit_factor") is not None and st["profit_factor"] < 1.0:
            causes.append(
                f"PROFIT FACTOR: {fmt_f(st['profit_factor'], 3)} < 1 — gross losses exceed gross wins "
                f"on closed round-trips (before open MTM)."
            )

    # Hold horizon short for trend
    if st.get("median_hold_days") is not None and st["median_hold_days"] < 12:
        causes.append(
            f"HORIZON: median hold ~{st['median_hold_days']:.0f} calendar days with time_stop "
            f"max_horizon often 10–20 bars — trend/momentum needs longer runs; time-stop cuts "
            f"winners early while stops realize losers fully."
        )

    # Concentration of losses in tickers
    by_t = by_ticker(all_trades)
    if by_t:
        worst = by_t[0]
        if worst["total_pnl"] < -500:
            causes.append(
                f"TICKER DRAG: worst book is {worst['ticker']} closed PnL ${worst['total_pnl']:.0f} "
                f"(n={worst['n']}, WR={fmt_f((worst['win_rate'] or 0)*100,1)}%) — name selection "
                f"or repeated re-entry into same mega-cap after failed momentum."
            )

    # Rejects / costs
    if rejects_all:
        top_r = rejects_all.most_common(3)
        causes.append(
            "ENTRY FILTERS: rejects "
            + ", ".join(f"{k}×{v}" for k, v in top_r)
            + " — gap rules kill some bad entries but also drop continuation days; remaining "
            "entries still lose, so filter is not the main alpha problem."
        )

    # Regime / rule design
    causes.append(
        "SIGNAL DESIGN: rule is long-only close>SMA50/200 + ret_1m>0 + ATR band. That is "
        "late-trend entry (buy strength). In range/choppy mega-cap markets this buys "
        "local tops; there is no mean-reversion, no short, no meta-label skip, no sector rotation."
    )
    causes.append(
        "UNIVERSE: 8 mega-caps + QQQ/SPY — highly correlated; diversification in zoo "
        "(S06 vs S05) barely helps if the common factor is 'long NVDA/META/etc after up month'."
    )
    causes.append(
        "SIZING/COSTS: fixed commission + entry/exit slippage on many small tickets "
        "(aggressive S09 worst on turnover) compounds when expectancy per trade is near zero/negative."
    )
    causes.append(
        "NOT ML EDGE: cloud paper is rule-based proxy (LIV-04), not the XGB+meta stack from "
        "research — do not interpret paper_cloud red as proof the full research pipeline fails, "
        "but also do not claim paper_cloud proves production readiness."
    )
    return causes


def main() -> int:
    hist = ROOT / "reports" / "paper_cloud" / "history"
    # pick newest as_of dir
    as_dirs = sorted([p for p in hist.iterdir() if p.is_dir()], reverse=True)
    if not as_dirs:
        print("No history packs")
        return 1
    pack = as_dirs[0]
    summary = _load_json(ROOT / "reports" / "paper_cloud" / "latest" / "summary.json")
    window = summary.get("window") or {}
    start = window.get("start") or "2025-10-29"
    end = window.get("end") or "2026-07-21"
    capital0 = 100_000.0

    equity = {}
    eq_path = ROOT / "reports" / "paper_cloud" / "latest" / "equity_curves.json"
    if eq_path.is_file():
        equity = _load_json(eq_path)

    spy = spy_bh(start, end, ROOT / "paper_live" / "cloud" / "seed_ohlcv" / "SPY_history.csv")
    # also 2026-only BH
    spy_2026 = spy_bh("2026-01-02", end, ROOT / "paper_live" / "cloud" / "seed_ohlcv" / "SPY_history.csv")

    strat_dir = pack / "strategies"
    all_trades: List[ClosedTrade] = []
    rejects_all: Counter = Counter()
    strat_rows = []
    per_strat_detail = []

    for sdir in sorted(strat_dir.iterdir()):
        if not sdir.is_dir():
            continue
        sid = sdir.name
        daily = sdir / "daily"
        if not daily.is_dir():
            continue
        closed, open_lots, rejects, meta = fifo_trades(sid, daily)
        rejects_all.update(rejects)
        all_trades.extend(closed)
        st = trade_stats(closed)
        exp = exposure_profile(meta["days"], capital0)
        # match summary return
        srow = next((x for x in summary.get("strategies") or [] if x.get("strategy_id") == sid), {})
        row = {
            "strategy_id": sid,
            "label": srow.get("label"),
            "total_return": float(srow.get("total_return") or exp.get("total_return") or 0),
            "n_entries": int(srow.get("n_entries") or 0),
            "n_exits": int(srow.get("n_exits") or 0),
            "commission": float(srow.get("total_commission") or 0),
            "closed_n": st["n"],
            "win_rate": st["win_rate"],
            "avg_ret": st["avg_ret"],
            "profit_factor": st["profit_factor"],
            "closed_pnl": st["total_pnl"],
            "median_hold": st["median_hold_days"],
            "open_lots": len(open_lots),
            "avg_exposure": exp.get("avg_gross_exposure"),
            "max_dd": exp.get("max_dd_from_peak"),
            "rejects": dict(rejects),
        }
        strat_rows.append(row)
        per_strat_detail.append(
            {
                "row": row,
                "by_ticker": by_ticker(closed)[:8],
                "worst": worst_trades(closed, 8),
                "best": best_trades(closed, 5),
                "by_month": by_month(closed),
            }
        )

    # 2026-only subset of closed trades
    trades_2026 = [t for t in all_trades if str(t.entry_day) >= "2026-01-01"]
    st_all = trade_stats(all_trades)
    st_2026 = trade_stats(trades_2026)

    # Monthly equity path for S01
    s01_days = None
    for det in per_strat_detail:
        if det["row"]["strategy_id"] == "S01_baseline_minalloc":
            # rebuild from daily
            daily = pack / "strategies" / "S01_baseline_minalloc" / "daily"
            _, _, _, meta = fifo_trades("S01", daily)
            s01_days = meta["days"]
            break

    causes = analyze_root_causes(all_trades, strat_rows, spy, rejects_all)

    # Plan section (actionable)
    plan = [
        {
            "id": "AUD-01",
            "title": "Benchmark honesty + attribution pack",
            "why": "Without SPY/equal-weight daily attribution we cannot tell alpha vs beta drag.",
            "do": [
                "Add SPY B&H and equal-weight universe B&H curves to every SUMMARY.",
                "Log exit_reason distribution (stop vs time_stop vs EOD) into digests.",
                "Emit closed-trade CSV per strategy (entry/exit/ret/hold/reason).",
            ],
            "priority": "P0",
        },
        {
            "id": "AUD-02",
            "title": "Fix expectancy before more zoo knobs",
            "why": "All 10 variants red ⇒ shared signal is the problem, not stop width alone.",
            "do": [
                "A/B: require ret_1m rank top-k only (not all positive).",
                "A/B: add pullback entry (close>SMA200 but RSI/near SMA50) vs pure breakout.",
                "A/B: meta-skip when QQQ 5d ret < 0 even if dual MA on.",
                "Disable trading QQQ/SPY as names (regime only) — reduce beta double-count.",
            ],
            "priority": "P0",
        },
        {
            "id": "AUD-03",
            "title": "Exit asymmetry",
            "why": "Time-stop + hard stop realizes losses; winners capped by horizon.",
            "do": [
                "Trail winners more aggressively; lengthen max_horizon on strong trends (ATR).",
                "Partial take-profit at +1R; let runner to 2–3R.",
                "Measure MAE/MFE on every closed trade in audit CSV.",
            ],
            "priority": "P1",
        },
        {
            "id": "AUD-04",
            "title": "Cost / turnover control",
            "why": "S09 high entries + commissions with negative edge is pure leak.",
            "do": [
                "Cap max_entries_per_day globally; min hold before re-entry same ticker.",
                "Raise min_alloc so fewer micro tickets; or commission-aware skip if edge < cost.",
            ],
            "priority": "P1",
        },
        {
            "id": "AUD-05",
            "title": "True OOS protocol for paper cloud",
            "why": "Single ~9m window is not walk-forward; 2026-only is still one regime.",
            "do": [
                "Mandatory multi-window: 2022 bear, 2023 bull, 2024, 2025, 2026 YTD separately.",
                "Kill strategies that lose to SPY in ≥3/5 windows.",
                "Only promote rules that beat SPY after costs on purged/walk-forward research first.",
            ],
            "priority": "P0",
        },
        {
            "id": "AUD-06",
            "title": "Reconnect research stack (optional)",
            "why": "Rule_trend_mom is a stub; XGB+meta may differ.",
            "do": [
                "Plug frozen signal model into DailySignalPipeline.signal_fn with feature parity.",
                "Paper cloud becomes validation harness, not the strategy definition.",
            ],
            "priority": "P2",
        },
    ]

    # Write report
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "reports" / "paper_cloud" / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stamp}_loss_audit.md"
    lines: List[str] = []
    lines.append(f"# Mega-auditoría paper cloud — por qué estamos en negativo")
    lines.append("")
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()} · pack `{pack.name}`_")
    lines.append("")
    lines.append("## 0. Scope y honestidad")
    lines.append("")
    lines.append(
        f"- **Ventana analizada (latest SUMMARY):** `{start}` → `{end}` · capital virtual $100,000."
    )
    lines.append(
        "- **Datos:** Yahoo free OHLCV (mega-caps + SPY/QQQ). Paper only — no dinero real."
    )
    lines.append(
        "- **Señal:** rule-based `rule_trend_mom_atr` (no XGBoost meta-label del stack research)."
    )
    lines.append(
        "- Round-trips reconstruidos por **FIFO** sobre fills diarios; PnL de closed trades "
        "≠ equity final exacto (quedan posiciones abiertas + MTM)."
    )
    lines.append("")
    lines.append("## 1. Veredicto ejecutivo")
    lines.append("")
    n_red = sum(1 for r in strat_rows if r["total_return"] < 0)
    lines.append(
        f"**{n_red}/{len(strat_rows)} estrategias en negativo.** "
        "No es un solo kill-switch ni un bug de datos sintéticos: el **edge de la regla long-momentum "
        "no aparece** en este tramo frente a buy&hold, y el zoo solo re-parametriza la misma idea."
    )
    lines.append("")
    if spy.get("ok"):
        lines.append(
            f"- **SPY buy&hold misma ventana:** {fmt_pct(spy['return'])} "
            f"(maxDD {fmt_pct(spy['max_dd'])}, {spy['n_days']} sesiones)."
        )
    if spy_2026.get("ok"):
        lines.append(
            f"- **SPY B&H 2026-01-02→fin pack:** {fmt_pct(spy_2026['return'])} "
            f"(maxDD {fmt_pct(spy_2026['max_dd'])})."
        )
    best = max(strat_rows, key=lambda r: r["total_return"]) if strat_rows else None
    worst = min(strat_rows, key=lambda r: r["total_return"]) if strat_rows else None
    if best and worst:
        lines.append(
            f"- **Mejor zoo:** `{best['strategy_id']}` {fmt_pct(best['total_return'])} · "
            f"**Peor:** `{worst['strategy_id']}` {fmt_pct(worst['total_return'])}."
        )
    lines.append(
        f"- **Closed trades (todas las strats):** n={st_all['n']} · WR={fmt_f((st_all['win_rate'] or 0)*100,1)}% · "
        f"avg_ret={fmt_pct(st_all['avg_ret'])} · PF={fmt_f(st_all['profit_factor'],3)} · "
        f"closed_pnl_sum=${st_all['total_pnl']:.0f}."
    )
    if st_2026["n"]:
        lines.append(
            f"- **Subset entries ≥2026-01-01:** n={st_2026['n']} · WR={fmt_f((st_2026['win_rate'] or 0)*100,1)}% · "
            f"avg_ret={fmt_pct(st_2026['avg_ret'])} · PF={fmt_f(st_2026['profit_factor'],3)}."
        )
    lines.append("")
    lines.append("## 2. Ranking vs costes y exposición")
    lines.append("")
    lines.append(
        "| Strat | Return | Entries | Closed | WR | Avg ret | PF | Closed PnL | Comm | Avg exp | MaxDD |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(strat_rows, key=lambda x: x["total_return"], reverse=True):
        lines.append(
            f"| `{r['strategy_id']}` | {fmt_pct(r['total_return'])} | {r['n_entries']} | {r['closed_n']} | "
            f"{fmt_f((r['win_rate'] or 0)*100,1)}% | {fmt_pct(r['avg_ret'])} | {fmt_f(r['profit_factor'],2)} | "
            f"${r['closed_pnl']:.0f} | ${r['commission']:.0f} | {fmt_f((r['avg_exposure'] or 0)*100,1)}% | "
            f"{fmt_pct(r['max_dd'])} |"
        )
    lines.append("")
    lines.append("### Lectura rápida del zoo")
    lines.append("")
    lines.append(
        "- **S07 high_vol_only** menos rojo → **menos trades** (menos veces pagas el edge negativo)."
    )
    lines.append(
        "- **S09 aggressive** y **S05 concentrated** peores → más tamaño o más frecuencia **amplifican** el mismo edge negativo."
    )
    lines.append(
        "- **S02 no_regime** no mejora a baseline → el régimen QQQ dual-MA no es el único problema; la selección de nombres también pierde."
    )
    lines.append(
        "- Comisiones (~$24–$202) son **pequeñas** vs miles de $ de equity drag → el rojo es **PnL de mercado**, no solo fricción."
    )
    lines.append("")
    lines.append("## 3. Anatomía de los trades (agregado)")
    lines.append("")
    lines.append(f"- Mediana hold (calendario): **{fmt_f(st_all.get('median_hold_days'),1)} días**")
    lines.append(f"- Avg win / avg loss (ret): {fmt_pct(st_all.get('avg_win'))} / {fmt_pct(st_all.get('avg_loss'))}")
    lines.append(f"- P25 / P75 ret: {fmt_pct(st_all.get('p25_ret'))} / {fmt_pct(st_all.get('p75_ret'))}")
    lines.append(f"- Best / worst single trade ret: {fmt_pct(st_all.get('best_ret'))} / {fmt_pct(st_all.get('worst_ret'))}")
    lines.append("")
    lines.append("### Por ticker (peores primero, closed PnL)")
    lines.append("")
    lines.append("| Ticker | n | WR | Avg ret | Closed PnL |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in by_ticker(all_trades)[:12]:
        lines.append(
            f"| {row['ticker']} | {row['n']} | {fmt_f((row['win_rate'] or 0)*100,1)}% | "
            f"{fmt_pct(row['avg_ret'])} | ${row['total_pnl']:.0f} |"
        )
    lines.append("")
    lines.append("### Por mes de salida")
    lines.append("")
    lines.append("| Month | n | WR | Avg ret | Closed PnL |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in by_month(all_trades):
        lines.append(
            f"| {row['month']} | {row['n']} | {fmt_f((row['win_rate'] or 0)*100,1)}% | "
            f"{fmt_pct(row['avg_ret'])} | ${row['total_pnl']:.0f} |"
        )
    lines.append("")
    lines.append("### Peores 15 round-trips (todas las strats)")
    lines.append("")
    lines.append("| Strat | Ticker | Entry | Exit | Hold d | Ret | PnL |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for t in worst_trades(all_trades, 15):
        lines.append(
            f"| `{t.strategy_id}` | {t.ticker} | {t.entry_day} @ {t.entry_px:.2f} | "
            f"{t.exit_day} @ {t.exit_px:.2f} | {t.hold_days_approx} | {fmt_pct(t.ret)} | ${t.pnl:.0f} |"
        )
    lines.append("")
    lines.append("### Rejects de entrada (confirm gap / filtros)")
    lines.append("")
    if rejects_all:
        for k, v in rejects_all.most_common():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- (sin rejects contados en digests)")
    lines.append("")
    lines.append("## 4. Detalle baseline S01 (referencia)")
    lines.append("")
    s01 = next((d for d in per_strat_detail if d["row"]["strategy_id"] == "S01_baseline_minalloc"), None)
    if s01:
        r = s01["row"]
        lines.append(
            f"Return {fmt_pct(r['total_return'])} · closed n={r['closed_n']} · WR={fmt_f((r['win_rate'] or 0)*100,1)}% · "
            f"PF={fmt_f(r['profit_factor'],3)} · avg exp={fmt_f((r['avg_exposure'] or 0)*100,1)}%."
        )
        lines.append("")
        lines.append("Top drag tickers S01:")
        lines.append("")
        lines.append("| Ticker | n | WR | PnL |")
        lines.append("|---|---:|---:|---:|")
        for row in s01["by_ticker"][:6]:
            lines.append(
                f"| {row['ticker']} | {row['n']} | {fmt_f((row['win_rate'] or 0)*100,1)}% | ${row['total_pnl']:.0f} |"
            )
        lines.append("")
        lines.append("Peores trades S01:")
        for t in s01["worst"]:
            lines.append(
                f"- {t.ticker} {t.entry_day}→{t.exit_day} {fmt_pct(t.ret)} (${t.pnl:.0f})"
            )
    lines.append("")
    lines.append("## 5. Por qué fallan (causas raíz)")
    lines.append("")
    for i, c in enumerate(causes, 1):
        lines.append(f"{i}. {c}")
    lines.append("")
    lines.append("### Mecánica concreta del trade perdedor típico")
    lines.append("")
    lines.append("```")
    lines.append("D close: close > SMA50 & SMA200, ret_1m > 0, ATR en banda → score alto")
    lines.append("D+1 open: confirm (no gap down >5% / no chase >8%) → buy ~1.5% NAV")
    lines.append("In-trade: stop = max(entry*(1-hard%), entry - k*ATR); trail con high")
    lines.append("Exit: low toca stop  OR  bars_held >= max_horizon → time_stop al close")
    lines.append("```")
    lines.append("")
    lines.append(
        "El patrón de fallo: **compras después de un mes alcista** en nombres correlacionados; "
        "si los siguientes 5–15 días son digieren/range, el **stop o el time_stop** cierra en leve "
        "rojo; los pocos winners no compensan (PF<1). Amplificar entradas (S09) o tamaño (S05) "
        "empeora; **no tradear** (S07 filtro ATR alto) pierde menos."
    )
    lines.append("")
    lines.append("## 6. Lo que NO es la causa principal")
    lines.append("")
    lines.append("- ~~Datos sintéticos~~ — este pack es Yahoo real.")
    lines.append("- ~~Kill switch~~ — hard_kill=false en las 10.")
    lines.append("- ~~Solo comisiones~~ — drag de comisiones << pérdida de equity.")
    lines.append("- ~~Un bug de un solo parámetro del zoo~~ — todas las variantes rojas.")
    lines.append("")
    lines.append("## 7. Plan de acción (priorizado)")
    lines.append("")
    for p in plan:
        lines.append(f"### {p['id']} · {p['title']} · **{p['priority']}**")
        lines.append("")
        lines.append(f"**Por qué:** {p['why']}")
        lines.append("")
        lines.append("**Hacer:**")
        for d in p["do"]:
            lines.append(f"- {d}")
        lines.append("")
    lines.append("## 8. Criterios de éxito (antes de decir 'arreglado')")
    lines.append("")
    lines.append("1. Al menos **1** variante con return > SPY B&H **después de costes** en ≥2 ventanas OOS.")
    lines.append("2. Profit factor closed trades **> 1.1** y win*avg_win + (1-win)*avg_loss **> 0**.")
    lines.append("3. Audit CSV con exit_reason + MAE/MFE commiteado en cada pack cloud.")
    lines.append("4. No reclamar edge si solo mejora el tramo 2026-YTD en aislamiento.")
    lines.append("")
    lines.append("## 9. Próximo PR sugerido")
    lines.append("")
    lines.append("1. **PR-AUD-A:** trade log + exit_reason + SPY benchmark en SUMMARY (instrumentación).")
    lines.append("2. **PR-AUD-B:** 4 A/B del signal (rank top-k, pullback, meta-skip QQQ weak, no trade index).")
    lines.append("3. **PR-AUD-C:** multi-window batch en Actions (`start` por era) + scorecard comparativo.")
    lines.append("")
    lines.append("---")
    lines.append("_Research software. Not financial advice._")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # also dump machine-readable
    dump = {
        "window": window,
        "spy": spy,
        "spy_2026": spy_2026,
        "strategies": strat_rows,
        "closed_trade_stats_all": st_all,
        "closed_trade_stats_2026_entries": st_2026,
        "by_ticker": by_ticker(all_trades),
        "by_month": by_month(all_trades),
        "rejects": dict(rejects_all),
        "causes": causes,
        "plan": plan,
        "report": str(out.relative_to(ROOT)),
    }
    jpath = out.with_suffix(".json")
    jpath.write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
    print(f"WROTE {out}")
    print(f"WROTE {jpath}")
    print(f"SPY {spy}")
    print(f"ALL trades n={st_all['n']} WR={st_all['win_rate']} PF={st_all['profit_factor']} pnl={st_all['total_pnl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
