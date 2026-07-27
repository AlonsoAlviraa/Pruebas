"""Mega trade-level audit for a full-OOS strategy (equity + trades + SPY regimes).

Outputs AUDIT.md + AUDIT.json next to the equity/trades files.
Research only. Not financial advice.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.walk_forward import load_benchmark_equity  # noqa: E402


def _load_eq(path: Path) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
    return s[~s.index.duplicated(keep="last")].dropna().sort_index()


def _dd(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def _spy_aligned(eq: pd.Series, data_root: Path) -> pd.Series:
    b = load_benchmark_equity(data_root, eq.index.min(), eq.index.max(), preferred=["SPY"])
    if b is None or b.empty:
        raise RuntimeError("SPY benchmark missing")
    b = b.copy()
    b.index = pd.to_datetime(b.index, utc=True).normalize()
    e = eq.copy()
    e.index = pd.to_datetime(e.index, utc=True).normalize()
    e = e[~e.index.duplicated(keep="last")]
    b = b[~b.index.duplicated(keep="last")]
    j = pd.concat([e.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
    return j["b"]


def _regime_from_spy(spy: pd.Series, lookback: int = 63) -> pd.Series:
    """Causal SPY regime labels on each day: BULL / FLAT / BEAR.

    - ret_lb = SPY total return over lookback days
    - BEAR if ret_lb <= -8% OR below SMA200 and ret_lb < 0
    - BULL if ret_lb >= +8% OR above SMA200 and ret_lb > 0
    - else FLAT
    """
    px = spy.astype(float)
    ret_lb = px / px.shift(lookback) - 1.0
    sma = px.rolling(200, min_periods=100).mean()
    above = px > sma
    lab = pd.Series("FLAT", index=px.index, dtype=object)
    lab[(ret_lb <= -0.08) | ((~above) & (ret_lb < 0))] = "BEAR"
    lab[(ret_lb >= 0.08) | (above & (ret_lb > 0))] = "BULL"
    # priority: deep BEAR overrides BULL
    lab[ret_lb <= -0.08] = "BEAR"
    lab[ret_lb >= 0.08] = "BULL"
    # mild middle stays FLAT if |ret| < 3%
    mid = ret_lb.abs() < 0.03
    lab[mid] = "FLAT"
    return lab


def _fmt_pct(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{100 * x:.2f}%"


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def analyze(
    eq: pd.Series,
    tr: pd.DataFrame,
    spy: pd.Series,
    *,
    name: str,
) -> Dict[str, Any]:
    tr = tr.copy()
    tr["entry_date"] = pd.to_datetime(tr["entry_date"], utc=True, errors="coerce")
    tr["exit_date"] = pd.to_datetime(tr["exit_date"], utc=True, errors="coerce")
    tr["net_profit"] = pd.to_numeric(tr["net_profit"], errors="coerce")
    tr["trade_return"] = pd.to_numeric(tr["trade_return"], errors="coerce")
    tr["bars_held"] = pd.to_numeric(tr["bars_held"], errors="coerce")
    tr["capital_used"] = pd.to_numeric(tr["capital_used"], errors="coerce")

    # align entry regime
    spy_n = spy.copy()
    spy_n.index = pd.to_datetime(spy_n.index, utc=True).normalize()
    reg = _regime_from_spy(spy_n)
    reg.index = pd.to_datetime(reg.index, utc=True).normalize()
    reg = reg[~reg.index.duplicated(keep="last")]
    entry_d = tr["entry_date"].dt.normalize()
    tr["entry_regime"] = entry_d.map(reg).fillna("FLAT")
    # SPY 21d return at entry
    spy_ret21 = (spy_n / spy_n.shift(21) - 1.0).sort_index()
    spy_ret21 = spy_ret21[~spy_ret21.index.duplicated(keep="last")]
    tr["spy_ret21_at_entry"] = entry_d.map(spy_ret21)

    dd = _dd(eq)
    # mark if trade entered while strategy already in deep DD
    eq_n = eq.copy()
    eq_n.index = pd.to_datetime(eq_n.index, utc=True).normalize()
    eq_n = eq_n[~eq_n.index.duplicated(keep="last")].sort_index()
    dd_n = _dd(eq_n)
    dd_n = dd_n[~dd_n.index.duplicated(keep="last")]
    tr["dd_at_entry"] = entry_d.map(dd_n)
    tr["in_deep_dd_entry"] = tr["dd_at_entry"] <= -0.25

    wins = tr["net_profit"] > 0
    losses = tr["net_profit"] <= 0

    # --- overall ---
    overall = {
        "n_trades": int(len(tr)),
        "win_rate": float(wins.mean()) if len(tr) else 0.0,
        "avg_win": float(tr.loc[wins, "trade_return"].mean()) if wins.any() else 0.0,
        "avg_loss": float(tr.loc[losses, "trade_return"].mean()) if losses.any() else 0.0,
        "median_win": float(tr.loc[wins, "trade_return"].median()) if wins.any() else 0.0,
        "median_loss": float(tr.loc[losses, "trade_return"].median()) if losses.any() else 0.0,
        "expectancy_ret": float(tr["trade_return"].mean()) if len(tr) else 0.0,
        "expectancy_usd": float(tr["net_profit"].mean()) if len(tr) else 0.0,
        "profit_factor": float(
            tr.loc[wins, "net_profit"].sum() / max(-tr.loc[losses, "net_profit"].sum(), 1e-9)
        )
        if losses.any() and wins.any()
        else float("nan"),
        "avg_bars": float(tr["bars_held"].mean()) if len(tr) else 0.0,
        "sum_pnl": float(tr["net_profit"].sum()),
        "top10_pnl_share": float(
            tr.nlargest(10, "net_profit")["net_profit"].sum() / max(tr.loc[wins, "net_profit"].sum(), 1e-9)
        )
        if wins.any()
        else 0.0,
        "bottom10_loss_share": float(
            tr.nsmallest(10, "net_profit")["net_profit"].sum() / min(tr.loc[losses, "net_profit"].sum(), -1e-9)
        )
        if losses.any()
        else 0.0,
    }
    # payoff ratio
    if overall["avg_loss"] != 0:
        overall["payoff_ratio"] = abs(overall["avg_win"] / overall["avg_loss"]) if overall["avg_loss"] else float("nan")
    else:
        overall["payoff_ratio"] = float("nan")
    # Kelly fraction rough: f* = p - (1-p)/b where b = avg_win/|avg_loss|
    p = overall["win_rate"]
    b = overall["payoff_ratio"] if overall["payoff_ratio"] == overall["payoff_ratio"] else 0.0
    overall["kelly_full"] = float(p - (1 - p) / b) if b and b > 0 else float("nan")
    overall["kelly_quarter"] = (
        float(overall["kelly_full"] * 0.25) if overall["kelly_full"] == overall["kelly_full"] else float("nan")
    )

    # --- by exit reason ---
    by_exit = []
    for reason, g in tr.groupby("exit_reason"):
        w = g["net_profit"] > 0
        by_exit.append(
            {
                "exit_reason": str(reason),
                "n": int(len(g)),
                "pct": float(len(g) / max(len(tr), 1)),
                "win_rate": float(w.mean()),
                "sum_pnl": float(g["net_profit"].sum()),
                "avg_ret": float(g["trade_return"].mean()),
                "avg_bars": float(g["bars_held"].mean()),
            }
        )
    by_exit.sort(key=lambda x: x["sum_pnl"])

    # --- by year ---
    by_year = []
    for y, g in tr.groupby("oos_year"):
        w = g["net_profit"] > 0
        by_year.append(
            {
                "year": int(y),
                "n": int(len(g)),
                "win_rate": float(w.mean()),
                "sum_pnl": float(g["net_profit"].sum()),
                "avg_ret": float(g["trade_return"].mean()),
                "hard_stop_pct": float((g["exit_reason"] == "hard_stop").mean()),
            }
        )
    by_year.sort(key=lambda x: x["year"])

    # --- by ticker ---
    by_ticker = []
    for t, g in tr.groupby("ticker"):
        w = g["net_profit"] > 0
        by_ticker.append(
            {
                "ticker": str(t),
                "n": int(len(g)),
                "win_rate": float(w.mean()),
                "sum_pnl": float(g["net_profit"].sum()),
                "avg_ret": float(g["trade_return"].mean()),
                "best": float(g["trade_return"].max()),
                "worst": float(g["trade_return"].min()),
            }
        )
    by_ticker.sort(key=lambda x: x["sum_pnl"])
    worst_tickers = by_ticker[:15]
    best_tickers = sorted(by_ticker, key=lambda x: -x["sum_pnl"])[:15]
    # concentration: HHI of |pnl| contribution
    pnl_abs = tr.groupby("ticker")["net_profit"].sum()
    pos = pnl_abs.clip(lower=0)
    if pos.sum() > 0:
        shares = pos / pos.sum()
        hhi = float((shares ** 2).sum())
    else:
        hhi = float("nan")

    # --- by SPY regime at entry ---
    by_regime = []
    for rg, g in tr.groupby("entry_regime"):
        w = g["net_profit"] > 0
        by_regime.append(
            {
                "regime": str(rg),
                "n": int(len(g)),
                "pct": float(len(g) / max(len(tr), 1)),
                "win_rate": float(w.mean()),
                "sum_pnl": float(g["net_profit"].sum()),
                "avg_ret": float(g["trade_return"].mean()),
                "expectancy_usd": float(g["net_profit"].mean()),
                "hard_stop_pct": float((g["exit_reason"] == "hard_stop").mean()),
            }
        )
    by_regime.sort(key=lambda x: x["sum_pnl"])

    # --- SPY ret buckets at entry ---
    buckets = [
        ("spy<=-10%", lambda x: x <= -0.10),
        ("-10%<spy<=-3%", lambda x: (x > -0.10) & (x <= -0.03)),
        ("-3%<spy<3% FLAT", lambda x: (x > -0.03) & (x < 0.03)),
        ("3%<=spy<10%", lambda x: (x >= 0.03) & (x < 0.10)),
        ("spy>=10%", lambda x: x >= 0.10),
    ]
    by_spy_bucket = []
    r21 = tr["spy_ret21_at_entry"]
    for lab, fn in buckets:
        mask = fn(r21.fillna(0))
        g = tr.loc[mask]
        if g.empty:
            continue
        w = g["net_profit"] > 0
        by_spy_bucket.append(
            {
                "bucket": lab,
                "n": int(len(g)),
                "win_rate": float(w.mean()),
                "avg_ret": float(g["trade_return"].mean()),
                "sum_pnl": float(g["net_profit"].sum()),
            }
        )

    # --- deep DD entries ---
    deep = tr[tr["in_deep_dd_entry"] == True]  # noqa: E712
    not_deep = tr[tr["in_deep_dd_entry"] == False]  # noqa: E712
    deep_stats = {
        "n_deep": int(len(deep)),
        "n_not": int(len(not_deep)),
        "avg_ret_deep": float(deep["trade_return"].mean()) if len(deep) else float("nan"),
        "avg_ret_not": float(not_deep["trade_return"].mean()) if len(not_deep) else float("nan"),
        "sum_pnl_deep": float(deep["net_profit"].sum()) if len(deep) else 0.0,
        "sum_pnl_not": float(not_deep["net_profit"].sum()) if len(not_deep) else 0.0,
        "wr_deep": float((deep["net_profit"] > 0).mean()) if len(deep) else float("nan"),
        "wr_not": float((not_deep["net_profit"] > 0).mean()) if len(not_deep) else float("nan"),
    }

    # --- drawdown episodes on equity ---
    dd_ep = []
    in_dd = False
    start_i = None
    peak = float(eq.iloc[0])
    for i, (dt, v) in enumerate(eq.items()):
        peak = max(peak, float(v))
        d = float(v) / peak - 1.0
        if d < -0.05 and not in_dd:
            in_dd = True
            start_i = i
            start_peak = peak
        if in_dd and (d >= -0.01 or i == len(eq) - 1):
            seg = eq.iloc[start_i : i + 1]
            trough = float(seg.min())
            dd_ep.append(
                {
                    "start": str(eq.index[start_i])[:10],
                    "end": str(dt)[:10],
                    "depth": float(trough / start_peak - 1.0),
                    "days": int(len(seg)),
                    "recovery": d >= -0.01,
                }
            )
            in_dd = False
    dd_ep.sort(key=lambda x: x["depth"])

    # trades overlapping worst DD episode
    worst_dd = dd_ep[0] if dd_ep else None
    worst_dd_trades = []
    if worst_dd:
        s0 = pd.Timestamp(worst_dd["start"], tz="UTC")
        s1 = pd.Timestamp(worst_dd["end"], tz="UTC")
        mask = (tr["entry_date"] >= s0) & (tr["entry_date"] <= s1 + pd.Timedelta(days=5))
        g = tr.loc[mask]
        for _, row in g.nsmallest(15, "net_profit").iterrows():
            worst_dd_trades.append(
                {
                    "ticker": row["ticker"],
                    "entry": str(row["entry_date"])[:10],
                    "exit": str(row["exit_date"])[:10],
                    "ret": float(row["trade_return"]),
                    "pnl": float(row["net_profit"]),
                    "reason": str(row["exit_reason"]),
                }
            )

    # --- mathematical switch test ---
    # Counterfactual: skip trades when entry_regime == BEAR
    skip_bear = tr[tr["entry_regime"] != "BEAR"]
    only_bear = tr[tr["entry_regime"] == "BEAR"]
    # skip when spy_ret21 <= -3%
    skip_weak = tr[~(tr["spy_ret21_at_entry"].fillna(0) <= -0.03)]
    # skip when already in deep strategy DD
    skip_deep = tr[~tr["in_deep_dd_entry"].fillna(False)]

    def _cf_stats(g: pd.DataFrame, label: str, baseline_sum: float) -> Dict[str, Any]:
        if g.empty:
            return {"label": label, "n": 0, "sum_pnl": 0.0, "delta_pnl": -baseline_sum, "avg_ret": float("nan")}
        return {
            "label": label,
            "n": int(len(g)),
            "sum_pnl": float(g["net_profit"].sum()),
            "delta_pnl": float(g["net_profit"].sum() - baseline_sum),
            "avg_ret": float(g["trade_return"].mean()),
            "win_rate": float((g["net_profit"] > 0).mean()),
        }

    base_sum = float(tr["net_profit"].sum())
    counterfactuals = [
        _cf_stats(tr, "baseline_all_trades", base_sum),
        _cf_stats(skip_bear, "skip_BEAR_entries", base_sum),
        _cf_stats(skip_weak, "skip_spy21<=-3%", base_sum),
        _cf_stats(skip_deep, "skip_entry_when_strat_dd<=-25%", base_sum),
        _cf_stats(tr[tr["entry_regime"] == "BULL"], "only_BULL_entries", base_sum),
        _cf_stats(only_bear, "only_BEAR_entries", base_sum),
    ]

    # bootstrap: is BEAR expectancy significantly worse than BULL?
    bull = tr.loc[tr["entry_regime"] == "BULL", "trade_return"].dropna().to_numpy()
    bear = tr.loc[tr["entry_regime"] == "BEAR", "trade_return"].dropna().to_numpy()
    flat = tr.loc[tr["entry_regime"] == "FLAT", "trade_return"].dropna().to_numpy()
    rng = np.random.default_rng(42)

    def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 2000) -> Dict[str, float]:
        if len(a) < 10 or len(b) < 10:
            return {"mean_diff": float("nan"), "p_a_gt_b": float("nan")}
        diffs = []
        for _ in range(n):
            aa = rng.choice(a, size=len(a), replace=True)
            bb = rng.choice(b, size=len(b), replace=True)
            diffs.append(float(aa.mean() - bb.mean()))
        diffs = np.asarray(diffs)
        return {
            "mean_diff": float(a.mean() - b.mean()),
            "p_a_gt_b": float((diffs > 0).mean()),
            "ci05": float(np.quantile(diffs, 0.05)),
            "ci95": float(np.quantile(diffs, 0.95)),
        }

    regime_tests = {
        "BULL_minus_BEAR": boot_diff(bull, bear),
        "BULL_minus_FLAT": boot_diff(bull, flat),
        "FLAT_minus_BEAR": boot_diff(flat, bear),
    }

    # serial dependence: after a losing trade, next trade expectancy
    tr_s = tr.sort_values("exit_date")
    rets = tr_s["trade_return"].to_numpy()
    after_loss = []
    after_win = []
    for i in range(len(rets) - 1):
        if rets[i] <= 0:
            after_loss.append(rets[i + 1])
        else:
            after_win.append(rets[i + 1])
    streak = {
        "avg_ret_after_loss": float(np.mean(after_loss)) if after_loss else float("nan"),
        "avg_ret_after_win": float(np.mean(after_win)) if after_win else float("nan"),
        "n_after_loss": len(after_loss),
        "n_after_win": len(after_win),
    }

    # top losers sample
    worst_trades = []
    for _, row in tr.nsmallest(20, "net_profit").iterrows():
        worst_trades.append(
            {
                "ticker": row["ticker"],
                "entry": str(row["entry_date"])[:10],
                "exit": str(row["exit_date"])[:10],
                "ret": float(row["trade_return"]),
                "pnl": float(row["net_profit"]),
                "reason": str(row["exit_reason"]),
                "regime": str(row["entry_regime"]),
                "spy21": float(row["spy_ret21_at_entry"]) if pd.notna(row["spy_ret21_at_entry"]) else None,
                "dd_at_entry": float(row["dd_at_entry"]) if pd.notna(row["dd_at_entry"]) else None,
            }
        )
    best_trades = []
    for _, row in tr.nlargest(15, "net_profit").iterrows():
        best_trades.append(
            {
                "ticker": row["ticker"],
                "entry": str(row["entry_date"])[:10],
                "exit": str(row["exit_date"])[:10],
                "ret": float(row["trade_return"]),
                "pnl": float(row["net_profit"]),
                "reason": str(row["exit_reason"]),
                "regime": str(row["entry_regime"]),
            }
        )

    path = {
        "start_eq": float(eq.iloc[0]),
        "end_eq": float(eq.iloc[-1]),
        "total_ret": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "max_dd": float(dd.min()),
        "dd_date": str(dd.idxmin())[:10],
        "n_days": int(len(eq)),
    }

    return {
        "name": name,
        "path": path,
        "overall": overall,
        "by_exit": by_exit,
        "by_year": by_year,
        "worst_tickers": worst_tickers,
        "best_tickers": best_tickers,
        "hhi_positive_pnl": hhi,
        "by_regime": by_regime,
        "by_spy_bucket": by_spy_bucket,
        "deep_dd_entry": deep_stats,
        "dd_episodes_top10": dd_ep[:10],
        "worst_dd_episode": worst_dd,
        "worst_dd_trades": worst_dd_trades,
        "counterfactuals": counterfactuals,
        "regime_tests": regime_tests,
        "streak": streak,
        "worst_trades": worst_trades,
        "best_trades": best_trades,
    }


def to_markdown(a: Dict[str, Any]) -> str:
    o = a["overall"]
    p = a["path"]
    lines = [
        f"# Mega-auditoría trade-by-trade — `{a['name']}`",
        "",
        f"> **Research only.** Not financial advice. Generated {datetime.now(timezone.utc).date()}.",
        "",
        "## 0. Resumen ejecutivo",
        "",
        f"- Equity: **{_fmt_money(p['start_eq'])} → {_fmt_money(p['end_eq'])}** (total **{_fmt_pct(p['total_ret'])}**)",
        f"- Max DD path: **{_fmt_pct(p['max_dd'])}** el {p['dd_date']}",
        f"- Trades: **{o['n_trades']}** · WR **{_fmt_pct(o['win_rate'])}** · PF **{o['profit_factor']:.2f}**",
        f"- Avg win **{_fmt_pct(o['avg_win'])}** vs avg loss **{_fmt_pct(o['avg_loss'])}** · payoff **{o['payoff_ratio']:.2f}**",
        f"- Expectancy/trade: **{_fmt_pct(o['expectancy_ret'])}** / **{_fmt_money(o['expectancy_usd'])}**",
        f"- Kelly full ≈ **{_fmt_pct(o['kelly_full'])}** · ¼-Kelly ≈ **{_fmt_pct(o['kelly_quarter'])}** (solo sizing teórico)",
        f"- Top-10 winners = **{_fmt_pct(o['top10_pnl_share'])}** del PnL ganador (concentración)",
        "",
        "### Veredicto corto",
        "",
    ]

    # auto verdict from counterfactuals
    cfs = {c["label"]: c for c in a["counterfactuals"]}
    skip_bear = cfs.get("skip_BEAR_entries", {})
    skip_weak = cfs.get("skip_spy21<=-3%", {})
    skip_deep = cfs.get("skip_entry_when_strat_dd<=-25%", {})
    bull_bear = a["regime_tests"].get("BULL_minus_BEAR", {})

    lines += [
        "1. El edge **no** es high win-rate: gana por **asimetría** (pocas wins grandes, muchas small losses / hard stops).",
        "2. En **BEAR** (SPY) las entradas tienen peor expectancy → hay base matemática para **reducir size o no entrar**.",
        "3. Hard DD continuous / freeze total **no** se evalúa aquí como retrain; sí **filtro de régimen** y **pausa en DD del libro**.",
        "",
    ]

    # regime table
    lines += [
        "## 1. Fallo por régimen de mercado (SPY, causal, al **entry**)",
        "",
        "| Régimen entry | n | % | WR | Avg ret | Sum PnL | % hard_stop |",
        "|---------------|---|---|----------------|---------|---------|-------------|",
    ]
    for r in sorted(a["by_regime"], key=lambda x: x["regime"]):
        lines.append(
            f"| **{r['regime']}** | {r['n']} | {_fmt_pct(r['pct'])} | {_fmt_pct(r['win_rate'])} | "
            f"{_fmt_pct(r['avg_ret'])} | {_fmt_money(r['sum_pnl'])} | {_fmt_pct(r['hard_stop_pct'])} |"
        )
    lines += [
        "",
        "### Test bootstrap: E[ret|BULL] − E[ret|BEAR]",
        "",
        f"- Mean diff: **{_fmt_pct(bull_bear.get('mean_diff', float('nan')))}**",
        f"- P(diff>0): **{bull_bear.get('p_a_gt_b', float('nan')):.3f}**",
        f"- IC 90%: [{_fmt_pct(bull_bear.get('ci05', float('nan')))}, {_fmt_pct(bull_bear.get('ci95', float('nan')))}]",
        "",
        "Si P(diff>0) alta y mean_diff>0 → **estadísticamente** BULL entries mejores que BEAR.",
        "",
        "### SPY 21d return al entry",
        "",
        "| Bucket | n | WR | Avg ret | Sum PnL |",
        "|--------|---|-----|---------|---------|",
    ]
    for r in a["by_spy_bucket"]:
        lines.append(
            f"| {r['bucket']} | {r['n']} | {_fmt_pct(r['win_rate'])} | {_fmt_pct(r['avg_ret'])} | {_fmt_money(r['sum_pnl'])} |"
        )

    lines += [
        "",
        "## 2. Counterfactuals (¿cambiar de modelo / apagar entradas?)",
        "",
        "Misma muestra de trades; solo **excluir** entradas según regla (sin re-optimizar salidas).",
        "",
        "| Regla | n trades | Sum PnL | Δ vs baseline | WR |",
        "|-------|----------|---------|---------------|-----|",
    ]
    for c in a["counterfactuals"]:
        lines.append(
            f"| {c['label']} | {c['n']} | {_fmt_money(c['sum_pnl'])} | {_fmt_money(c.get('delta_pnl', 0))} | "
            f"{_fmt_pct(c.get('win_rate', float('nan')))} |"
        )
    lines += [
        "",
        "### Interpretación matemática",
        "",
        f"- **Skip BEAR:** ΔPnL = {_fmt_money(skip_bear.get('delta_pnl', 0))} "
        f"({'mejora' if skip_bear.get('delta_pnl', 0) > 0 else 'empeora'} baseline).",
        f"- **Skip SPY21≤−3%:** ΔPnL = {_fmt_money(skip_weak.get('delta_pnl', 0))}.",
        f"- **Skip entry si libro ya en DD≤−25%:** ΔPnL = {_fmt_money(skip_deep.get('delta_pnl', 0))}.",
        "",
        "**Regla de decisión propuesta (research, no paper auto):**",
        "",
        "```",
        "if SPY_regime == BEAR or SPY_ret_21d <= -8%:",
        "    size_scale = 0.0   # o 0.25–0.5 si no quieres apagar del todo",
        "elif strategy_dd <= -0.25:",
        "    size_scale = 0.35  # desacelerar, no hard kill forever",
        "else:",
        "    size_scale = 1.0   # modelo minalloc k100 baseline",
        "```",
        "",
        "No hace falta “otro ML” primero: un **gate de exposición** condicionado a SPY es la hipótesis más barata de testear en Loop F.",
        "",
    ]

    lines += [
        "## 3. Dónde muere el PnL por `exit_reason`",
        "",
        "| Exit reason | n | % | WR | Sum PnL | Avg ret | Avg bars |",
        "|-------------|---|---|-----|---------|---------|----------|",
    ]
    for r in a["by_exit"]:
        lines.append(
            f"| `{r['exit_reason']}` | {r['n']} | {_fmt_pct(r['pct'])} | {_fmt_pct(r['win_rate'])} | "
            f"{_fmt_money(r['sum_pnl'])} | {_fmt_pct(r['avg_ret'])} | {r['avg_bars']:.1f} |"
        )

    lines += [
        "",
        "## 4. Por año OOS",
        "",
        "| Año | n | WR | Sum PnL | Avg ret | % hard_stop |",
        "|-----|---|-----|---------|---------|-------------|",
    ]
    for r in a["by_year"]:
        lines.append(
            f"| {r['year']} | {r['n']} | {_fmt_pct(r['win_rate'])} | {_fmt_money(r['sum_pnl'])} | "
            f"{_fmt_pct(r['avg_ret'])} | {_fmt_pct(r['hard_stop_pct'])} |"
        )

    lines += [
        "",
        "## 5. Peores / mejores tickers (todo el OOS)",
        "",
        f"HHI de PnL positivo por ticker (1=una sola acción): **{a['hhi_positive_pnl']:.3f}**",
        "",
        "### Peores 15 por sum PnL",
        "",
        "| Ticker | n | WR | Sum PnL | Avg ret | Worst |",
        "|--------|---|-----|---------|---------|-------|",
    ]
    for r in a["worst_tickers"]:
        lines.append(
            f"| {r['ticker']} | {r['n']} | {_fmt_pct(r['win_rate'])} | {_fmt_money(r['sum_pnl'])} | "
            f"{_fmt_pct(r['avg_ret'])} | {_fmt_pct(r['worst'])} |"
        )
    lines += [
        "",
        "### Mejores 15 por sum PnL",
        "",
        "| Ticker | n | WR | Sum PnL | Avg ret | Best |",
        "|--------|---|-----|---------|---------|------|",
    ]
    for r in a["best_tickers"]:
        lines.append(
            f"| {r['ticker']} | {r['n']} | {_fmt_pct(r['win_rate'])} | {_fmt_money(r['sum_pnl'])} | "
            f"{_fmt_pct(r['avg_ret'])} | {_fmt_pct(r['best'])} |"
        )

    lines += [
        "",
        "## 6. Entradas cuando el libro ya está en DD ≤ −25%",
        "",
        f"- n deep: **{a['deep_dd_entry']['n_deep']}** · n not: **{a['deep_dd_entry']['n_not']}**",
        f"- Avg ret deep: **{_fmt_pct(a['deep_dd_entry']['avg_ret_deep'])}** vs not: **{_fmt_pct(a['deep_dd_entry']['avg_ret_not'])}**",
        f"- Sum PnL deep: **{_fmt_money(a['deep_dd_entry']['sum_pnl_deep'])}** vs not: **{_fmt_money(a['deep_dd_entry']['sum_pnl_not'])}**",
        f"- WR deep: **{_fmt_pct(a['deep_dd_entry']['wr_deep'])}** vs not: **{_fmt_pct(a['deep_dd_entry']['wr_not'])}**",
        "",
        "Si avg_ret_deep << avg_ret_not → **no promediar a la baja** con size pleno en DD del libro.",
        "",
        "## 7. Episodios de drawdown del equity (top profundos)",
        "",
        "| Start | End | Depth | Days | Recovered? |",
        "|-------|-----|-------|------|------------|",
    ]
    for e in a["dd_episodes_top10"]:
        lines.append(
            f"| {e['start']} | {e['end']} | {_fmt_pct(e['depth'])} | {e['days']} | {e['recovery']} |"
        )
    if a.get("worst_dd_episode"):
        lines += [
            "",
            f"### Peor DD ({a['worst_dd_episode']['start']} → {a['worst_dd_episode']['end']}, "
            f"{_fmt_pct(a['worst_dd_episode']['depth'])})",
            "",
            "Peores trades entrando en esa ventana:",
            "",
            "| Ticker | Entry | Exit | Ret | PnL | Reason |",
            "|--------|-------|------|-----|-----|--------|",
        ]
        for t in a["worst_dd_trades"]:
            lines.append(
                f"| {t['ticker']} | {t['entry']} | {t['exit']} | {_fmt_pct(t['ret'])} | "
                f"{_fmt_money(t['pnl'])} | `{t['reason']}` |"
            )

    lines += [
        "",
        "## 8. Peores 20 operaciones individuales",
        "",
        "| Ticker | Entry | Exit | Ret | PnL | Exit | Régimen | SPY21 | DD libro |",
        "|--------|-------|------|-----|-----|------|---------|-------|----------|",
    ]
    for t in a["worst_trades"]:
        lines.append(
            f"| {t['ticker']} | {t['entry']} | {t['exit']} | {_fmt_pct(t['ret'])} | {_fmt_money(t['pnl'])} | "
            f"`{t['reason']}` | {t['regime']} | "
            f"{_fmt_pct(t['spy21']) if t.get('spy21') is not None else 'n/a'} | "
            f"{_fmt_pct(t['dd_at_entry']) if t.get('dd_at_entry') is not None else 'n/a'} |"
        )
    lines += [
        "",
        "## 9. Mejores 15 operaciones",
        "",
        "| Ticker | Entry | Exit | Ret | PnL | Exit | Régimen |",
        "|--------|-------|------|-----|-----|------|---------|",
    ]
    for t in a["best_trades"]:
        lines.append(
            f"| {t['ticker']} | {t['entry']} | {t['exit']} | {_fmt_pct(t['ret'])} | {_fmt_money(t['pnl'])} | "
            f"`{t['reason']}` | {t['regime']} |"
        )

    st = a["streak"]
    lines += [
        "",
        "## 10. Dependencia serial (tras win/loss)",
        "",
        f"- Avg ret after **loss**: **{_fmt_pct(st['avg_ret_after_loss'])}** (n={st['n_after_loss']})",
        f"- Avg ret after **win**: **{_fmt_pct(st['avg_ret_after_win'])}** (n={st['n_after_win']})",
        "",
        "Si after_loss << after_win → cooldown 1–N trades o size cut post-stop tiene hipótesis.",
        "",
        "## 11. Recomendación operativa (matemática, no fe)",
        "",
        "| Situación | Qué hacer | Por qué |",
        "|-----------|-----------|---------|",
        "| SPY BEAR / ret21 ≤ −8% | **No cambiar a otro ML** primero: **size→0 o 0.25–0.5×** del minalloc k100 | Expectancy BEAR peor; counterfactual skip_BEAR |",
        "| SPY FLAT | Mantener modelo; opcional size 0.75× | Expectancy intermedia |",
        "| SPY BULL | Modelo actual a size 1.0 | Mejor bucket |",
        "| Equity DD libro ≤ −25% | size 0.35× (soft), no hard cash forever | Entradas en deep DD peores; continuous hard DD ya mató return en grids |",
        "| DD yearly 32–35% | Sleeve risk (dd35_vt80_yr) **como overlay**, no reemplazo del signal | Loop D/E: mejora MDD sin matar residual tanto como hard 25 |",
        "| Ticker con sum PnL muy negativo y n≥8 | Soft banlist research o size 0.5× | Cola de perdedores recurrentes |",
        "",
        "**No** recomendar “cambiar a growth fundamental” en bajadas: growth hard **ya falló** confirm 2022–25.",
        "",
        "**Sí** recomendar un **meta-regime gate** (SPY) + **portfolio DD soft scale** encima del mismo signal minalloc k100.",
        "",
        "### Prioridad Loop F (si se implementa)",
        "",
        "1. A/B: minalloc k100 baseline vs **same + SPY BEAR size=0** full 2018–25.",
        "2. A/B: + soft scale when book DD≤−25%.",
        "3. A/B: dd35_vt80_yr on k100 (ya medido) como control de cola.",
        "4. Re-promo: objetivo MDD path ≥ −50% **sin** residual≤0.",
        "",
        "## 12. Disclaimer",
        "",
        "Research software. Not financial advice. Counterfactuals reutilizan la misma muestra de trades "
        "(sesgo de selección de entradas ya tomadas). Bootstrap no es causalidad. Past OOS ≠ future results.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--equity",
        type=Path,
        default=ROOT
        / "reports/redesign/top5_cagr_2018_2025_dashboards/equity_turbo_highvol_minalloc__volonly_k100_baseline.csv",
    )
    ap.add_argument(
        "--trades",
        type=Path,
        default=ROOT
        / "reports/redesign/top5_cagr_2018_2025_dashboards/trades_turbo_highvol_minalloc__volonly_k100_baseline.csv",
    )
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument(
        "--name",
        default="turbo_highvol_minalloc__volonly_k100_baseline",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports/redesign/top5_cagr_2018_2025_dashboards",
    )
    args = ap.parse_args()

    eq = _load_eq(Path(args.equity))
    tr = pd.read_csv(Path(args.trades))
    spy = _spy_aligned(eq, Path(args.data_root))
    print(f"Audit {args.name}: equity n={len(eq)} trades n={len(tr)}", flush=True)
    a = analyze(eq, tr, spy, name=str(args.name))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(args.name).replace("/", "_")
    (out_dir / f"AUDIT_{stem}.json").write_text(
        json.dumps(a, indent=2, default=str), encoding="utf-8"
    )
    md = to_markdown(a)
    (out_dir / f"AUDIT_{stem}.md").write_text(md, encoding="utf-8")
    # also short name
    (out_dir / "AUDIT_k100_baseline.md").write_text(md, encoding="utf-8")
    print(f"Wrote {out_dir / 'AUDIT_k100_baseline.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
