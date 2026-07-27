"""Rescore SPY/QQQ excess from saved equities and rebuild longhist top5 index."""
from __future__ import annotations

import importlib.util
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from trad_research.metrics import equity_metrics
from trad_research.risk_metrics import extended_risk_from_equity
from trad_research.walk_forward import load_benchmark_equity

COMMISSION = 0.001
SLIPPAGE = 0.0005

spec = importlib.util.spec_from_file_location(
    "lh", ROOT / "scripts" / "run_longhist_top5_dashboards.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def load_eq(p: Path) -> pd.Series:
    s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
    try:
        s.index = s.index.normalize()
    except Exception:
        pass
    return s[~s.index.duplicated(keep="last")].dropna().sort_index()


def bench_total(eq: pd.Series, name: str, data_root: Path):
    b = load_benchmark_equity(
        data_root, eq.index.min(), eq.index.max(), preferred=[name]
    )
    if b is None or b.empty:
        return None
    b = b.copy()
    b.index = pd.to_datetime(b.index, utc=True).normalize()
    b = b[~b.index.duplicated(keep="last")].sort_index()
    eq2 = eq[~eq.index.duplicated(keep="last")].sort_index()
    j = pd.concat([eq2.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 3:
        return None
    return float(j["b"].iloc[-1] / float(j["b"].iloc[0]) - 1.0)


def main() -> int:
    out = ROOT / "reports" / "redesign" / "longhist_2015_top5_dashboards"
    data_root = ROOT / "data"
    first, last = 2015, 2026
    names = [
        "turbo_highvol",
        "turbo_highvol_minalloc",
        "turbo_highvol_minalloc_sector_rot",
        "turbo_highvol_minalloc_softreg",
        "turbo_strict",
    ]
    results = []
    for name in names:
        eq = load_eq(out / f"equity_{name}.csv")
        tpath = out / f"trades_{name}.csv"
        tdf = pd.read_csv(tpath) if tpath.is_file() else pd.DataFrame()
        start_eq = float(eq.iloc[0])
        rep = equity_metrics(
            eq, start_equity=start_eq, trades=tdf if not tdf.empty else None
        )
        risk = extended_risk_from_equity(
            eq.to_numpy(),
            trade_pnls=tdf["net_profit"].to_numpy()
            if not tdf.empty and "net_profit" in tdf.columns
            else None,
        )
        total = float(eq.iloc[-1] / start_eq - 1)
        spy = bench_total(eq, "SPY", data_root)
        qqq = bench_total(eq, "QQQ", data_root)
        if not tdf.empty and "capital_used" in tdf.columns:
            entry_comm = tdf["capital_used"] * COMMISSION / (1 + COMMISSION)
            exit_comm = (tdf["capital_used"] + tdf.get("net_profit", 0)).clip(
                lower=0
            ) * COMMISSION / max(1e-9, 1 - COMMISSION)
            total_comm = float(entry_comm.sum() + exit_comm.sum())
        else:
            total_comm = 0.0
        year_results = []
        eqy = eq.copy()
        eqy.index = pd.to_datetime(eqy.index, utc=True)
        for y, g in eqy.groupby(eqy.index.year):
            if len(g) < 2:
                continue
            ntr = (
                int((tdf["oos_year"] == y).sum())
                if not tdf.empty and "oos_year" in tdf.columns
                else 0
            )
            year_results.append(
                {
                    "year": int(y),
                    "year_return": float(g.iloc[-1] / g.iloc[0] - 1),
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                    "n_trades": ntr,
                }
            )
        r = {
            "strategy": name,
            "overrides": {},
            "total_return": total,
            "cagr": rep.cagr,
            "sharpe": rep.sharpe,
            "sortino": risk.sortino,
            "max_drawdown": rep.max_drawdown,
            "calmar": risk.calmar,
            "n_trades": rep.n_trades,
            "win_rate": rep.win_rate,
            "profit_factor": rep.profit_factor,
            "final_equity": float(eq.iloc[-1]),
            "start_equity": start_eq,
            "spy_total": spy,
            "qqq_total": qqq,
            "excess_spy": (total - spy) if spy is not None else None,
            "excess_qqq": (total - qqq) if qqq is not None else None,
            "approx_total_commissions_usd": total_comm,
            "start": str(eq.index.min()),
            "end": str(eq.index.max()),
            "n_days": len(eq),
            "year_results": year_results,
            "equity": eq,
            "trades": tdf,
        }
        results.append(r)
        print(
            name,
            f"total={total:.1%}",
            f"cagr={rep.cagr:.2%}",
            f"vsSPY={r['excess_spy']}",
            f"spy={spy}",
            f"trades={rep.n_trades}",
        )

    ranked = sorted(
        results,
        key=lambda x: float(
            x.get("excess_spy")
            if x.get("excess_spy") is not None
            else x.get("total_return") or -9
        ),
        reverse=True,
    )
    for rank, r in enumerate(ranked, 1):
        safe = r["strategy"].replace("/", "_")
        (out / f"dashboard_{safe}.html").write_text(
            mod.build_one_html(r, rank, first, last), encoding="utf-8"
        )

    rows_md = []
    cards = []
    for rank, r in enumerate(ranked, 1):
        safe = r["strategy"].replace("/", "_")
        ex = r.get("excess_spy")
        exs = f"{ex:.1%}" if ex is not None else "n/a"
        rows_md.append(
            f"| {rank} | [`{r['strategy']}`](dashboard_{safe}.html) | "
            f"{r.get('total_return', 0):.1%} | {r.get('cagr', 0):.2%} | {exs} | "
            f"{r.get('sharpe', 0):.2f} | {r.get('sortino', 0):.2f} | "
            f"{r.get('max_drawdown', 0):.1%} | {r.get('n_trades')} | "
            f"{r.get('win_rate', 0):.1%} |"
        )
        cards.append(
            f"""<a class="card" href="dashboard_{safe}.html">
            <div class="rank">#{rank}</div>
            <div class="name"><code>{r['strategy']}</code></div>
            <div class="v">{r.get('cagr', 0):.1%} CAGR</div>
            <div class="k">total {r.get('total_return', 0):.0%} · vs SPY {exs}</div>
            <div class="k">Sharpe {r.get('sharpe', 0):.2f} · MDD {r.get('max_drawdown', 0):.0%} · {r.get('n_trades')} trades · WR {r.get('win_rate', 0):.0%}</div>
            </a>"""
        )

    spy_line = ""
    if ranked[0].get("spy_total") is not None:
        spy_line = (
            f"**SPY total same window:** {ranked[0]['spy_total']:.1%}  \n"
            f"**QQQ total same window:** {(ranked[0].get('qqq_total') or float('nan')):.1%}"
        )
    index_md = [
        f"# Top-5 dashboards vs SPY — OOS {first}→{last}",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Equity window:** {str(ranked[0]['start'])[:10]} → {str(ranked[0]['end'])[:10]}",
        "(Year 2015 often skipped if train history insufficient; first OOS equity typically 2016+.)",
        f"**Universe:** universe_highvol80.txt limit=80",
        "",
        "Long-only cash (no margin). Commission **0.10%** + slippage **0.05%** per side. WF annual retrain.",
        "",
        spy_line,
        "",
        "| Rank | Dashboard | Total | CAGR | vs SPY | Sharpe | Sortino | MDD | Trades | WR |",
        "|------|-----------|-------|------|--------|--------|---------|-----|--------|----|",
        *rows_md,
        "",
        "Each dashboard: equity vs SPY (log scale), yearly returns, winners >33%, full trade blotter.",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "INDEX.md").write_text("\n".join(index_md), encoding="utf-8")

    spy_note = ""
    if ranked[0].get("spy_total") is not None:
        spy_note = (
            f"<p style='color:#9fb0c3'>SPY total same window ≈ <b>{ranked[0]['spy_total']:.1%}</b> · "
            f"QQQ ≈ <b>{(ranked[0].get('qqq_total') or float('nan')):.1%}</b></p>"
        )
    index_html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<title>Top-5 vs SPY — {first}–{last}</title>
<style>
 body {{ font-family: system-ui,sans-serif; margin:24px; background:#0f1419; color:#e7ecf1; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px; }}
 a.card {{ display:block; background:#1a2332; border:1px solid #2a3a4f; border-radius:12px;
   padding:16px; text-decoration:none; color:inherit; }}
 a.card:hover {{ border-color:#38bdf8; }}
 .rank {{ color:#f97316; font-weight:800; }}
 .name {{ margin:8px 0; }}
 .v {{ font-size:22px; font-weight:700; }}
 .k {{ color:#9fb0c3; font-size:13px; margin-top:4px; }}
</style></head><body>
<h1>Top-5 vs SPY — OOS {first} → {last}</h1>
<p style="color:#9fb0c3">Long-only cash · comisión 0.10% + slip 0.05%/lado · highvol80 · WF anual</p>
{spy_note}
<div class="grid">{''.join(cards)}</div>
<p style="color:#9fb0c3;margin-top:24px">Research only. Updated {date.today().isoformat()}. Equity starts ~2016 if 2015 train skipped.</p>
</body></html>"""
    (out / "index.html").write_text(index_html, encoding="utf-8")

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "first_oos": first,
        "last_oos": last,
        "leverage_borrowed": False,
        "commission": COMMISSION,
        "slippage": SLIPPAGE,
        "spy_total": ranked[0].get("spy_total"),
        "qqq_total": ranked[0].get("qqq_total"),
        "strategies": [
            {
                "rank": i,
                "name": r["strategy"],
                "total_return": r.get("total_return"),
                "cagr": r.get("cagr"),
                "excess_spy": r.get("excess_spy"),
                "spy_total": r.get("spy_total"),
                "qqq_total": r.get("qqq_total"),
                "excess_qqq": r.get("excess_qqq"),
                "sharpe": r.get("sharpe"),
                "sortino": r.get("sortino"),
                "max_drawdown": r.get("max_drawdown"),
                "n_trades": r.get("n_trades"),
                "win_rate": r.get("win_rate"),
                "start": r.get("start"),
                "end": r.get("end"),
                "dashboard": f"dashboard_{r['strategy'].replace('/', '_')}.html",
            }
            for i, r in enumerate(ranked, 1)
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print((out / "INDEX.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
