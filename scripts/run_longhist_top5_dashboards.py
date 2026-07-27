"""Run top-5 SPY-beaters OOS 2015→now and build trade dashboards + index.

Strategies (default): turbo_highvol family + turbo_strict from 2026 YTD ranking.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.backtest import BacktestConfig
from trad_research.metrics import equity_metrics
from trad_research.risk_metrics import extended_risk_from_equity
from trad_research.strategies import get_strategy
from trad_research.strategy_runner import run_strategy_walk_forward
from trad_research.walk_forward import load_benchmark_equity

_spec = importlib.util.spec_from_file_location(
    "run_ytd_trade_dashboard", ROOT / "scripts" / "run_ytd_trade_dashboard.py"
)
_dash = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dash)
COMMISSION = _dash.COMMISSION
SLIPPAGE = _dash.SLIPPAGE
_trades_html_table = _dash._trades_html_table
_equity_js = _dash._equity_js

TOP5 = [
    "turbo_highvol",
    "turbo_highvol_minalloc",
    "turbo_highvol_minalloc_sector_rot",
    "turbo_highvol_minalloc_softreg",
    "turbo_strict",
]


def _bench_total(equity: pd.Series, data_root: Path, name: str) -> Optional[float]:
    try:
        b = load_benchmark_equity(
            data_root, equity.index.min(), equity.index.max(), preferred=[name]
        )
        if b is None or b.empty:
            return None
        b = b.copy()
        b.index = pd.to_datetime(b.index, utc=True).normalize()
        b = b[~b.index.duplicated(keep="last")].sort_index()
        eq = equity.copy()
        eq.index = pd.to_datetime(eq.index, utc=True).normalize()
        eq = eq[~eq.index.duplicated(keep="last")].sort_index()
        j = pd.concat([eq.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
        if len(j) < 3:
            return None
        return float(j["b"].iloc[-1] / j["b"].iloc[0] - 1.0)
    except Exception:
        return None


def run_one(
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
    base_ov = strat.backtest_overrides()
    merged = {
        **base_ov,
        "commission": COMMISSION,
        "slippage": SLIPPAGE,
    }

    def _ov() -> Dict[str, Any]:
        return dict(merged)

    orig = strat.backtest_overrides
    strat.backtest_overrides = _ov  # type: ignore[method-assign]
    try:
        res = run_strategy_walk_forward(
            strat,
            data_root=data_root,
            ticker_file=ticker_file,
            universe_limit=universe_limit,
            first_oos_year=first,
            last_oos_year=last,
            min_train_rows=min_train_rows,
            preferred_index=["SPY", "QQQ"],
            base_bt=BacktestConfig(commission=COMMISSION, slippage=SLIPPAGE),
        )
    finally:
        strat.backtest_overrides = orig  # type: ignore[method-assign]

    eq = res.get("equity")
    trades = res.get("trades")
    if eq is None or (hasattr(eq, "empty") and eq.empty):
        return {"strategy": name, "error": "empty equity"}
    eq = eq.dropna().astype(float)
    start_eq = float(eq.iloc[0])
    tdf = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame()
    rep = equity_metrics(eq, start_equity=start_eq, trades=tdf if not tdf.empty else None)
    risk = extended_risk_from_equity(
        eq.to_numpy(),
        trade_pnls=tdf["net_profit"].to_numpy() if not tdf.empty and "net_profit" in tdf.columns else None,
    )
    total = float(eq.iloc[-1] / start_eq - 1.0)
    spy_tot = _bench_total(eq, data_root, "SPY")
    qqq_tot = _bench_total(eq, data_root, "QQQ")

    if not tdf.empty and "capital_used" in tdf.columns:
        entry_comm = tdf["capital_used"] * COMMISSION / (1.0 + COMMISSION)
        if "net_profit" in tdf.columns:
            exit_comm = (tdf["capital_used"] + tdf["net_profit"]).clip(lower=0) * COMMISSION / max(
                1e-9, (1.0 - COMMISSION)
            )
        else:
            exit_comm = tdf["capital_used"] * COMMISSION
        total_comm = float(entry_comm.sum() + exit_comm.sum())
    else:
        total_comm = 0.0

    return {
        "strategy": name,
        "overrides": merged,
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
        "spy_total": spy_tot,
        "qqq_total": qqq_tot,
        "excess_spy": (total - spy_tot) if spy_tot is not None else None,
        "excess_qqq": (total - qqq_tot) if qqq_tot is not None else None,
        "approx_total_commissions_usd": total_comm,
        "commission_rate": COMMISSION,
        "slippage_rate": SLIPPAGE,
        "start": str(eq.index.min()),
        "end": str(eq.index.max()),
        "n_days": int(len(eq)),
        "year_results": res.get("year_results"),
        "equity": eq,
        "trades": tdf,
    }


def build_one_html(r: Dict[str, Any], rank: int, first: int, last: int) -> str:
    if r.get("error"):
        return f"<html><body><h1>{r.get('strategy')}</h1><p>{r['error']}</p></body></html>"
    tdf = r["trades"]
    wins = int((tdf["net_profit"] > 0).sum()) if not tdf.empty and "net_profit" in tdf.columns else 0
    losses = int((tdf["net_profit"] <= 0).sum()) if not tdf.empty and "net_profit" in tdf.columns else 0
    wr = wins / max(wins + losses, 1)
    gp = float(tdf.loc[tdf["net_profit"] > 0, "net_profit"].sum()) if wins else 0.0
    gl = float(-tdf.loc[tdf["net_profit"] <= 0, "net_profit"].sum()) if losses else 0.0

    big = []
    if not tdf.empty and "trade_return" in tdf.columns:
        b = tdf[tdf["trade_return"] > 0.33].sort_values("trade_return", ascending=False)
        for _, row in b.head(12).iterrows():
            big.append(
                f"<li><b>{row.get('ticker')}</b> "
                f"{str(row.get('entry_date'))[:10]} → {str(row.get('exit_date'))[:10]} · "
                f"PnL ${float(row.get('net_profit') or 0):,.0f} · "
                f"{float(row.get('trade_return') or 0):.1%} · {row.get('exit_reason')}</li>"
            )
    big_html = "<ul>" + ("".join(big) if big else "<li><em>Ninguna op &gt;33%</em></li>") + "</ul>"

    # yearly table
    years_html = ""
    yr = r.get("year_results") or []
    if yr:
        rows = []
        for y in yr:
            rows.append(
                f"<tr><td>{y.get('year')}</td>"
                f"<td>{float(y.get('year_return') or 0):.1%}</td>"
                f"<td>{float(y.get('sharpe') or 0):.2f}</td>"
                f"<td>{float(y.get('max_drawdown') or 0):.1%}</td>"
                f"<td>{y.get('n_trades')}</td></tr>"
            )
        years_html = (
            "<h2>Por año OOS</h2><table><thead><tr>"
            "<th>Año</th><th>Retorno</th><th>Sharpe</th><th>MDD</th><th>Trades</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )

    eq_js = _equity_js(r["equity"], r["strategy"])
    spy_js = "null"
    try:
        eq = r["equity"]
        b = load_benchmark_equity(
            ROOT / "data", eq.index.min(), eq.index.max(), preferred=["SPY"]
        )
        b.index = pd.to_datetime(b.index, utc=True).normalize()
        e2 = eq.copy()
        e2.index = pd.to_datetime(e2.index, utc=True).normalize()
        j = pd.concat([e2.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
        if len(j) > 2:
            s0 = float(j["s"].iloc[0])
            b0 = float(j["b"].iloc[0])
            spy_y = (j["b"] / b0 * s0).tolist()
            spy_x = [d.strftime("%Y-%m-%d") for d in j.index]
            spy_js = json.dumps(
                {"name": "SPY scaled", "x": spy_x, "y": [round(v, 2) for v in spy_y]}
            )
    except Exception:
        pass

    ov = r.get("overrides") or {}
    return f"""<!DOCTYPE html>
<html lang="es"><head>
<meta charset="utf-8"/>
<title>#{rank} {r['strategy']} — {first}–{last} OOS vs SPY</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
 body {{ font-family: system-ui,Segoe UI,sans-serif; margin: 24px; background:#0f1419; color:#e7ecf1; }}
 h1,h2 {{ color:#f3f6f9; }}
 .cards {{ display:flex; flex-wrap:wrap; gap:12px; margin:12px 0 20px; }}
 .card {{ background:#1a2332; border:1px solid #2a3a4f; border-radius:10px; padding:12px 16px; min-width:130px; }}
 .card .k {{ font-size:12px; color:#9fb0c3; }} .card .v {{ font-size:18px; font-weight:700; margin-top:4px; }}
 table {{ border-collapse:collapse; width:100%; font-size:12px; margin-bottom:20px; }}
 th,td {{ border:1px solid #2a3a4f; padding:5px 7px; text-align:right; }}
 th {{ background:#1a2332; }} td:first-child, th:first-child {{ text-align:left; }}
 tr.win td {{ background:rgba(34,197,94,0.08); }} tr.loss td {{ background:rgba(239,68,68,0.10); }}
 .meta {{ color:#9fb0c3; font-size:13px; }}
 .ok {{ background:#14301f; border-left:4px solid #22c55e; padding:12px 16px; margin:16px 0; }}
 a {{ color:#7dd3fc; }}
</style></head><body>
<p class="meta"><a href="index.html">← Índice top-5</a></p>
<h1>#{rank} <code>{r['strategy']}</code> — OOS {first}→{last}</h1>
<p class="meta">Long-only cash · comisión {COMMISSION:.2%} + slip {SLIPPAGE:.2%}/lado · highvol80 ·
ventana {str(r.get('start'))[:10]} → {str(r.get('end'))[:10]}</p>
<div class="ok">
<strong>Total:</strong> {r['total_return']:.2%} &nbsp;|&nbsp;
<strong>CAGR:</strong> {r['cagr']:.2%} &nbsp;|&nbsp;
<strong>vs SPY:</strong> {(r.get('excess_spy') if r.get('excess_spy') is not None else float('nan')):.2%} &nbsp;|&nbsp;
<strong>SPY total:</strong> {(r.get('spy_total') if r.get('spy_total') is not None else float('nan')):.2%}
</div>
<div class="cards">
  <div class="card"><div class="k">Total</div><div class="v">{r['total_return']:.1%}</div></div>
  <div class="card"><div class="k">CAGR</div><div class="v">{r['cagr']:.2%}</div></div>
  <div class="card"><div class="k">Sharpe</div><div class="v">{r['sharpe']:.2f}</div></div>
  <div class="card"><div class="k">Sortino</div><div class="v">{r['sortino']:.2f}</div></div>
  <div class="card"><div class="k">Max DD</div><div class="v">{r['max_drawdown']:.2%}</div></div>
  <div class="card"><div class="k">Trades</div><div class="v">{r['n_trades']} (W{wins}/L{losses})</div></div>
  <div class="card"><div class="k">Win rate</div><div class="v">{wr:.1%}</div></div>
  <div class="card"><div class="k">PF</div><div class="v">{r['profit_factor']:.2f}</div></div>
  <div class="card"><div class="k">vs SPY</div><div class="v">{(r.get('excess_spy') if r.get('excess_spy') is not None else float('nan')):.1%}</div></div>
  <div class="card"><div class="k">vs QQQ</div><div class="v">{(r.get('excess_qqq') if r.get('excess_qqq') is not None else float('nan')):.1%}</div></div>
  <div class="card"><div class="k">Comisiones ≈</div><div class="v">${r.get('approx_total_commissions_usd',0):,.0f}</div></div>
</div>
<p class="meta">Sizing: vol_target={ov.get('volatility_target_pct')} · max_pos={ov.get('max_position_pct')} · max_positions={ov.get('max_positions')}</p>
<p class="meta">Wins ${gp:,.0f} · Losses ${gl:,.0f} · Net ${gp-gl:,.0f}</p>
<div id="eqchart" style="height:400px;margin:16px 0;"></div>
{years_html}
<h2>Ganadoras &gt; 33% (top 12)</h2>
{big_html}
<h2>Todas las operaciones ({r['n_trades']})</h2>
{_trades_html_table(tdf)}
<p class="meta">Research only. No es consejo financiero. WF anual, train ≤ year-start.</p>
<script>
const E = {eq_js};
const S = {spy_js};
const traces = [{{x:E.x,y:E.y,name:E.name,type:'scatter',mode:'lines',line:{{color:'#f97316',width:2}}}}];
if (S) traces.push({{x:S.x,y:S.y,name:S.name,type:'scatter',mode:'lines',line:{{color:'#94a3b8',width:1.5,dash:'dot'}}}});
Plotly.newPlot('eqchart', traces, {{
  paper_bgcolor:'#0f1419', plot_bgcolor:'#0f1419', font:{{color:'#e7ecf1'}},
  title:'Equity {first}–{last} vs SPY', legend:{{orientation:'h'}}, margin:{{t:40,r:20,b:40,l:60}},
  xaxis:{{gridcolor:'#1f2a37'}}, yaxis:{{gridcolor:'#1f2a37', tickprefix:'$', type:'log'}}
}}, {{responsive:true}});
</script>
</body></html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-oos", type=int, default=2015)
    ap.add_argument("--last-oos", type=int, default=date.today().year)
    ap.add_argument("--strategies", type=str, default=",".join(TOP5))
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--ticker-file", type=Path, default=ROOT / "universe_highvol80.txt")
    ap.add_argument("--universe-limit", type=int, default=80)
    ap.add_argument("--min-train-rows", type=int, default=3000)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "longhist_2015_top5_dashboards",
    )
    args = ap.parse_args(argv)

    first, last = int(args.first_oos), int(args.last_oos)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    results: List[Dict[str, Any]] = []

    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name} OOS {first}-{last} …", flush=True)
        r = run_one(
            name,
            first=first,
            last=last,
            data_root=Path(args.data_root),
            ticker_file=Path(args.ticker_file),
            universe_limit=int(args.universe_limit),
            min_train_rows=int(args.min_train_rows),
        )
        print(
            f"  total={r.get('total_return')} cagr={r.get('cagr')} "
            f"vsSPY={r.get('excess_spy')} trades={r.get('n_trades')} mdd={r.get('max_drawdown')}",
            flush=True,
        )
        results.append(r)
        safe = name.replace("/", "_")
        tdf = r.get("trades")
        if isinstance(tdf, pd.DataFrame) and not tdf.empty:
            tdf.to_csv(out / f"trades_{safe}.csv", index=False)
        eq = r.get("equity")
        if isinstance(eq, pd.Series):
            eq.to_csv(out / f"equity_{safe}.csv", header=["equity"])
        if not r.get("error"):
            (out / f"dashboard_{safe}.html").write_text(
                build_one_html(r, i, first, last), encoding="utf-8"
            )

    ok = [r for r in results if not r.get("error")]
    # rank by excess vs SPY for index display
    ranked = sorted(ok, key=lambda x: float(x.get("excess_spy") or -9e9), reverse=True)

    # rebuild dashboards with rank by excess
    for rank, r in enumerate(ranked, 1):
        safe = r["strategy"].replace("/", "_")
        (out / f"dashboard_{safe}.html").write_text(
            build_one_html(r, rank, first, last), encoding="utf-8"
        )

    rows_md = []
    cards = []
    for rank, r in enumerate(ranked, 1):
        safe = r["strategy"].replace("/", "_")
        rows_md.append(
            f"| {rank} | [`{r['strategy']}`](dashboard_{safe}.html) | "
            f"{r.get('total_return',0):.1%} | {r.get('cagr',0):.2%} | "
            f"{(r.get('excess_spy') if r.get('excess_spy') is not None else float('nan')):.1%} | "
            f"{r.get('sharpe',0):.2f} | {r.get('sortino',0):.2f} | {r.get('max_drawdown',0):.1%} | "
            f"{r.get('n_trades')} | {r.get('win_rate',0):.1%} |"
        )
        cards.append(
            f"""<a class="card" href="dashboard_{safe}.html">
            <div class="rank">#{rank}</div>
            <div class="name"><code>{r['strategy']}</code></div>
            <div class="v">{r.get('cagr',0):.1%} CAGR</div>
            <div class="k">total {r.get('total_return',0):.0%} · vs SPY {(r.get('excess_spy') if r.get('excess_spy') is not None else float('nan')):.0%}</div>
            <div class="k">Sharpe {r.get('sharpe',0):.2f} · MDD {r.get('max_drawdown',0):.0%} · {r.get('n_trades')} trades · WR {r.get('win_rate',0):.0%}</div>
            </a>"""
        )

    index_md = [
        f"# Top-5 dashboards vs SPY — OOS {first}→{last}",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Window equity:** first OOS year {first} through last available bars in {last}.",
        f"**Universe:** `{Path(args.ticker_file).name}` limit={args.universe_limit}",
        "",
        "Long-only cash (no margin leverage). Commission **0.10%** + slippage **0.05%** per side.",
        "Walk-forward: retrain each calendar year; train ends at year-start.",
        "",
        "| Rank | Dashboard | Total | CAGR | vs SPY | Sharpe | Sortino | MDD | Trades | WR |",
        "|------|-----------|-------|------|--------|--------|---------|-----|--------|----|",
        *rows_md,
        "",
        "Each dashboard: equity vs SPY, yearly table, winners >33%, full trade blotter.",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "INDEX.md").write_text("\n".join(index_md), encoding="utf-8")

    spy_note = ""
    if ranked and ranked[0].get("spy_total") is not None:
        spy_note = f"<p style='color:#9fb0c3'>SPY total same window ≈ <b>{ranked[0]['spy_total']:.1%}</b></p>"

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
<p style="color:#9fb0c3;margin-top:24px">Research only. Updated {date.today().isoformat()}.</p>
</body></html>"""
    (out / "index.html").write_text(index_html, encoding="utf-8")

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "first_oos": first,
        "last_oos": last,
        "leverage_borrowed": False,
        "commission": COMMISSION,
        "slippage": SLIPPAGE,
        "strategies": [
            {
                "rank": i,
                "name": r["strategy"],
                "total_return": r.get("total_return"),
                "cagr": r.get("cagr"),
                "excess_spy": r.get("excess_spy"),
                "spy_total": r.get("spy_total"),
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
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n".join(index_md))
    print(f"\nIndex: {out / 'index.html'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
