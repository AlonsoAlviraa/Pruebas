"""Build 5 trade dashboards for strategies that most beat SPY in 2026 YTD.

Default top-5 from bake-off excess vs SPY:
  turbo_highvol, minalloc, sector_rot, softreg, turbo_strict
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_ytd_trade_dashboard", ROOT / "scripts" / "run_ytd_trade_dashboard.py"
)
_dash = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dash)
COMMISSION = _dash.COMMISSION
SLIPPAGE = _dash.SLIPPAGE
_run = _dash._run
_trades_html_table = _dash._trades_html_table
_equity_js = _dash._equity_js

from trad_research.walk_forward import load_benchmark_equity

# Ranked by 2026 YTD excess vs SPY (bake-off)
TOP5 = [
    "turbo_highvol",
    "turbo_highvol_minalloc",
    "turbo_highvol_minalloc_sector_rot",
    "turbo_highvol_minalloc_softreg",
    "turbo_strict",
]


def build_one_html(r: Dict[str, Any], rank: int) -> str:
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
        for _, row in b.head(8).iterrows():
            big.append(
                f"<li><b>{row.get('ticker')}</b> "
                f"{str(row.get('entry_date'))[:10]} → {str(row.get('exit_date'))[:10]} · "
                f"PnL ${float(row.get('net_profit') or 0):,.0f} · "
                f"{float(row.get('trade_return') or 0):.1%} · {row.get('exit_reason')}</li>"
            )
    big_html = "<ul>" + ("".join(big) if big else "<li><em>Ninguna op &gt;33%</em></li>") + "</ul>"

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
<title>#{rank} {r['strategy']} — 2026 YTD vs SPY</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
 body {{ font-family: system-ui,Segoe UI,sans-serif; margin: 24px; background:#0f1419; color:#e7ecf1; }}
 h1,h2 {{ color:#f3f6f9; }}
 .cards {{ display:flex; flex-wrap:wrap; gap:12px; margin:12px 0 20px; }}
 .card {{ background:#1a2332; border:1px solid #2a3a4f; border-radius:10px; padding:12px 16px; min-width:130px; }}
 .card .k {{ font-size:12px; color:#9fb0c3; }} .card .v {{ font-size:20px; font-weight:700; margin-top:4px; }}
 table {{ border-collapse:collapse; width:100%; font-size:12px; }}
 th,td {{ border:1px solid #2a3a4f; padding:5px 7px; text-align:right; }}
 th {{ background:#1a2332; }} td:first-child, th:first-child {{ text-align:left; }}
 tr.win td {{ background:rgba(34,197,94,0.08); }} tr.loss td {{ background:rgba(239,68,68,0.10); }}
 .meta {{ color:#9fb0c3; font-size:13px; }}
 .ok {{ background:#14301f; border-left:4px solid #22c55e; padding:12px 16px; margin:16px 0; }}
 a {{ color:#7dd3fc; }}
</style></head><body>
<p class="meta"><a href="index.html">← Índice top-5</a></p>
<h1>#{rank} <code>{r['strategy']}</code> — 2026 YTD</h1>
<p class="meta">Long-only cash (sin margin) · comisión {COMMISSION:.2%} + slip {SLIPPAGE:.2%} por lado · round-trip ≈ {2*COMMISSION+2*SLIPPAGE:.2%}</p>
<div class="ok">
<strong>vs SPY:</strong> {(r.get('excess_spy') if r.get('excess_spy') is not None else float('nan')):.2%} &nbsp;|&nbsp;
<strong>Retorno:</strong> {r['total_return']:.2%} &nbsp;|&nbsp;
<strong>SPY ventana:</strong> {(r.get('spy_total') if r.get('spy_total') is not None else float('nan')):.2%}
</div>
<div class="cards">
  <div class="card"><div class="k">Retorno</div><div class="v">{r['total_return']:.2%}</div></div>
  <div class="card"><div class="k">Sharpe</div><div class="v">{r['sharpe']:.2f}</div></div>
  <div class="card"><div class="k">Max DD</div><div class="v">{r['max_drawdown']:.2%}</div></div>
  <div class="card"><div class="k">Trades</div><div class="v">{r['n_trades']} (W{wins}/L{losses})</div></div>
  <div class="card"><div class="k">Win rate</div><div class="v">{wr:.1%}</div></div>
  <div class="card"><div class="k">Profit factor</div><div class="v">{r['profit_factor']:.2f}</div></div>
  <div class="card"><div class="k">vs SPY</div><div class="v">{(r.get('excess_spy') if r.get('excess_spy') is not None else float('nan')):.2%}</div></div>
  <div class="card"><div class="k">vs QQQ</div><div class="v">{(r.get('excess_qqq') if r.get('excess_qqq') is not None else float('nan')):.2%}</div></div>
  <div class="card"><div class="k">Comisiones ≈</div><div class="v">${r.get('approx_total_commissions_usd',0):,.0f}</div></div>
</div>
<p class="meta">Sizing bake-off: vol_target={ov.get('volatility_target_pct')} · max_pos={ov.get('max_position_pct')} · max_positions={ov.get('max_positions')}</p>
<p class="meta">Wins ${gp:,.0f} · Losses ${gl:,.0f} · Net trades ${gp-gl:,.0f}</p>
<div id="eqchart" style="height:380px;margin:16px 0;"></div>
<h2>Ganadoras &gt; 33%</h2>
{big_html}
<h2>Todas las operaciones</h2>
{_trades_html_table(tdf)}
<p class="meta">Research only. No es consejo financiero.</p>
<script>
const E = {eq_js};
const S = {spy_js};
const traces = [{{x:E.x,y:E.y,name:E.name,type:'scatter',mode:'lines',line:{{color:'#f97316',width:2}}}}];
if (S) traces.push({{x:S.x,y:S.y,name:S.name,type:'scatter',mode:'lines',line:{{color:'#94a3b8',width:1.5,dash:'dot'}}}});
Plotly.newPlot('eqchart', traces, {{
  paper_bgcolor:'#0f1419', plot_bgcolor:'#0f1419', font:{{color:'#e7ecf1'}},
  title:'Equity vs SPY', legend:{{orientation:'h'}}, margin:{{t:40,r:20,b:40,l:60}},
  xaxis:{{gridcolor:'#1f2a37'}}, yaxis:{{gridcolor:'#1f2a37', tickprefix:'$'}}
}}, {{responsive:true}});
</script>
</body></html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--strategies", type=str, default=",".join(TOP5))
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--ticker-file", type=Path, default=ROOT / "universe_highvol80.txt")
    ap.add_argument("--universe-limit", type=int, default=80)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "ytd_2026_top5_dashboards",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    # bake-off sizing (same as YTD #1)
    overrides = {
        "commission": COMMISSION,
        "slippage": SLIPPAGE,
        # leave strategy defaults for vol_target/max_pos unless missing
    }

    results: List[Dict[str, Any]] = []
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name} …", flush=True)
        # merge strategy defaults: use empty extra so strategy backtest_overrides apply;
        # still force costs
        r = _run(
            name=name,
            label=name,
            year=args.year,
            data_root=Path(args.data_root),
            ticker_file=Path(args.ticker_file),
            universe_limit=int(args.universe_limit),
            overrides=overrides,
        )
        # _run merges overrides into strategy overrides — good
        # Fix excess_qqq key for template
        if "excess_vs_qqq" in r and "excess_qqq" not in r:
            r["excess_qqq"] = r.get("excess_vs_qqq")
        if r.get("excess_spy") is None and r.get("excess_vs_spy") is not None:
            r["excess_spy"] = r["excess_vs_spy"]
        print(
            f"  total={r.get('total_return')} vsSPY={r.get('excess_spy')} trades={r.get('n_trades')}",
            flush=True,
        )
        results.append(r)

        # save artifacts
        safe = name.replace("/", "_")
        tdf = r.get("trades")
        if isinstance(tdf, pd.DataFrame) and not tdf.empty:
            tdf.to_csv(out / f"trades_{safe}.csv", index=False)
        eq = r.get("equity")
        if isinstance(eq, pd.Series):
            eq.to_csv(out / f"equity_{safe}.csv", header=["equity"])
        (out / f"dashboard_{safe}.html").write_text(
            build_one_html(r, i), encoding="utf-8"
        )

    # rank by excess vs SPY
    ok = [r for r in results if not r.get("error")]
    ok.sort(key=lambda x: float(x.get("excess_spy") or -9e9), reverse=True)

    # rebuild rank numbers in filenames already 1..n by input order (TOP5 order)
    # index page
    rows = []
    for i, r in enumerate(ok, 1):
        safe = r["strategy"].replace("/", "_")
        rows.append(
            f"| {i} | [`{r['strategy']}`](dashboard_{safe}.html) | "
            f"{r.get('total_return',0):.2%} | {r.get('excess_spy') if r.get('excess_spy') is not None else float('nan'):.2%} | "
            f"{r.get('sharpe',0):.2f} | {r.get('max_drawdown',0):.2%} | {r.get('n_trades')} | "
            f"{r.get('win_rate',0):.1%} |"
        )
    index_md = [
        f"# Top-5 dashboards vs SPY — {args.year} YTD",
        "",
        "Estrategias que **más superaron al SPY** en el bake-off 2026 (universo highvol80).",
        "",
        "**Sin apalancamiento prestado** (long-only cash). Comisión 0.10% + slip 0.05% por lado.",
        "",
        "| Rank | Dashboard | Total | vs SPY | Sharpe | MDD | Trades | WR |",
        "|------|-----------|-------|--------|--------|-----|--------|----|",
        *rows,
        "",
        "Cada dashboard incluye equity vs SPY, resumen, ganadoras >33% y **todas las operaciones**.",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "INDEX.md").write_text("\n".join(index_md), encoding="utf-8")

    # HTML index
    cards = []
    for i, r in enumerate(ok, 1):
        safe = r["strategy"].replace("/", "_")
        cards.append(
            f"""<a class="card" href="dashboard_{safe}.html">
            <div class="rank">#{i}</div>
            <div class="name"><code>{r['strategy']}</code></div>
            <div class="v">{r.get('total_return',0):.1%} total</div>
            <div class="k">vs SPY {(r.get('excess_spy') if r.get('excess_spy') is not None else float('nan')):.1%}</div>
            <div class="k">{r.get('n_trades')} trades · WR {r.get('win_rate',0):.0%} · MDD {r.get('max_drawdown',0):.1%}</div>
            </a>"""
        )
    index_html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<title>Top-5 vs SPY — 2026 YTD</title>
<style>
 body {{ font-family: system-ui,sans-serif; margin:24px; background:#0f1419; color:#e7ecf1; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:16px; }}
 a.card {{ display:block; background:#1a2332; border:1px solid #2a3a4f; border-radius:12px;
   padding:16px; text-decoration:none; color:inherit; }}
 a.card:hover {{ border-color:#38bdf8; }}
 .rank {{ color:#f97316; font-weight:800; font-size:14px; }}
 .name {{ margin:8px 0; font-size:15px; }}
 .v {{ font-size:22px; font-weight:700; }}
 .k {{ color:#9fb0c3; font-size:13px; margin-top:4px; }}
</style></head><body>
<h1>Top-5 estrategias vs SPY — 2026 YTD</h1>
<p style="color:#9fb0c3">Long-only cash · comisión 0.10% + slip 0.05%/lado · highvol80</p>
<div class="grid">{''.join(cards)}</div>
<p style="color:#9fb0c3;margin-top:24px">Research only.</p>
</body></html>"""
    (out / "index.html").write_text(index_html, encoding="utf-8")

    summary = {
        "year": args.year,
        "strategies": [
            {
                "rank": i,
                "name": r["strategy"],
                "total_return": r.get("total_return"),
                "excess_spy": r.get("excess_spy"),
                "sharpe": r.get("sharpe"),
                "max_drawdown": r.get("max_drawdown"),
                "n_trades": r.get("n_trades"),
                "win_rate": r.get("win_rate"),
                "dashboard": f"dashboard_{r['strategy'].replace('/', '_')}.html",
            }
            for i, r in enumerate(ok, 1)
        ],
        "commission": COMMISSION,
        "slippage": SLIPPAGE,
        "leverage_borrowed": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n".join(index_md))
    print(f"\nIndex: {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
