"""Dashboard + index for turbo_strict__longhist_L50 (Kaggle research PASS #1).

Equity full path, drawdown, yearly table, all trades (screen+confirm).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.metrics import equity_metrics  # noqa: E402
from trad_research.risk_metrics import extended_risk_from_equity  # noqa: E402
from trad_research.walk_forward import load_benchmark_equity  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ytd_dash", ROOT / "scripts" / "run_ytd_trade_dashboard.py"
)
_ytd = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ytd)
_trades_html_table = _ytd._trades_html_table
_equity_js = _ytd._equity_js

ARM = "turbo_strict__longhist_L50"
DEFAULT_SRC = (
    ROOT
    / "reports/redesign/kaggle_overnight_t4x2/shard_1/arms"
    / ARM
)
DEFAULT_OUT = ROOT / "reports/redesign/dashboard_turbo_strict_longhist_L50"


def _eq(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df.iloc[:, 0].astype(float)
    s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
    return s[~s.index.duplicated(keep="last")].dropna().sort_index()


def _dd(eq: pd.Series) -> pd.Series:
    e = eq.astype(float)
    return e / e.cummax() - 1.0


def _bench_scaled(eq: pd.Series, data_root: Path, name: str = "SPY") -> Optional[pd.Series]:
    try:
        b = load_benchmark_equity(
            data_root, eq.index.min(), eq.index.max(), preferred=[name]
        )
        if b is None or b.empty:
            return None
        b = b.copy()
        b.index = pd.to_datetime(b.index, utc=True).normalize()
        e2 = eq.copy()
        e2.index = pd.to_datetime(e2.index, utc=True).normalize()
        e2 = e2[~e2.index.duplicated(keep="last")]
        b = b[~b.index.duplicated(keep="last")]
        j = pd.concat([e2.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
        if len(j) < 3:
            return None
        return j["b"] / float(j["b"].iloc[0]) * float(j["s"].iloc[0])
    except Exception:
        return None


def _year_table(eq: pd.Series, tdf: pd.DataFrame) -> str:
    e = eq.copy()
    e.index = pd.to_datetime(e.index, utc=True)
    rows = []
    for y in range(int(e.index.year.min()), int(e.index.year.max()) + 1):
        m = e[e.index.year == y]
        if len(m) < 5:
            continue
        ret = float(m.iloc[-1] / m.iloc[0] - 1.0)
        dd = float(_dd(m).min())
        n = 0
        if not tdf.empty and "exit_date" in tdf.columns:
            ed = pd.to_datetime(tdf["exit_date"], utc=True, errors="coerce")
            n = int((ed.dt.year == y).sum())
        cls = "win" if ret > 0 else "loss"
        rows.append(
            f"<tr class='{cls}'><td>{y}</td><td>{ret:.1%}</td><td>{dd:.1%}</td><td>{n}</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<h2>Por año</h2>"
        "<table><thead><tr><th>Año</th><th>Retorno</th><th>MDD año</th><th>Trades</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _ticker_summary(tdf: pd.DataFrame) -> str:
    if tdf.empty or "ticker" not in tdf.columns or "net_profit" not in tdf.columns:
        return ""
    g = (
        tdf.groupby("ticker", as_index=False)
        .agg(
            n=("net_profit", "count"),
            sum_pnl=("net_profit", "sum"),
            wr=("net_profit", lambda s: float((s > 0).mean())),
            avg_ret=("trade_return", "mean") if "trade_return" in tdf.columns else ("net_profit", "mean"),
        )
        .sort_values("sum_pnl", ascending=False)
    )
    best = g.head(15)
    worst = g.sort_values("sum_pnl").head(15)
    def tbl(df: pd.DataFrame, title: str) -> str:
        rows = []
        for _, r in df.iterrows():
            rows.append(
                f"<tr><td>{r['ticker']}</td><td>{int(r['n'])}</td>"
                f"<td>{r['wr']:.1%}</td><td>{r['sum_pnl']:,.0f}</td></tr>"
            )
        return (
            f"<h3>{title}</h3><table><thead><tr><th>Ticker</th><th>n</th><th>WR</th>"
            f"<th>Sum PnL</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
    return "<h2>Por ticker</h2>" + tbl(best, "Top 15 por sum PnL") + tbl(worst, "Peores 15")


def _exit_summary(tdf: pd.DataFrame) -> str:
    if tdf.empty or "exit_reason" not in tdf.columns:
        return ""
    g = tdf.groupby("exit_reason").agg(
        n=("net_profit", "count"),
        wr=("net_profit", lambda s: float((s > 0).mean())),
        sum_pnl=("net_profit", "sum"),
    ).sort_values("n", ascending=False)
    rows = []
    for reason, r in g.iterrows():
        rows.append(
            f"<tr><td><code>{reason}</code></td><td>{int(r['n'])}</td>"
            f"<td>{r['wr']:.1%}</td><td>{r['sum_pnl']:,.0f}</td></tr>"
        )
    return (
        "<h2>Por exit_reason</h2>"
        "<table><thead><tr><th>Exit</th><th>n</th><th>WR</th><th>Sum PnL</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def main() -> int:
    src = DEFAULT_SRC
    out = DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)

    eq = _eq(src / "equity_full.csv")
    tr_s = pd.read_csv(src / "trades_screen.csv")
    tr_c = pd.read_csv(src / "trades_confirm.csv")
    tr_s["segment"] = "screen_2010_2017"
    tr_c["segment"] = "confirm_2018_2025"
    tdf = pd.concat([tr_s, tr_c], ignore_index=True)
    # sort by entry
    if "entry_date" in tdf.columns:
        tdf["_ed"] = pd.to_datetime(tdf["entry_date"], utc=True, errors="coerce")
        tdf = tdf.sort_values("_ed").drop(columns=["_ed"]).reset_index(drop=True)

    start = float(eq.iloc[0])
    rep = equity_metrics(eq, start_equity=start, trades=tdf)
    risk = extended_risk_from_equity(
        eq.to_numpy(),
        trade_pnls=tdf["net_profit"].to_numpy() if "net_profit" in tdf.columns else None,
    )
    total = float(eq.iloc[-1] / start - 1.0)
    wins = int((tdf["net_profit"] > 0).sum())
    losses = int((tdf["net_profit"] <= 0).sum())
    wr = wins / max(wins + losses, 1)
    gp = float(tdf.loc[tdf["net_profit"] > 0, "net_profit"].sum()) if wins else 0.0
    gl = float(-tdf.loc[tdf["net_profit"] <= 0, "net_profit"].sum()) if losses else 0.0
    pf = gp / gl if gl > 0 else float("nan")
    calmar = float(rep.cagr / abs(rep.max_drawdown)) if rep.max_drawdown else float("nan")

    excess = None
    spy = _bench_scaled(eq, ROOT / "data", "SPY")
    try:
        b = load_benchmark_equity(
            ROOT / "data", eq.index.min(), eq.index.max(), preferred=["SPY"]
        )
        if b is not None and not b.empty:
            b = b.copy()
            b.index = pd.to_datetime(b.index, utc=True).normalize()
            e2 = eq.copy()
            e2.index = pd.to_datetime(e2.index, utc=True).normalize()
            j = pd.concat([e2.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
            if len(j) > 3:
                st = float(j["s"].iloc[-1] / j["s"].iloc[0] - 1.0)
                bt = float(j["b"].iloc[-1] / j["b"].iloc[0] - 1.0)
                excess = st - bt
                total = st  # aligned total
    except Exception:
        pass

    meta = {
        "arm": ARM,
        "source": str(src),
        "cagr": float(rep.cagr),
        "sharpe": float(rep.sharpe),
        "sortino": float(risk.sortino),
        "max_drawdown": float(rep.max_drawdown),
        "n_trades": int(len(tdf)),
        "win_rate": wr,
        "profit_factor": pf,
        "total_return": total,
        "excess_spy_total": excess,
        "calmar": calmar,
        "n_screen_trades": int(len(tr_s)),
        "n_confirm_trades": int(len(tr_c)),
        "generated": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Research only. Kaggle T4x2 research PASS #1. Not paper ADVANCE.",
    }
    (out / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    eq.to_csv(out / "equity_full.csv", header=["equity"])
    tdf.to_csv(out / "trades_all.csv", index=False)

    eq_js = _equity_js(eq, ARM)
    dd = _dd(eq)
    dd_x = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(eq.index, utc=True)]
    dd_y = [round(float(v) * 100, 3) for v in dd.values]
    dd_js = json.dumps({"name": "Drawdown %", "x": dd_x, "y": dd_y})
    spy_js = "null"
    if spy is not None:
        spy_js = _equity_js(spy, "SPY scaled")

    xs = f"{excess:.1%}" if excess is not None else "n/a"
    dash = f"""<!DOCTYPE html>
<html lang="es"><head>
<meta charset="utf-8"/>
<title>{ARM} — dashboard 2010–2025</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
 body {{ font-family: system-ui,Segoe UI,sans-serif; margin: 24px; background:#0f1419; color:#e7ecf1; }}
 h1,h2,h3 {{ color:#f3f6f9; }}
 .cards {{ display:flex; flex-wrap:wrap; gap:12px; margin:12px 0 20px; }}
 .card {{ background:#1a2332; border:1px solid #2a3a4f; border-radius:10px; padding:12px 16px; min-width:130px; }}
 .card .k {{ font-size:12px; color:#9fb0c3; }} .card .v {{ font-size:18px; font-weight:700; margin-top:4px; }}
 table {{ border-collapse:collapse; width:100%; font-size:11px; margin-bottom:20px; }}
 th,td {{ border:1px solid #2a3a4f; padding:4px 6px; text-align:right; }}
 th {{ background:#1a2332; position:sticky; top:0; }} td:first-child, th:first-child {{ text-align:left; }}
 tr.win td {{ background:rgba(34,197,94,0.08); }} tr.loss td {{ background:rgba(239,68,68,0.10); }}
 .meta {{ color:#9fb0c3; font-size:13px; }} a {{ color:#7dd3fc; }}
 .ok {{ background:#14301f; border-left:4px solid #22c55e; padding:12px 16px; margin:16px 0; }}
 .warn {{ background:#3a2a10; border-left:4px solid #f59e0b; padding:12px 16px; margin:16px 0; }}
</style></head><body>
<p class="meta"><a href="index.html">← Índice</a></p>
<h1><code>{ARM}</code></h1>
<p class="meta">Kaggle GPU T4×2 research PASS #1 · screen 2010–17 + confirm 2018–25 stitch · longhist L50 · turbo_strict</p>
<p class="meta">Equity: {str(eq.index.min())[:10]} → {str(eq.index.max())[:10]} · n_bars={len(eq)} · trades screen={len(tr_s)} + confirm={len(tr_c)} = {len(tdf)}</p>
<div class="ok">
<strong>Research PASS</strong> (confirm∩full gates Kaggle) ·
CAGR <strong>{rep.cagr:.2%}</strong> · Max DD <strong>{rep.max_drawdown:.2%}</strong> ·
Sharpe {rep.sharpe:.2f} · Sortino {risk.sortino:.2f} · Trades {len(tdf)} · WR {wr:.1%}
</div>
<div class="warn">Research only. No es paper ADVANCE automático. Paper freeze sigue turbo_highvol_minalloc. Not financial advice.</div>
<div class="cards">
  <div class="card"><div class="k">Total (aligned)</div><div class="v">{total:.1%}</div></div>
  <div class="card"><div class="k">CAGR</div><div class="v">{rep.cagr:.2%}</div></div>
  <div class="card"><div class="k">Sharpe</div><div class="v">{rep.sharpe:.2f}</div></div>
  <div class="card"><div class="k">Sortino</div><div class="v">{risk.sortino:.2f}</div></div>
  <div class="card"><div class="k">Max DD</div><div class="v">{rep.max_drawdown:.2%}</div></div>
  <div class="card"><div class="k">Calmar</div><div class="v">{calmar:.2f}</div></div>
  <div class="card"><div class="k">Trades</div><div class="v">{len(tdf)} (W{wins}/L{losses})</div></div>
  <div class="card"><div class="k">Win rate</div><div class="v">{wr:.1%}</div></div>
  <div class="card"><div class="k">PF</div><div class="v">{pf:.2f}</div></div>
  <div class="card"><div class="k">vs SPY total</div><div class="v">{xs}</div></div>
  <div class="card"><div class="k">Wins $</div><div class="v">{gp:,.0f}</div></div>
  <div class="card"><div class="k">Losses $</div><div class="v">{gl:,.0f}</div></div>
</div>
<div id="eqchart" style="height:400px;margin:16px 0;"></div>
<div id="ddchart" style="height:280px;margin:16px 0;"></div>
{_year_table(eq, tdf)}
{_exit_summary(tdf)}
{_ticker_summary(tdf)}
<h2>Todas las operaciones ({len(tdf)})</h2>
<p class="meta">Incluye screen 2010–17 y confirm 2018–25 (column <code>segment</code>).</p>
{_trades_html_table(tdf)}
<p class="meta">Research only. Not financial advice. Source: Kaggle trad-overnight-t4x2.</p>
<script>
const E = {eq_js};
const S = {spy_js};
const D = {dd_js};
const eqTraces = [{{x:E.x,y:E.y,name:E.name,type:'scatter',mode:'lines',line:{{color:'#38bdf8',width:2}}}}];
if (S) eqTraces.push({{x:S.x,y:S.y,name:S.name,type:'scatter',mode:'lines',line:{{color:'#94a3b8',width:1.5,dash:'dot'}}}});
Plotly.newPlot('eqchart', eqTraces, {{
  paper_bgcolor:'#0f1419', plot_bgcolor:'#0f1419', font:{{color:'#e7ecf1'}},
  title:'Equity 2010–2025 (log) vs SPY scaled', legend:{{orientation:'h'}}, margin:{{t:40,r:20,b:40,l:60}},
  xaxis:{{gridcolor:'#1f2a37'}}, yaxis:{{gridcolor:'#1f2a37', tickprefix:'$', type:'log'}}
}}, {{responsive:true}});
Plotly.newPlot('ddchart', [{{
  x:D.x, y:D.y, name:'Drawdown %', type:'scatter', mode:'lines',
  fill:'tozeroy', line:{{color:'#ef4444', width:1.5}}, fillcolor:'rgba(239,68,68,0.25)'
}}], {{
  paper_bgcolor:'#0f1419', plot_bgcolor:'#0f1419', font:{{color:'#e7ecf1'}},
  title:'Drawdown path', legend:{{orientation:'h'}}, margin:{{t:40,r:20,b:40,l:50}},
  xaxis:{{gridcolor:'#1f2a37'}}, yaxis:{{gridcolor:'#1f2a37', ticksuffix:'%', rangemode:'tozero'}}
}}, {{responsive:true}});
</script>
</body></html>
"""
    (out / f"dashboard_{ARM}.html").write_text(dash, encoding="utf-8")

    index = f"""<!DOCTYPE html>
<html lang="es"><head>
<meta charset="utf-8"/>
<title>Index — {ARM}</title>
<style>
 body {{ font-family: system-ui,Segoe UI,sans-serif; margin: 28px; background:#0f1419; color:#e7ecf1; max-width:1000px; }}
 h1 {{ color:#f3f6f9; }} table {{ border-collapse:collapse; width:100%; font-size:13px; }}
 th,td {{ border:1px solid #2a3a4f; padding:8px 10px; text-align:right; }}
 th {{ background:#1a2332; }} td:first-child, th:first-child {{ text-align:left; }}
 a {{ color:#7dd3fc; }} .meta {{ color:#9fb0c3; }}
 .ok {{ background:#14301f; border-left:4px solid #22c55e; padding:12px 16px; margin:16px 0; }}
 .warn {{ background:#3a2a10; border-left:4px solid #f59e0b; padding:12px 16px; margin:16px 0; }}
</style></head><body>
<h1>Research PASS #1 — <code>{ARM}</code></h1>
<p class="meta">Kaggle GPU T4×2 · overnight definitive · longhist 2010-pass · limit 50 · turbo_strict</p>
<div class="ok">
<strong>CAGR {rep.cagr:.2%}</strong> · Max DD <strong>{rep.max_drawdown:.2%}</strong> ·
Sharpe {rep.sharpe:.2f} · Sortino {risk.sortino:.2f} ·
Trades <strong>{len(tdf)}</strong> · WR {wr:.1%} · PF {pf:.2f} · vs SPY {xs}
</div>
<div class="warn">Research only. Not paper ADVANCE. Freeze paper = turbo_highvol_minalloc.</div>
<table>
<thead><tr>
<th>Dashboard</th><th>CAGR</th><th>Total</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Trades</th><th>WR</th><th>vs SPY</th>
</tr></thead>
<tbody>
<tr>
<td><a href="dashboard_{ARM}.html"><code>{ARM}</code></a></td>
<td>{rep.cagr:.2%}</td>
<td>{total:.1%}</td>
<td>{rep.sharpe:.2f}</td>
<td>{risk.sortino:.2f}</td>
<td>{rep.max_drawdown:.2%}</td>
<td>{len(tdf)}</td>
<td>{wr:.1%}</td>
<td>{xs}</td>
</tr>
</tbody></table>
<ul>
<li><a href="dashboard_{ARM}.html">Dashboard completo (equity + DD + todas las operaciones)</a></li>
<li><a href="trades_all.csv">CSV trades (screen+confirm)</a></li>
<li><a href="equity_full.csv">CSV equity full</a></li>
<li><a href="metrics.json">metrics.json</a></li>
</ul>
<p class="meta">Generated {meta['generated']}. Source: reports/redesign/kaggle_overnight_t4x2/…</p>
</body></html>
"""
    (out / "index.html").write_text(index, encoding="utf-8")
    (out / "INDEX.md").write_text(
        "\n".join(
            [
                f"# {ARM} — dashboard",
                "",
                f"- Open: [`index.html`](index.html) or [`dashboard_{ARM}.html`](dashboard_{ARM}.html)",
                f"- CAGR **{rep.cagr:.2%}** · MDD **{rep.max_drawdown:.2%}** · trades **{len(tdf)}**",
                "- Research PASS Kaggle T4×2. Not paper ADVANCE.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out / 'index.html'}")
    print(f"  CAGR={rep.cagr:.2%} MDD={rep.max_drawdown:.2%} n={len(tdf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
