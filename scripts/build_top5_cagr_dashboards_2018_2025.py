"""Build index + per-strategy dashboards: top-5 CAGR full OOS 2018–2025.

Uses Loop E artifacts when present; re-runs missing configs via mega-loop helpers.
Shows equity, drawdown, and full trade blotter.
"""
from __future__ import annotations

import argparse
import importlib.util
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

from trad_research.metrics import equity_metrics  # noqa: E402
from trad_research.risk_metrics import extended_risk_from_equity  # noqa: E402
from trad_research.walk_forward import load_benchmark_equity  # noqa: E402

# reuse trade table helpers
_spec_y = importlib.util.spec_from_file_location(
    "ytd_dash", ROOT / "scripts" / "run_ytd_trade_dashboard.py"
)
_ytd = importlib.util.module_from_spec(_spec_y)
assert _spec_y.loader is not None
sys.modules["ytd_dash"] = _ytd
_spec_y.loader.exec_module(_ytd)
_trades_html_table = _ytd._trades_html_table
_equity_js = _ytd._equity_js

_spec_m = importlib.util.spec_from_file_location(
    "vol_fund_mega", ROOT / "scripts" / "run_vol_fund_mega_loop.py"
)
_mega = importlib.util.module_from_spec(_spec_m)
assert _spec_m.loader is not None
sys.modules["vol_fund_mega"] = _mega
_spec_m.loader.exec_module(_mega)

# Prefer known full-OOS high-CAGR sleeves (Loop D/E)
CANDIDATES = [
    "turbo_highvol_minalloc__volonly_k100_baseline",
    "turbo_highvol_minalloc__volonly_k100_vt60_only",
    "turbo_highvol_minalloc__volonly_k100_dd35_vt80_yr",
    "turbo_highvol_minalloc__volonly_k100_dd25_vt70_yr",
    "turbo_highvol_minalloc__volonly_k60_vt60_only",
    "turbo_highvol_minalloc__volonly_k80_dd35_vt80_yr",
    "turbo_highvol_minalloc__volonly_k100_dd18_vt70_pos75",
    "turbo_highvol_minalloc__volonly_k80_baseline",
]

LOOP_E = ROOT / "reports" / "redesign" / "vol_fund_loop_e" / "configs"


def _parse_cfg(config_id: str) -> Any:
    rest = config_id
    if rest.startswith("turbo_highvol_minalloc__volonly_"):
        rest = rest.replace("turbo_highvol_minalloc__volonly_", "")
        parts = rest.split("_", 1)
        top = int(parts[0].replace("k", ""))
        lever = parts[1] if len(parts) > 1 else "baseline"
        return _mega.GridConfig(
            config_id=config_id,
            strategy="turbo_highvol_minalloc",
            growth_hard=False,
            growth_top_k=top,
            lever_id=lever,
            vol_only_top=top,
            label="dashboard",
        )
    raise ValueError(config_id)


def _load_existing(config_id: str) -> Optional[Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]]:
    d = LOOP_E / config_id
    eqp = d / "equity.csv"
    if not eqp.is_file():
        return None
    eq = pd.read_csv(eqp, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    eq.index = pd.to_datetime(eq.index, utc=True, errors="coerce")
    eq = eq[~eq.index.duplicated(keep="last")].dropna().sort_index()
    trp = d / "trades.csv"
    tdf = pd.read_csv(trp) if trp.is_file() else pd.DataFrame()
    meta: Dict[str, Any] = {}
    mp = d / "metrics.json"
    if mp.is_file():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    return eq, tdf, meta


def _run_full(
    config_id: str,
    *,
    panel: Path,
    data_root: Path,
    l0_cache: Path,
    years: List[int],
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    cfg = _parse_cfg(config_id)
    static_pool = [
        ln.strip().upper()
        for ln in panel.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    r = _mega.run_config_years(
        cfg,
        years=years,
        data_root=data_root,
        panel_file=panel,
        l0_cache=l0_cache,
        static_pool=static_pool,
        min_train_rows=2500,
        use_dynamic_vol=False,
    )
    eq = r.get("equity")
    if eq is None or (hasattr(eq, "empty") and eq.empty):
        raise RuntimeError(f"empty equity {config_id}")
    tdf = r.get("trades") if isinstance(r.get("trades"), pd.DataFrame) else pd.DataFrame()
    meta = {k: v for k, v in r.items() if k not in ("equity", "trades")}
    return eq.astype(float), tdf, meta


def _dd_series(eq: pd.Series) -> pd.Series:
    e = eq.astype(float)
    peak = e.cummax()
    return e / peak - 1.0


def _bench_scaled(eq: pd.Series, data_root: Path, name: str = "SPY") -> Optional[pd.Series]:
    try:
        b = load_benchmark_equity(data_root, eq.index.min(), eq.index.max(), preferred=[name])
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
    for y in range(e.index.year.min(), e.index.year.max() + 1):
        m = e[e.index.year == y]
        if len(m) < 5:
            continue
        ret = float(m.iloc[-1] / m.iloc[0] - 1.0)
        dd = float(_dd_series(m).min())
        n = 0
        if not tdf.empty and "exit_date" in tdf.columns:
            ed = pd.to_datetime(tdf["exit_date"], utc=True, errors="coerce")
            n = int((ed.dt.year == y).sum())
        rows.append(
            f"<tr><td>{y}</td><td>{ret:.1%}</td><td>{dd:.1%}</td><td>{n}</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<h2>Por año (equity path + MDD del año)</h2>"
        "<table><thead><tr><th>Año</th><th>Retorno</th><th>MDD año</th><th>Trades cerrados</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_dashboard(r: Dict[str, Any], rank: int, first: int, last: int) -> str:
    name = r["config_id"]
    eq: pd.Series = r["equity"]
    tdf: pd.DataFrame = r["trades"] if isinstance(r["trades"], pd.DataFrame) else pd.DataFrame()
    dd = _dd_series(eq)
    wins = int((tdf["net_profit"] > 0).sum()) if not tdf.empty and "net_profit" in tdf.columns else 0
    losses = int((tdf["net_profit"] <= 0).sum()) if not tdf.empty and "net_profit" in tdf.columns else 0
    wr = wins / max(wins + losses, 1)
    gp = float(tdf.loc[tdf["net_profit"] > 0, "net_profit"].sum()) if wins else 0.0
    gl = float(-tdf.loc[tdf["net_profit"] <= 0, "net_profit"].sum()) if losses else 0.0

    eq_js = _equity_js(eq, name)
    dd_js = _equity_js(dd * 100, "Drawdown %")  # percent points for axis
    # fix: store raw dd not *100 in y for clarity - use percent as fraction *100
    dd_x = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(eq.index, utc=True)]
    dd_y = [round(float(v) * 100, 3) for v in dd.values]
    dd_js = json.dumps({"name": "Drawdown %", "x": dd_x, "y": dd_y})

    spy_js = "null"
    spy = _bench_scaled(eq, ROOT / "data", "SPY")
    if spy is not None:
        spy_js = _equity_js(spy, "SPY scaled")

    return f"""<!DOCTYPE html>
<html lang="es"><head>
<meta charset="utf-8"/>
<title>#{rank} {name} — {first}–{last}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
 body {{ font-family: system-ui,Segoe UI,sans-serif; margin: 24px; background:#0f1419; color:#e7ecf1; }}
 h1,h2 {{ color:#f3f6f9; }}
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
<p class="meta"><a href="index.html">← Índice top-5 CAGR 2018–2025</a></p>
<h1>#{rank} <code>{name}</code></h1>
<p class="meta">OOS {first}→{last} · highvol200 vol-only · minalloc family · comisión 0.10% + slip 0.05%</p>
<p class="meta">Ventana equity: {str(eq.index.min())[:10]} → {str(eq.index.max())[:10]} · n_bars={len(eq)}</p>
<div class="ok">
<strong>CAGR {r['cagr']:.2%}</strong> · Total {r['total_return']:.1%} ·
Sharpe {r['sharpe']:.2f} · Sortino {r['sortino']:.2f} ·
<strong>Max DD {r['max_drawdown']:.2%}</strong> · Trades {r['n_trades']} · WR {wr:.1%}
</div>
<div class="warn">Research only. Promo Stage1 MDD gate was KILL for full-path (&lt; −50%). Not paper ADVANCE.</div>
<div class="cards">
  <div class="card"><div class="k">Total</div><div class="v">{r['total_return']:.1%}</div></div>
  <div class="card"><div class="k">CAGR</div><div class="v">{r['cagr']:.2%}</div></div>
  <div class="card"><div class="k">Sharpe</div><div class="v">{r['sharpe']:.2f}</div></div>
  <div class="card"><div class="k">Sortino</div><div class="v">{r['sortino']:.2f}</div></div>
  <div class="card"><div class="k">Max DD</div><div class="v">{r['max_drawdown']:.2%}</div></div>
  <div class="card"><div class="k">Calmar</div><div class="v">{r.get('calmar', float('nan')):.2f}</div></div>
  <div class="card"><div class="k">Trades</div><div class="v">{r['n_trades']} (W{wins}/L{losses})</div></div>
  <div class="card"><div class="k">Win rate</div><div class="v">{wr:.1%}</div></div>
  <div class="card"><div class="k">PF</div><div class="v">{r['profit_factor']:.2f}</div></div>
  <div class="card"><div class="k">vs SPY (total)</div><div class="v">{(r.get('excess_spy') if r.get('excess_spy') is not None else float('nan')):.1%}</div></div>
  <div class="card"><div class="k">Wins $</div><div class="v">{gp:,.0f}</div></div>
  <div class="card"><div class="k">Losses $</div><div class="v">{gl:,.0f}</div></div>
</div>
<div id="eqchart" style="height:380px;margin:16px 0;"></div>
<div id="ddchart" style="height:280px;margin:16px 0;"></div>
{_year_table(eq, tdf)}
<h2>Todas las operaciones ({r['n_trades']})</h2>
{_trades_html_table(tdf)}
<p class="meta">Research only. No es consejo financiero. WF anual, train ≤ year-start. Panel universe_highvol200.</p>
<script>
const E = {eq_js};
const S = {spy_js};
const D = {dd_js};
const eqTraces = [{{x:E.x,y:E.y,name:E.name,type:'scatter',mode:'lines',line:{{color:'#f97316',width:2}}}}];
if (S) eqTraces.push({{x:S.x,y:S.y,name:S.name,type:'scatter',mode:'lines',line:{{color:'#94a3b8',width:1.5,dash:'dot'}}}});
Plotly.newPlot('eqchart', eqTraces, {{
  paper_bgcolor:'#0f1419', plot_bgcolor:'#0f1419', font:{{color:'#e7ecf1'}},
  title:'Equity {first}–{last} (log) vs SPY', legend:{{orientation:'h'}}, margin:{{t:40,r:20,b:40,l:60}},
  xaxis:{{gridcolor:'#1f2a37'}}, yaxis:{{gridcolor:'#1f2a37', tickprefix:'$', type:'log'}}
}}, {{responsive:true}});
Plotly.newPlot('ddchart', [{{
  x:D.x, y:D.y, name:'Drawdown %', type:'scatter', mode:'lines',
  fill:'tozeroy', line:{{color:'#ef4444', width:1.5}}, fillcolor:'rgba(239,68,68,0.25)'
}}], {{
  paper_bgcolor:'#0f1419', plot_bgcolor:'#0f1419', font:{{color:'#e7ecf1'}},
  title:'Drawdown path (peak-to-trough %)', legend:{{orientation:'h'}}, margin:{{t:40,r:20,b:40,l:50}},
  xaxis:{{gridcolor:'#1f2a37'}}, yaxis:{{gridcolor:'#1f2a37', ticksuffix:'%', rangemode:'tozero'}}
}}, {{responsive:true}});
</script>
</body></html>
"""


def build_index(ranked: List[Dict[str, Any]], first: int, last: int) -> str:
    rows = []
    for i, r in enumerate(ranked, 1):
        safe = r["config_id"].replace("/", "_")
        rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td><a href='dashboard_{safe}.html'><code>{r['config_id']}</code></a></td>"
            f"<td>{r['cagr']:.2%}</td>"
            f"<td>{r['total_return']:.1%}</td>"
            f"<td>{r['sharpe']:.2f}</td>"
            f"<td>{r['sortino']:.2f}</td>"
            f"<td>{r['max_drawdown']:.2%}</td>"
            f"<td>{r['n_trades']}</td>"
            f"<td>{r['win_rate']:.1%}</td>"
            f"<td>{(r.get('excess_spy') if r.get('excess_spy') is not None else float('nan')):.1%}</td>"
            f"</tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="es"><head>
<meta charset="utf-8"/>
<title>Top-5 CAGR OOS {first}–{last} — highvol200 minalloc</title>
<style>
 body {{ font-family: system-ui,Segoe UI,sans-serif; margin: 28px; background:#0f1419; color:#e7ecf1; max-width:1100px; }}
 h1 {{ color:#f3f6f9; }} table {{ border-collapse:collapse; width:100%; font-size:13px; }}
 th,td {{ border:1px solid #2a3a4f; padding:8px 10px; text-align:right; }}
 th {{ background:#1a2332; }} td:nth-child(2), th:nth-child(2) {{ text-align:left; }}
 a {{ color:#7dd3fc; }} .meta {{ color:#9fb0c3; }} .ok {{ background:#14301f; border-left:4px solid #22c55e; padding:12px 16px; }}
</style></head><body>
<h1>Top-5 por CAGR · OOS {first}–{last}</h1>
<p class="meta">Panel <code>universe_highvol200</code> · solo <code>turbo_highvol_minalloc</code> vol-only + risk levers · Loop D/E</p>
<div class="ok">
<strong>Research only.</strong> Full-path promo: todos KILL por MDD &lt; −50%. No es paper ADVANCE.
Click cada fila para equity, <strong>drawdown</strong> y <strong>todos los trades</strong>.
</div>
<table>
<thead><tr>
<th>#</th><th>Estrategia</th><th>CAGR</th><th>Total</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Trades</th><th>WR</th><th>vs SPY</th>
</tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
<p class="meta">Generado {datetime.now(timezone.utc).isoformat()} · Not financial advice.</p>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=ROOT / "universe_highvol200.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--first", type=int, default=2018)
    ap.add_argument("--last", type=int, default=2025)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "top5_cagr_2018_2025_dashboards",
    )
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    l0_cache = out / "l0_cache"
    years = list(range(int(args.first), int(args.last) + 1))

    loaded: List[Dict[str, Any]] = []
    for cid in CANDIDATES:
        print(f"Loading/running {cid} …", flush=True)
        try:
            ex = _load_existing(cid)
            if ex is not None:
                eq, tdf, meta = ex
                print(f"  cache loop_e n={len(eq)} trades={len(tdf)}", flush=True)
            else:
                eq, tdf, meta = _run_full(
                    cid,
                    panel=Path(args.panel),
                    data_root=Path(args.data_root),
                    l0_cache=l0_cache,
                    years=years,
                )
                # persist under out for reuse
                cdir = out / "configs" / cid
                cdir.mkdir(parents=True, exist_ok=True)
                eq.to_csv(cdir / "equity.csv", header=["equity"])
                if not tdf.empty:
                    tdf.to_csv(cdir / "trades.csv", index=False)
                (cdir / "metrics.json").write_text(
                    json.dumps(meta, indent=2, default=str), encoding="utf-8"
                )
                print(f"  ran full cagr={meta.get('cagr')}", flush=True)

            start = float(eq.iloc[0])
            rep = equity_metrics(
                eq, start_equity=start, trades=tdf if not tdf.empty else None
            )
            risk = extended_risk_from_equity(
                eq.to_numpy(),
                trade_pnls=tdf["net_profit"].to_numpy()
                if not tdf.empty and "net_profit" in tdf.columns
                else None,
            )
            total = float(eq.iloc[-1] / start - 1.0)
            spy_tot = None
            try:
                spy = _bench_scaled(eq, Path(args.data_root), "SPY")
                if spy is not None:
                    # total SPY on join is not same as excess total — compute simple
                    b = load_benchmark_equity(
                        Path(args.data_root), eq.index.min(), eq.index.max(), preferred=["SPY"]
                    )
                    if b is not None and not b.empty:
                        b = b.copy()
                        b.index = pd.to_datetime(b.index, utc=True).normalize()
                        e2 = eq.copy()
                        e2.index = pd.to_datetime(e2.index, utc=True).normalize()
                        j = pd.concat(
                            [e2[~e2.index.duplicated()].rename("s"), b[~b.index.duplicated()].rename("b")],
                            axis=1,
                            join="inner",
                        ).dropna()
                        if len(j) > 2:
                            spy_tot = float(j["b"].iloc[-1] / j["b"].iloc[0] - 1.0)
            except Exception:
                pass
            loaded.append(
                {
                    "config_id": cid,
                    "equity": eq,
                    "trades": tdf,
                    "cagr": rep.cagr,
                    "total_return": total,
                    "sharpe": rep.sharpe,
                    "sortino": risk.sortino,
                    "max_drawdown": rep.max_drawdown,
                    "calmar": risk.calmar,
                    "n_trades": rep.n_trades,
                    "win_rate": rep.win_rate,
                    "profit_factor": rep.profit_factor,
                    "spy_total": spy_tot,
                    "excess_spy": (total - spy_tot) if spy_tot is not None else None,
                    "years_meta": meta.get("years"),
                }
            )
        except Exception as e:
            print(f"  FAIL {cid}: {e}", flush=True)

    loaded.sort(key=lambda x: float(x.get("cagr") or -9), reverse=True)
    top = loaded[: int(args.top)]
    print("TOP by CAGR:", [(r["config_id"], r["cagr"]) for r in top], flush=True)

    for i, r in enumerate(top, 1):
        safe = r["config_id"].replace("/", "_")
        html = build_dashboard(r, i, int(args.first), int(args.last))
        (out / f"dashboard_{safe}.html").write_text(html, encoding="utf-8")
        # also export csv copies at top level
        r["equity"].to_csv(out / f"equity_{safe}.csv", header=["equity"])
        if isinstance(r["trades"], pd.DataFrame) and not r["trades"].empty:
            r["trades"].to_csv(out / f"trades_{safe}.csv", index=False)
        print(f"  wrote dashboard_{safe}.html", flush=True)

    (out / "index.html").write_text(
        build_index(top, int(args.first), int(args.last)), encoding="utf-8"
    )
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "first": int(args.first),
        "last": int(args.last),
        "top": [
            {
                k: v
                for k, v in r.items()
                if k not in ("equity", "trades", "years_meta")
            }
            for r in top
        ],
        "disclaimer": "Research only. Not financial advice.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "INDEX.md").write_text(
        "\n".join(
            [
                f"# Top-5 CAGR {args.first}–{args.last}",
                "",
                "Open **[index.html](index.html)** for the dashboard index.",
                "",
                "| # | Config | CAGR | MDD | Trades |",
                "|---|--------|------|-----|--------|",
            ]
            + [
                f"| {i} | `{r['config_id']}` | {r['cagr']:.2%} | {r['max_drawdown']:.2%} | {r['n_trades']} |"
                for i, r in enumerate(top, 1)
            ]
            + ["", "Research only. Not financial advice.", ""]
        ),
        encoding="utf-8",
    )
    print(f"Index: {out / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
