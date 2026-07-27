"""YTD trade dashboard for turbo_highvol: vol-target sizing vs conservative cash sizing.

Clarifies: engine is **long-only cash** (no borrowed margin leverage).
"Aggressive" = vol-target 4% / max pos 22% (current turbo_highvol).
"Conservative" = vol-target 1% / max pos 10% / max 10 positions (no aggressive sizing).

Costs (BacktestConfig defaults used by research):
  commission = 0.10% of notional on buy AND sell
  slippage   = 0.05% adverse (buy mark-up / sell mark-down)
  round-trip ≈ 0.30% of notional before spreads
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.backtest import BacktestConfig
from trad_research.metrics import equity_metrics
from trad_research.strategies import get_strategy
from trad_research.strategy_runner import run_strategy_walk_forward
from trad_research.walk_forward import load_benchmark_equity


COMMISSION = 0.001  # 0.10%
SLIPPAGE = 0.0005  # 0.05%


def _run(
    *,
    name: str,
    label: str,
    year: int,
    data_root: Path,
    ticker_file: Path,
    universe_limit: int,
    overrides: Dict[str, Any],
    min_train_rows: int = 3000,
) -> Dict[str, Any]:
    strat = get_strategy(name)
    if hasattr(strat, "universe_source_file"):
        strat.universe_source_file = str(ticker_file)
    # Patch backtest_overrides to inject cost + sizing
    base_ov = strat.backtest_overrides()
    merged = {**base_ov, **overrides}
    merged["commission"] = float(overrides.get("commission", COMMISSION))
    merged["slippage"] = float(overrides.get("slippage", SLIPPAGE))

    orig = strat.backtest_overrides

    def _ov() -> Dict[str, Any]:
        return dict(merged)

    strat.backtest_overrides = _ov  # type: ignore[method-assign]
    try:
        res = run_strategy_walk_forward(
            strat,
            data_root=data_root,
            ticker_file=ticker_file,
            universe_limit=universe_limit,
            first_oos_year=year,
            last_oos_year=year,
            min_train_rows=min_train_rows,
            preferred_index=["SPY", "QQQ"],
            base_bt=BacktestConfig(
                commission=COMMISSION,
                slippage=SLIPPAGE,
            ),
        )
    finally:
        strat.backtest_overrides = orig  # type: ignore[method-assign]

    eq = res.get("equity")
    trades = res.get("trades")
    if eq is None or (hasattr(eq, "empty") and eq.empty):
        return {"label": label, "error": "empty equity"}
    eq = eq.dropna().astype(float)
    start_eq = float(eq.iloc[0])
    rep = equity_metrics(eq, start_equity=start_eq, trades=trades if isinstance(trades, pd.DataFrame) else None)
    total = float(eq.iloc[-1] / start_eq - 1.0)

    # benches
    spy_tot = qqq_tot = None
    try:
        for bname, key in [("SPY", "spy"), ("QQQ", "qqq")]:
            b = load_benchmark_equity(data_root, eq.index.min(), eq.index.max(), preferred=[bname])
            if b is None or b.empty:
                continue
            b.index = pd.to_datetime(b.index, utc=True).normalize()
            e2 = eq.copy()
            e2.index = pd.to_datetime(e2.index, utc=True).normalize()
            j = pd.concat([e2.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
            if len(j) > 2:
                tot = float(j["b"].iloc[-1] / j["b"].iloc[0] - 1.0)
                if key == "spy":
                    spy_tot = tot
                else:
                    qqq_tot = tot
    except Exception:
        pass

    tdf = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame()
    # Commission estimate from notional (entry capital_used includes entry commission)
    # Document round-trip on capital_used ≈ 2*commission + 2*slippage of mid notionals
    if not tdf.empty and "capital_used" in tdf.columns:
        # capital_used = cost + entry_comm; exit also pays commission
        # Approximate total commissions paid ≈ entry_comm + exit_comm
        # entry_comm ≈ capital_used * commission/(1+commission) ≈ capital_used * commission
        entry_comm = tdf["capital_used"] * COMMISSION / (1.0 + COMMISSION)
        # exit notional ≈ capital_used + net_profit + exit_comm ≈ use exit value
        if "net_profit" in tdf.columns:
            exit_proceeds_approx = tdf["capital_used"] + tdf["net_profit"]
            # net = proceeds - exit_comm, proceeds = shares*px_exit_slip
            # exit_comm ≈ proceeds * commission ≈ (net + exit_comm)*commission
            # exit_comm ≈ net * c/(1-c) + capital?  simpler: exit_comm ≈ (capital_used+net_profit)*c/(1-c) roughly
            exit_comm = (tdf["capital_used"] + tdf["net_profit"]).clip(lower=0) * COMMISSION / max(
                1e-9, (1.0 - COMMISSION)
            )
        else:
            exit_comm = tdf["capital_used"] * COMMISSION
        total_comm = float(entry_comm.sum() + exit_comm.sum())
    else:
        total_comm = 0.0

    return {
        "label": label,
        "strategy": name,
        "overrides": merged,
        "total_return": total,
        "sharpe": rep.sharpe,
        "max_drawdown": rep.max_drawdown,
        "n_trades": rep.n_trades,
        "win_rate": rep.win_rate,
        "profit_factor": rep.profit_factor,
        "final_equity": float(eq.iloc[-1]),
        "start_equity": start_eq,
        "spy_total": spy_tot,
        "qqq_total": qqq_tot,
        "excess_spy": (total - spy_tot) if spy_tot is not None else None,
        "approx_total_commissions_usd": total_comm,
        "commission_rate": COMMISSION,
        "slippage_rate": SLIPPAGE,
        "round_trip_cost_pct_approx": 2 * COMMISSION + 2 * SLIPPAGE,
        "equity": eq,
        "trades": tdf,
        "year_results": res.get("year_results"),
        "note_leverage": (
            "Long-only cash account: buys limited by cash (no margin borrow). "
            "Aggressive = larger vol-target position sizes, not 2x broker leverage."
        ),
    }


def _trades_html_table(tdf: pd.DataFrame) -> str:
    if tdf is None or tdf.empty:
        return "<p><em>Sin trades.</em></p>"
    cols_pref = [
        "ticker",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "shares",
        "capital_used",
        "net_profit",
        "trade_return",
        "exit_reason",
    ]
    cols = [c for c in cols_pref if c in tdf.columns] + [
        c for c in tdf.columns if c not in cols_pref
    ]
    show = tdf[cols].copy()
    for c in ("entry_date", "exit_date"):
        if c in show.columns:
            show[c] = pd.to_datetime(show[c], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ("entry_price", "exit_price", "capital_used", "net_profit"):
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                lambda x: f"{x:,.2f}" if pd.notna(x) else ""
            )
    if "trade_return" in show.columns:
        show["trade_return"] = pd.to_numeric(tdf["trade_return"], errors="coerce").map(
            lambda x: f"{x:.2%}" if pd.notna(x) else ""
        )
    # color rows by profit
    rows_html = []
    for i, row in show.iterrows():
        pnl_raw = tdf.loc[i, "net_profit"] if "net_profit" in tdf.columns else 0
        try:
            pnl_f = float(pnl_raw)
        except Exception:
            pnl_f = 0.0
        cls = "win" if pnl_f > 0 else ("loss" if pnl_f < 0 else "")
        tds = "".join(f"<td>{row[c]}</td>" for c in show.columns)
        rows_html.append(f'<tr class="{cls}">{tds}</tr>')
    head = "".join(f"<th>{c}</th>" for c in show.columns)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"


def _equity_js(eq: pd.Series, name: str) -> str:
    eq = eq.copy()
    eq.index = pd.to_datetime(eq.index, utc=True)
    xs = [d.strftime("%Y-%m-%d") for d in eq.index]
    ys = [round(float(v), 2) for v in eq.values]
    return json.dumps({"name": name, "x": xs, "y": ys})


def build_html(agg: Dict[str, Any], cons: Dict[str, Any], out: Path) -> str:
    def block(r: Dict[str, Any], title: str) -> str:
        if r.get("error"):
            return f"<h2>{title}</h2><p class='err'>{r['error']}</p>"
        tdf = r["trades"]
        wins = int((tdf["net_profit"] > 0).sum()) if not tdf.empty and "net_profit" in tdf.columns else 0
        losses = int((tdf["net_profit"] <= 0).sum()) if not tdf.empty and "net_profit" in tdf.columns else 0
        gp = float(tdf.loc[tdf["net_profit"] > 0, "net_profit"].sum()) if wins else 0.0
        gl = float(-tdf.loc[tdf["net_profit"] <= 0, "net_profit"].sum()) if losses else 0.0
        return f"""
        <h2>{title}</h2>
        <div class="cards">
          <div class="card"><div class="k">Retorno total</div><div class="v">{r['total_return']:.2%}</div></div>
          <div class="card"><div class="k">Sharpe</div><div class="v">{r['sharpe']:.2f}</div></div>
          <div class="card"><div class="k">Max DD</div><div class="v">{r['max_drawdown']:.2%}</div></div>
          <div class="card"><div class="k">Trades</div><div class="v">{r['n_trades']} (W{wins}/L{losses})</div></div>
          <div class="card"><div class="k">Win rate</div><div class="v">{r['win_rate']:.1%}</div></div>
          <div class="card"><div class="k">Profit factor</div><div class="v">{r['profit_factor']:.2f}</div></div>
          <div class="card"><div class="k">vs SPY</div><div class="v">{(r['excess_spy'] if r['excess_spy'] is not None else float('nan')):.2%}</div></div>
          <div class="card"><div class="k">Comisiones ≈</div><div class="v">${r['approx_total_commissions_usd']:,.0f}</div></div>
        </div>
        <p class="meta">Sizing: vol_target={r['overrides'].get('volatility_target_pct')} ·
        max_pos={r['overrides'].get('max_position_pct')} · max_positions={r['overrides'].get('max_positions')} ·
        commission={r['commission_rate']:.2%} · slippage={r['slippage_rate']:.2%} ·
        round-trip≈{r['round_trip_cost_pct_approx']:.2%}</p>
        <p class="meta">PnL bruto wins ${gp:,.0f} · losses ${gl:,.0f} · net trades ${gp-gl:,.0f}</p>
        <h3>Operaciones</h3>
        {_trades_html_table(tdf)}
        """

    eq_a = _equity_js(agg["equity"], "Aggressive vol-target (cash)")
    eq_c = _equity_js(cons["equity"], "Conservative sizing (cash)")
    # SPY series if possible
    spy_js = "null"
    try:
        eq = agg["equity"]
        b = load_benchmark_equity(
            ROOT / "data", eq.index.min(), eq.index.max(), preferred=["SPY"]
        )
        b.index = pd.to_datetime(b.index, utc=True).normalize()
        e2 = eq.copy()
        e2.index = pd.to_datetime(e2.index, utc=True).normalize()
        j = pd.concat([e2.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
        if len(j) > 2:
            b0 = float(j["b"].iloc[0])
            s0 = float(j["s"].iloc[0])
            spy_y = (j["b"] / b0 * s0).tolist()
            spy_x = [d.strftime("%Y-%m-%d") for d in j.index]
            spy_js = json.dumps({"name": "SPY (scaled)", "x": spy_x, "y": [round(v, 2) for v in spy_y]})
    except Exception:
        pass

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>YTD dashboard turbo_highvol — con/sin sizing agresivo</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
 body {{ font-family: system-ui,Segoe UI,sans-serif; margin: 24px; background:#0f1419; color:#e7ecf1; }}
 h1,h2,h3 {{ color:#f3f6f9; }}
 .cards {{ display:flex; flex-wrap:wrap; gap:12px; margin:12px 0 20px; }}
 .card {{ background:#1a2332; border:1px solid #2a3a4f; border-radius:10px; padding:12px 16px; min-width:120px; }}
 .card .k {{ font-size:12px; color:#9fb0c3; }}
 .card .v {{ font-size:20px; font-weight:700; margin-top:4px; }}
 table {{ border-collapse:collapse; width:100%; font-size:13px; margin:12px 0 32px; }}
 th,td {{ border:1px solid #2a3a4f; padding:6px 8px; text-align:right; }}
 th {{ background:#1a2332; position:sticky; top:0; }}
 td:first-child, th:first-child {{ text-align:left; }}
 tr.win td {{ background:rgba(34,197,94,0.08); }}
 tr.loss td {{ background:rgba(239,68,68,0.10); }}
 .meta {{ color:#9fb0c3; font-size:13px; }}
 .warn {{ background:#3b2f1a; border-left:4px solid #f59e0b; padding:12px 16px; margin:16px 0; }}
 .ok {{ background:#14301f; border-left:4px solid #22c55e; padding:12px 16px; margin:16px 0; }}
 a {{ color:#7dd3fc; }}
</style>
</head>
<body>
<h1>Dashboard 2026 YTD — <code>turbo_highvol</code> (#1 bake-off)</h1>
<p class="meta">Generado {date.today().isoformat()} · Capital inicial $100,000 · Universo highvol80</p>

<div class="ok">
<strong>¿Hay apalancamiento prestado (margin 2×)?</strong> <b>NO.</b><br/>
El motor es <b>long-only en efectivo</b>: cada compra se paga con cash disponible
(<code>alloc ≤ cash × 0.98</code>). No hay short ni deuda de bróker.<br/>
Lo que <em>sí</em> hay es <b>sizing por volatilidad</b> (posiciones grandes si ATR/precio es bajo),
capadas a <code>max_position_pct</code> (22% agresivo / 10% conservador). Eso concentra el libro,
pero no es apalancamiento financiero.
</div>

<div class="warn">
<strong>Comisiones y fricción (por lado, cada compra y cada venta):</strong><br/>
• Comisión: <b>{COMMISSION:.2%}</b> del nocional<br/>
• Slippage: <b>{SLIPPAGE:.2%}</b> adverso (compra más cara / venta más barata)<br/>
• Ida y vuelta aproximada: <b>{2*COMMISSION+2*SLIPPAGE:.2%}</b> del nocional<br/>
Ejemplo: posición de $20,000 → comisión entrada $20 + salida ~$20 + slippage ~$10+$10 ≈ <b>$60</b> round-trip.
</div>

<div id="eqchart" style="height:420px;margin:24px 0;"></div>

{block(agg, "Ventana A — sizing AGRESIVO (como bake-off #1: vol_target 4%, max pos 22%)")}
{block(cons, "Ventana B — sizing CONSERVADOR (sin vol-target agresivo: vol_target 1%, max pos 10%)")}

<p class="meta">Research only. No es consejo financiero. Past partial-year results ≠ future.</p>
<script>
const A = {eq_a};
const C = {eq_c};
const S = {spy_js};
const traces = [
  {{x:A.x, y:A.y, name:A.name, type:'scatter', mode:'lines', line:{{color:'#f97316', width:2}}}},
  {{x:C.x, y:C.y, name:C.name, type:'scatter', mode:'lines', line:{{color:'#38bdf8', width:2}}}},
];
if (S) traces.push({{x:S.x, y:S.y, name:S.name, type:'scatter', mode:'lines', line:{{color:'#94a3b8', width:1.5, dash:'dot'}}}});
Plotly.newPlot('eqchart', traces, {{
  paper_bgcolor:'#0f1419', plot_bgcolor:'#0f1419',
  font:{{color:'#e7ecf1'}},
  title:'Equity curve 2026 YTD',
  legend:{{orientation:'h'}},
  margin:{{t:40,r:20,b:40,l:60}},
  xaxis:{{gridcolor:'#1f2a37'}}, yaxis:{{gridcolor:'#1f2a37', tickprefix:'$'}}
}}, {{responsive:true}});
</script>
</body>
</html>
"""
    return html


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--strategy", type=str, default="turbo_highvol")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--ticker-file", type=Path, default=ROOT / "universe_highvol80.txt")
    ap.add_argument("--universe-limit", type=int, default=80)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "ytd_2026_trade_dashboard",
    )
    args = ap.parse_args(argv)

    data_root = Path(args.data_root)
    ticker_file = Path(args.ticker_file)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    aggressive_ov = {
        "volatility_target_pct": 0.04,
        "max_position_pct": 0.22,
        "max_positions": 16,
        "min_alloc_pct": 0.0,
        "commission": COMMISSION,
        "slippage": SLIPPAGE,
    }
    conservative_ov = {
        "volatility_target_pct": 0.01,
        "max_position_pct": 0.10,
        "max_positions": 10,
        "min_alloc_pct": 0.0,
        "commission": COMMISSION,
        "slippage": SLIPPAGE,
    }

    print("Running AGGRESSIVE sizing…", flush=True)
    agg = _run(
        name=args.strategy,
        label="aggressive_vol_target",
        year=args.year,
        data_root=data_root,
        ticker_file=ticker_file,
        universe_limit=args.universe_limit,
        overrides=aggressive_ov,
    )
    print(
        f"  total={agg.get('total_return')} trades={agg.get('n_trades')} mdd={agg.get('max_drawdown')}",
        flush=True,
    )

    print("Running CONSERVATIVE sizing…", flush=True)
    cons = _run(
        name=args.strategy,
        label="conservative_cash",
        year=args.year,
        data_root=data_root,
        ticker_file=ticker_file,
        universe_limit=args.universe_limit,
        overrides=conservative_ov,
    )
    print(
        f"  total={cons.get('total_return')} trades={cons.get('n_trades')} mdd={cons.get('max_drawdown')}",
        flush=True,
    )

    # Save trades CSV
    for tag, r in [("aggressive", agg), ("conservative", cons)]:
        tdf = r.get("trades")
        if isinstance(tdf, pd.DataFrame) and not tdf.empty:
            tdf.to_csv(out / f"trades_{tag}.csv", index=False)
        eq = r.get("equity")
        if isinstance(eq, pd.Series):
            eq.to_csv(out / f"equity_{tag}.csv", header=["equity"])

    html = build_html(agg, cons, out)
    (out / "dashboard.html").write_text(html, encoding="utf-8")

    summary = {
        "strategy": args.strategy,
        "year": args.year,
        "leverage_borrowed": False,
        "commission_pct_per_side": COMMISSION,
        "slippage_pct_per_side": SLIPPAGE,
        "round_trip_pct_approx": 2 * COMMISSION + 2 * SLIPPAGE,
        "aggressive": {
            k: v
            for k, v in agg.items()
            if k not in ("equity", "trades")
        },
        "conservative": {
            k: v
            for k, v in cons.items()
            if k not in ("equity", "trades")
        },
    }
    # serialize numpy
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md = [
        f"# Trade dashboard {args.strategy} — {args.year} YTD",
        "",
        "## Apalancamiento",
        "",
        "**No hay apalancamiento prestado (margin).** Cuenta long-only cash.",
        "La ventana “agresiva” usa **vol-target 4% / max 22% por nombre** (más concentración).",
        "La ventana “conservadora” usa **vol-target 1% / max 10% por nombre**.",
        "",
        "## Costes",
        "",
        f"- Comisión: **{COMMISSION:.2%}** del nocional por compra y por venta",
        f"- Slippage: **{SLIPPAGE:.2%}** por lado",
        f"- Round-trip approx: **{2*COMMISSION+2*SLIPPAGE:.2%}**",
        "",
        "## Resultados",
        "",
        "| Modo | Total | Sharpe | MDD | Trades | vs SPY | Comisiones ≈ |",
        "|------|-------|--------|-----|--------|--------|--------------|",
        f"| Agresivo (bake-off) | {agg.get('total_return',0):.2%} | {agg.get('sharpe',0):.2f} | {agg.get('max_drawdown',0):.2%} | {agg.get('n_trades')} | {agg.get('excess_spy')} | ${agg.get('approx_total_commissions_usd',0):,.0f} |",
        f"| Conservador | {cons.get('total_return',0):.2%} | {cons.get('sharpe',0):.2f} | {cons.get('max_drawdown',0):.2%} | {cons.get('n_trades')} | {cons.get('excess_spy')} | ${cons.get('approx_total_commissions_usd',0):,.0f} |",
        "",
        f"Dashboard HTML: `{out / 'dashboard.html'}`",
        "",
        "Research only.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
