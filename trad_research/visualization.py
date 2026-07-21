"""Charts and HTML dashboard for equity, drawdown, and trades."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def equity_to_drawdown(equity: pd.Series) -> pd.Series:
    eq = equity.astype(float).dropna().sort_index()
    if eq.empty:
        return eq
    peak = eq.cummax()
    return eq / peak - 1.0


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_equity_and_drawdown(
    equity: pd.Series,
    *,
    title: str = "Equity",
    benchmark: Optional[pd.Series] = None,
    start_equity: float = 100_000.0,
) -> str:
    """Return base64 PNG with equity (top) and drawdown (bottom)."""
    eq = equity.astype(float).dropna().sort_index()
    if eq.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No equity data", ha="center")
        return _fig_to_b64(fig)

    dd = equity_to_drawdown(eq)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax1.plot(eq.index, eq.values, color="#1f77b4", lw=1.4, label="Strategy")
    ax1.axhline(start_equity, color="gray", ls="--", lw=0.8, alpha=0.7)
    if benchmark is not None and not benchmark.empty:
        b = benchmark.reindex(eq.index).ffill().dropna()
        if len(b) > 2:
            # scale BH to start_equity
            b_scaled = b / float(b.iloc[0]) * start_equity
            ax1.plot(b_scaled.index, b_scaled.values, color="#ff7f0e", lw=1.0, alpha=0.85, label="Benchmark")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.45)
    ax2.plot(dd.index, dd.values, color="#8b0000", lw=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_trade_pnl_by_ticker(trades: pd.DataFrame, top_n: int = 15, title: str = "PnL by ticker") -> str:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if trades is None or trades.empty or "ticker" not in trades.columns:
        ax.text(0.5, 0.5, "No trades", ha="center")
        return _fig_to_b64(fig)
    g = trades.groupby("ticker")["net_profit"].sum().sort_values(ascending=False)
    top = pd.concat([g.head(top_n), g.tail(min(5, len(g)))]).drop_duplicates()
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in top.values]
    ax.barh(top.index.astype(str)[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("Net profit ($)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_monthly_returns_heatmap(equity: pd.Series, title: str = "Monthly returns") -> str:
    eq = equity.astype(float).dropna().sort_index()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    if len(eq) < 30:
        ax.text(0.5, 0.5, "Not enough data", ha="center")
        return _fig_to_b64(fig)
    # month-end last equity
    m = eq.resample("ME").last().pct_change().dropna()
    if m.empty:
        ax.text(0.5, 0.5, "No monthly returns", ha="center")
        return _fig_to_b64(fig)
    df = pd.DataFrame({"ret": m.values}, index=m.index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc="last")
    data = pivot.fillna(0).values
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.15, vmax=0.15)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    return _fig_to_b64(fig)


def trades_html_table(trades: pd.DataFrame, max_rows: int = 80) -> str:
    if trades is None or trades.empty:
        return "<p><em>Sin trades.</em></p>"
    cols = [
        c
        for c in (
            "ticker",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "shares",
            "net_profit",
            "trade_return",
            "exit_reason",
            "oos_year",
        )
        if c in trades.columns
    ]
    t = trades[cols].copy()
    # show most recent first
    if "exit_date" in t.columns:
        t = t.sort_values("exit_date", ascending=False)
    t = t.head(max_rows)
    for c in ("entry_date", "exit_date"):
        if c in t.columns:
            t[c] = pd.to_datetime(t[c], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    if "net_profit" in t.columns:
        t["net_profit"] = t["net_profit"].map(lambda x: f"{x:,.0f}")
    if "trade_return" in t.columns:
        t["trade_return"] = t["trade_return"].map(lambda x: f"{x:.1%}")
    if "entry_price" in t.columns:
        t["entry_price"] = t["entry_price"].map(lambda x: f"{x:.2f}")
    if "exit_price" in t.columns:
        t["exit_price"] = t["exit_price"].map(lambda x: f"{x:.2f}")
    return t.to_html(index=False, classes="trades", border=0, escape=True)


def scenario_summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    rep = result.get("report")
    return {
        "scenario_id": result.get("scenario_id", ""),
        "market": result.get("market", ""),
        "strategy": result.get("strategy", ""),
        "window": result.get("window", ""),
        "universe": result.get("universe_label", ""),
        "passed": result.get("passed", False),
        "cagr": getattr(rep, "cagr", None) if rep else None,
        "sharpe": getattr(rep, "sharpe", None) if rep else None,
        "max_drawdown": getattr(rep, "max_drawdown", None) if rep else None,
        "total_return": getattr(rep, "total_return", None) if rep else None,
        "n_trades": getattr(rep, "n_trades", None) if rep else None,
        "final_equity": getattr(rep, "final_equity", None) if rep else None,
        "error": result.get("error"),
    }


def build_dashboard_html(
    scenarios: Sequence[Dict[str, Any]],
    *,
    title: str = "TRAD Multi-Market Dashboard",
    start_equity: float = 100_000.0,
) -> str:
    rows = [scenario_summary_row(s) for s in scenarios]
    summary = pd.DataFrame(rows)

    def fmt_pct(x):
        return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"

    def fmt_num(x):
        return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}"

    table_html = "<table class='summary'><thead><tr>"
    headers = [
        "scenario_id",
        "market",
        "strategy",
        "window",
        "universe",
        "passed",
        "cagr",
        "sharpe",
        "max_drawdown",
        "n_trades",
        "final_equity",
    ]
    for h in headers:
        table_html += f"<th>{h}</th>"
    table_html += "</tr></thead><tbody>"
    for _, r in summary.iterrows():
        table_html += "<tr>"
        table_html += f"<td><a href='#{r['scenario_id']}'>{r['scenario_id']}</a></td>"
        table_html += f"<td>{r['market']}</td><td>{r['strategy']}</td><td>{r['window']}</td>"
        table_html += f"<td>{r['universe']}</td>"
        ok = r.get("passed")
        badge = "PASS" if ok else ("ERR" if r.get("error") else "fail")
        cls = "pass" if ok else "fail"
        table_html += f"<td class='{cls}'>{badge}</td>"
        sh = r["sharpe"]
        sh_s = "" if sh is None or (isinstance(sh, float) and np.isnan(sh)) else f"{float(sh):.2f}"
        table_html += f"<td>{fmt_pct(r['cagr'])}</td><td>{sh_s}</td>"
        table_html += f"<td>{fmt_pct(r['max_drawdown'])}</td><td>{r['n_trades']}</td>"
        table_html += f"<td>{fmt_num(r['final_equity'])}</td></tr>"
    table_html += "</tbody></table>"

    sections = []
    for s in scenarios:
        sid = s.get("scenario_id", "s")
        if s.get("error"):
            sections.append(
                f"<section id='{sid}'><h2>{sid}</h2><p class='fail'>ERROR: {s['error']}</p></section>"
            )
            continue
        eq = s.get("equity")
        trades = s.get("trades")
        bench = s.get("benchmark")
        rep = s["report"]
        title_s = (
            f"{s.get('market')} · {s.get('strategy')} · {s.get('window')} · {s.get('universe_label')}"
        )
        img_eq = plot_equity_and_drawdown(
            eq, title=f"Dinero día a día — {title_s}", benchmark=bench, start_equity=start_equity
        )
        img_tk = plot_trade_pnl_by_ticker(trades, title=f"PnL por acción — {sid}")
        img_m = plot_monthly_returns_heatmap(eq, title=f"Retornos mensuales — {sid}")
        years = s.get("year_results") or []
        yhtml = "<ul>" + "".join(
            f"<li>{y['year']}: {y['year_return']:+.1%} · trades={y['n_trades']} · "
            f"sharpe={y['sharpe']:.2f} · mdd={y['max_drawdown']:.1%}</li>"
            for y in years
        ) + "</ul>"
        sections.append(
            f"""
<section id="{sid}">
  <h2>{sid}</h2>
  <p><strong>{title_s}</strong></p>
  <ul>
    <li>CAGR <b>{rep.cagr:.1%}</b> · Sharpe <b>{rep.sharpe:.2f}</b> · MaxDD <b>{rep.max_drawdown:.1%}</b></li>
    <li>Total return <b>{rep.total_return:.1%}</b> · Final equity <b>${rep.final_equity:,.0f}</b> · Trades <b>{rep.n_trades}</b></li>
    <li>Gates research: <b class="{'pass' if s.get('passed') else 'fail'}">{'PASS' if s.get('passed') else 'FAIL'}</b></li>
  </ul>
  <h3>Equity &amp; Drawdown (día a día)</h3>
  <img src="data:image/png;base64,{img_eq}" alt="equity"/>
  <h3>Heatmap mensual</h3>
  <img src="data:image/png;base64,{img_m}" alt="monthly"/>
  <h3>Acciones (PnL agregado)</h3>
  <img src="data:image/png;base64,{img_tk}" alt="tickers"/>
  <h3>Por año OOS</h3>
  {yhtml}
  <h3>Trades (más recientes, máx 80)</h3>
  {trades_html_table(trades)}
</section>
"""
        )

    css = """
    body { font-family: Segoe UI, system-ui, sans-serif; margin: 24px; background: #0f1115; color: #e8eaed; }
    h1,h2,h3 { color: #fff; }
    a { color: #8ab4f8; }
    .summary { border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 32px; }
    .summary th, .summary td { border: 1px solid #333; padding: 6px 8px; text-align: left; }
    .summary th { background: #1b1f27; position: sticky; top: 0; }
    .summary tr:nth-child(even) { background: #161a22; }
    .pass { color: #81c995; font-weight: 600; }
    .fail { color: #f28b82; font-weight: 600; }
    section { margin: 48px 0; padding-top: 12px; border-top: 1px solid #333; }
    img { max-width: 100%; background: #fff; border-radius: 6px; padding: 4px; }
    table.trades { border-collapse: collapse; font-size: 12px; width: 100%; }
    table.trades th, table.trades td { border: 1px solid #333; padding: 4px 6px; }
    table.trades th { background: #1b1f27; }
    .note { color: #9aa0a6; font-size: 13px; max-width: 900px; }
    """
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<title>{title}</title>
<style>{css}</style>
</head><body>
<h1>{title}</h1>
<p class="note">
Walk-forward OOS por escenario. Capital inicial ${start_equity:,.0f}.
Equity = valor de cartera día a día. Drawdown = caída desde máximo previo.
US: data/ + QQQ/SPY régimen. ES: data_es/ + IBEX. Universos highvol fijados as-of pre-ventana cuando aplica.
</p>
<h2>Resumen de todos los escenarios</h2>
{table_html}
{''.join(sections)}
</body></html>
"""


def save_scenario_artifacts(
    out_dir: Path,
    scenario_id: str,
    result: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eq = result.get("equity")
    trades = result.get("trades")
    if eq is not None and not getattr(eq, "empty", True):
        eq.to_csv(out_dir / f"{scenario_id}_equity.csv", header=["equity"])
        equity_to_drawdown(eq).to_csv(out_dir / f"{scenario_id}_drawdown.csv", header=["drawdown"])
    if trades is not None and not getattr(trades, "empty", True):
        trades.to_csv(out_dir / f"{scenario_id}_trades.csv", index=False)
