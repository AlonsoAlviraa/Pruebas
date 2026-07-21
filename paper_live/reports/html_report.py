"""HTML dashboard for paper digests (self-contained, no external CDN required)."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from paper_live.reports.daily_digest import DailyDigest
from paper_live.reports.weekly_scorecard import WeeklyScorecard


def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _money(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"${float(x):,.2f}"


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{float(x):.2%}"


def render_html_dashboard(
    *,
    title: str,
    run_id: str,
    strategy_id: str,
    daily: Sequence[DailyDigest] = (),
    weekly: Optional[WeeklyScorecard] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a single self-contained HTML page."""
    daily = list(daily)
    eq_labels = [d.day for d in daily if d.equity is not None]
    eq_vals = [float(d.equity) for d in daily if d.equity is not None]
    comm_vals = [float(d.commission) for d in daily]

    summary = summary or {}
    weekly_block = ""
    if weekly is not None:
        flags_html = "".join(f"<li>{_esc(f)}</li>" for f in weekly.flags) or "<li>None</li>"
        weekly_block = f"""
        <section class="card">
          <h2>Weekly scorecard ({_esc(weekly.week_start)} → {_esc(weekly.week_end)})</h2>
          <div class="grid">
            <div><span class="lbl">Week return</span><div class="val">{_pct(weekly.week_return)}</div></div>
            <div><span class="lbl">Max DD</span><div class="val">{_pct(weekly.max_dd)}</div></div>
            <div><span class="lbl">Sharpe approx</span><div class="val">{_esc(weekly.rolling_sharpe_approx)}</div></div>
            <div><span class="lbl">Fills</span><div class="val">{weekly.n_fills}</div></div>
            <div><span class="lbl">Commission</span><div class="val">{_money(weekly.commission)}</div></div>
            <div><span class="lbl">Cost drag bps</span><div class="val">{_esc(weekly.cost_drag_bps)}</div></div>
            <div><span class="lbl">Micro fills</span><div class="val">{weekly.micro_trade_pct:.1%}</div></div>
            <div><span class="lbl">Kill events</span><div class="val">{weekly.n_kill_events}</div></div>
          </div>
          <h3>Flags</h3>
          <ul>{flags_html}</ul>
        </section>
        """

    rows = []
    for d in daily:
        rows.append(
            "<tr>"
            f"<td>{_esc(d.day)}</td>"
            f"<td>{_money(d.equity)}</td>"
            f"<td>{_pct(d.dd_from_peak)}</td>"
            f"<td>{d.n_fills}</td>"
            f"<td>{_money(d.commission)}</td>"
            f"<td>{_money(d.fees)}</td>"
            f"<td>{d.n_rejects}</td>"
            f"<td>{d.n_kill_events}</td>"
            f"<td>{d.n_positions}</td>"
            "</tr>"
        )
    table_body = "\n".join(rows) if rows else "<tr><td colspan='9'>No daily rows</td></tr>"

    # Simple SVG sparkline for equity
    spark = _svg_sparkline(eq_vals) if len(eq_vals) >= 2 else "<p class='muted'>Need ≥2 equity points</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{_esc(title)}</title>
  <style>
    :root {{
      --bg: #0f1419; --card: #1a2332; --text: #e7ecf3; --muted: #8b9bb4;
      --accent: #3d9cf0; --good: #3dd68c; --bad: #f07178; --border: #2a3548;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: ui-sans-serif, system-ui, Segoe UI, Roboto, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.45; padding: 1.5rem;
    }}
    h1 {{ font-size: 1.4rem; margin: 0 0 0.25rem; }}
    h2 {{ font-size: 1.1rem; margin: 0 0 0.75rem; color: var(--accent); }}
    h3 {{ font-size: 0.95rem; color: var(--muted); }}
    .sub {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.25rem; }}
    .badge {{
      display: inline-block; background: #2a3f5f; color: #9ecbff;
      padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem; margin-right: 0.35rem;
    }}
    .badge.warn {{ background: #4a3020; color: #ffcc99; }}
    .card {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 1rem 1.15rem; margin-bottom: 1rem;
    }}
    .grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem;
    }}
    .lbl {{ display: block; color: var(--muted); font-size: 0.75rem; }}
    .val {{ font-size: 1.05rem; font-weight: 600; margin-top: 0.15rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    th, td {{ text-align: left; padding: 0.45rem 0.5rem; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; }}
    .muted {{ color: var(--muted); }}
    footer {{ margin-top: 1.5rem; color: var(--muted); font-size: 0.8rem; }}
    svg.spark {{ width: 100%; height: 80px; background: #121a26; border-radius: 8px; }}
  </style>
</head>
<body>
  <header>
    <h1>{_esc(title)}</h1>
    <div class="sub">
      <span class="badge">PAPER</span>
      <span class="badge">VIRTUAL CAPITAL</span>
      <span class="badge warn">NOT FINANCIAL ADVICE</span>
    </div>
    <div class="sub">
      Run <code>{_esc(run_id)}</code> · Strategy <code>{_esc(strategy_id)}</code>
      · Days {len(daily)}
    </div>
  </header>

  <section class="card">
    <h2>Equity path</h2>
    {spark}
  </section>

  {weekly_block}

  <section class="card">
    <h2>Daily table</h2>
    <table>
      <thead>
        <tr>
          <th>Date</th><th>Equity</th><th>DD</th><th>Fills</th>
          <th>Comm</th><th>Fees</th><th>Rejects</th><th>Kill</th><th>Pos</th>
        </tr>
      </thead>
      <tbody>
        {table_body}
      </tbody>
    </table>
  </section>

  <footer>
    Paper live year dashboard (LIV-08). Commissions and fees from ledger fills.
    Past paper results do not guarantee future results.
  </footer>
  <script type="application/json" id="equity-data">{json.dumps({"labels": eq_labels, "equity": eq_vals, "commission": comm_vals})}</script>
</body>
</html>
"""


def _svg_sparkline(values: Sequence[float], width: int = 640, height: int = 80) -> str:
    if not values:
        return ""
    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1.0
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = 8 + (width - 16) * (i / max(n - 1, 1))
        y = height - 8 - (height - 16) * ((v - vmin) / span)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    last = values[-1]
    color = "#3dd68c" if last >= values[0] else "#f07178"
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{poly}"/>'
        f"</svg>"
    )


def write_html_dashboard(
    out_path: Union[str, Path],
    *,
    title: str = "Paper Live Dashboard",
    run_id: str,
    strategy_id: str,
    daily: Sequence[DailyDigest] = (),
    weekly: Optional[WeeklyScorecard] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html_doc = render_html_dashboard(
        title=title,
        run_id=run_id,
        strategy_id=strategy_id,
        daily=daily,
        weekly=weekly,
        summary=summary,
    )
    path.write_text(html_doc, encoding="utf-8")
    return path
