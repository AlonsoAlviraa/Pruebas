"""Multi-hour redesign v2 mega loop: new features/strategies + screen/confirm + graphs.

Pre-registered protocol:
  - Screen OOS 2010–2017 (rank only)
  - Confirm OOS 2018–2025 (gates)
  - Full stitch 2010–2025 (honesty)
  - Gates: CAGR>10%, MDD≥−65%, n_trades≥80 on confirm
  - No soft-ban, no paper freeze auto-change

Resume via PROGRESS.json. Research only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.backtest import BacktestConfig  # noqa: E402
from trad_research.metrics import equity_metrics  # noqa: E402
from trad_research.redesign_v2.graph_math import (  # noqa: E402
    graph_summary_dict,
    graph_to_html,
    hub_scores,
    trade_cooccurrence_graph,
)
from trad_research.risk_metrics import extended_risk_from_equity  # noqa: E402
from trad_research.strategies import get_strategy  # noqa: E402
from trad_research.strategy_runner import run_strategy_walk_forward  # noqa: E402
from trad_research.walk_forward import load_benchmark_equity  # noqa: E402

COMMISSION = 0.001
SLIPPAGE = 0.0005
GATE_CAGR = 0.10
GATE_MDD = -0.65
GATE_TRADES = 80

# Pre-registered zoo (structurally distinct)
STRATEGIES: Tuple[str, ...] = (
    "turbo_highvol_minalloc",  # control
    "turbo_strict",
    "champion_ml",
    "r2_residual_mom",
    "r2_mom_sharpe",
    "r2_trend_stack",
    "r2_defensive_vt",
    "r2_rsi_reclaim",
)

UNIVERSE_ARMS: Tuple[Tuple[str, Path, int], ...] = (
    ("longhist_L50", ROOT / "universe_longhist2010_pass.txt", 50),
    ("longhist_L80", ROOT / "universe_longhist2010_pass.txt", 80),
    ("highvol2010_L50", ROOT / "universe_highvol80_2010_pass.txt", 50),
)


@dataclass
class Arm:
    arm_id: str
    strategy: str
    universe_label: str
    ticker_file: Path
    universe_limit: int


def _eq_norm(s: pd.Series) -> pd.Series:
    out = s.dropna().astype(float)
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    return out[~out.index.duplicated(keep="last")].dropna().sort_index()


def _stitch(a: pd.Series, b: pd.Series) -> pd.Series:
    segs = []
    prev = None
    for seg in (a, b):
        s = _eq_norm(seg)
        if s.empty:
            continue
        if prev is not None and float(s.iloc[0]) != 0:
            s = s * (prev / float(s.iloc[0]))
        segs.append(s)
        prev = float(s.iloc[-1])
    if not segs:
        return pd.Series(dtype=float)
    out = pd.concat(segs)
    return out[~out.index.duplicated(keep="last")].sort_index()


def _metrics(eq: pd.Series, trades: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    eq = _eq_norm(eq)
    if eq.empty:
        return {"error": "empty", "cagr": 0.0, "max_drawdown": -1.0, "n_trades": 0}
    start = float(eq.iloc[0])
    tdf = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame()
    rep = equity_metrics(eq, start_equity=start, trades=tdf if not tdf.empty else None)
    risk = extended_risk_from_equity(
        eq.to_numpy(),
        trade_pnls=tdf["net_profit"].to_numpy()
        if not tdf.empty and "net_profit" in tdf.columns
        else None,
    )
    return {
        "cagr": float(rep.cagr),
        "sharpe": float(rep.sharpe),
        "sortino": float(risk.sortino),
        "max_drawdown": float(rep.max_drawdown),
        "n_trades": int(rep.n_trades),
        "win_rate": float(rep.win_rate) if rep.win_rate is not None else None,
        "total_return": float(eq.iloc[-1] / start - 1.0),
    }


def _gates(m: Dict[str, Any]) -> Dict[str, Any]:
    cagr = m.get("cagr")
    mdd = m.get("max_drawdown")
    n = m.get("n_trades")
    cagr_f = float(cagr) if cagr is not None else 0.0
    mdd_f = float(mdd) if mdd is not None else -1.0
    n_i = int(n) if n is not None else 0
    ok_c = cagr_f > GATE_CAGR
    ok_m = mdd_f >= GATE_MDD
    ok_t = n_i >= GATE_TRADES
    return {
        "cagr_ok": ok_c,
        "mdd_ok": ok_m,
        "trades_ok": ok_t,
        "pass": bool(ok_c and ok_m and ok_t),
    }


def _honest_score(m: Dict[str, Any], xs: Optional[float]) -> float:
    cagr = float(m.get("cagr") or 0.0)
    sortino = float(m.get("sortino") or 0.0)
    mdd = float(m.get("max_drawdown") or -1.0)
    score = 2.0 * cagr + 1.0 * sortino
    if xs is not None:
        score += 0.5 * max(0.0, float(xs))
    if mdd < -0.50:
        score -= 2.0 * ((-0.50) - mdd)
    return float(score)


def _spy_excess(eq: pd.Series, data_root: Path) -> Optional[float]:
    try:
        b = load_benchmark_equity(
            data_root, eq.index.min(), eq.index.max(), preferred=["SPY"]
        )
        if b is None or b.empty:
            return None
        j = pd.concat(
            [_eq_norm(eq).rename("s"), _eq_norm(b).rename("b")], axis=1, join="inner"
        ).dropna()
        if len(j) < 5:
            return None
        return float(j["s"].iloc[-1] / j["s"].iloc[0] - j["b"].iloc[-1] / j["b"].iloc[0])
    except Exception:
        return None


def run_window(
    strategy: str,
    *,
    first: int,
    last: int,
    data_root: Path,
    ticker_file: Path,
    universe_limit: int,
    min_train_rows: int,
) -> Dict[str, Any]:
    strat = get_strategy(strategy)
    if hasattr(strat, "universe_source_file"):
        strat.universe_source_file = str(ticker_file)
    base = strat.backtest_overrides() if hasattr(strat, "backtest_overrides") else {}
    merged = {**base, "commission": COMMISSION, "slippage": SLIPPAGE}

    def _ov() -> Dict[str, Any]:
        return dict(merged)

    orig = getattr(strat, "backtest_overrides", None)
    if orig is not None:
        strat.backtest_overrides = _ov  # type: ignore[method-assign]
    try:
        res = run_strategy_walk_forward(
            strat,
            data_root=data_root,
            ticker_file=ticker_file,
            universe_limit=int(universe_limit),
            first_oos_year=int(first),
            last_oos_year=int(last),
            min_train_rows=int(min_train_rows),
            preferred_index=["SPY", "QQQ"],
            base_bt=BacktestConfig(commission=COMMISSION, slippage=SLIPPAGE),
        )
    finally:
        if orig is not None:
            strat.backtest_overrides = orig  # type: ignore[method-assign]
    return res


def build_arms(
    strategies: Sequence[str],
    universe_arms: Sequence[Tuple[str, Path, int]],
) -> List[Arm]:
    arms: List[Arm] = []
    for ulab, upath, lim in universe_arms:
        if not upath.is_file():
            # fallback longhist100
            upath = ROOT / "universe_longhist100.txt"
            if not upath.is_file():
                continue
        for s in strategies:
            arms.append(
                Arm(
                    arm_id=f"{s}__{ulab}",
                    strategy=s,
                    universe_label=ulab,
                    ticker_file=upath,
                    universe_limit=lim,
                )
            )
    return arms


def save_progress(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def equity_chart_html(
    series_map: Dict[str, pd.Series],
    *,
    title: str,
) -> str:
    """Simple multi-series SVG equity chart (log-ish via normalize to 1)."""
    colors = ["#4cc9f0", "#f72585", "# copffb020", "#80ed99", "#c77dff", "#ff6b6b", "#90e0ef"]
    # fix typo in colors
    colors = ["#4cc9f0", "#f72585", "#ffb020", "#80ed99", "#c77dff", "#ff6b6b", "#90e0ef"]
    paths = []
    legend = []
    w, h, pad = 900, 360, 40
    for i, (name, eq) in enumerate(series_map.items()):
        s = _eq_norm(eq)
        if s.empty or len(s) < 2:
            continue
        s = s / float(s.iloc[0])
        xs = np.linspace(pad, w - pad, len(s))
        ymin, ymax = float(s.min()), float(s.max())
        if ymax <= ymin:
            ymax = ymin + 1e-6
        ys = pad + (1.0 - (s.to_numpy(dtype=float) - ymin) / (ymax - ymin)) * (h - 2 * pad)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        col = colors[i % len(colors)]
        paths.append(f"<polyline fill='none' stroke='{col}' stroke-width='2' points='{pts}' />")
        legend.append(f"<span style='color:{col}'>■</span> {name} &nbsp;")
    return (
        f"<div><h3>{title}</h3><div>{''.join(legend)}</div>"
        f"<svg width='{w}' height='{h}' style='background:#121a2f;border-radius:8px'>"
        f"{''.join(paths)}</svg></div>"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=16.0)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--screen-first", type=int, default=2010)
    ap.add_argument("--screen-last", type=int, default=2017)
    ap.add_argument("--confirm-first", type=int, default=2018)
    ap.add_argument("--confirm-last", type=int, default=2025)
    ap.add_argument("--min-train-rows", type=int, default=1500)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "redesign_v2",
    )
    ap.add_argument(
        "--strategies",
        type=str,
        default=",".join(STRATEGIES),
        help="Comma strategies (default full zoo)",
    )
    ap.add_argument("--max-arms", type=int, default=0, help="0=all")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    arms_dir = out / "arms"
    graphs_dir = out / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    arms_dir.mkdir(exist_ok=True)
    graphs_dir.mkdir(exist_ok=True)
    prog_path = out / "PROGRESS.json"

    strategies = [s.strip() for s in str(args.strategies).split(",") if s.strip()]
    arms = build_arms(strategies, UNIVERSE_ARMS)
    if int(args.max_arms) > 0:
        arms = arms[: int(args.max_arms)]

    t0 = time.time()
    deadline = t0 + float(args.hours) * 3600.0
    state: Dict[str, Any] = {
        "started": datetime.now(timezone.utc).isoformat(),
        "hours": float(args.hours),
        "n_arms": len(arms),
        "done": [],
        "rows": [],
        "stop_reason": None,
    }
    if prog_path.is_file():
        try:
            prev = json.loads(prog_path.read_text(encoding="utf-8"))
            state["done"] = list(prev.get("done") or [])
            state["rows"] = list(prev.get("rows") or [])
            print(f"Resume done={len(state['done'])}", flush=True)
        except Exception:
            pass

    done_set = set(state["done"])
    print(
        f"Redesign v2 mega arms={len(arms)} hours={args.hours} "
        f"screen={args.screen_first}-{args.screen_last} "
        f"confirm={args.confirm_first}-{args.confirm_last}",
        flush=True,
    )

    for arm in arms:
        if time.time() > deadline:
            state["stop_reason"] = "hours_exhausted"
            break
        if arm.arm_id in done_set:
            continue
        print(f"[arm] {arm.arm_id} …", flush=True)
        row: Dict[str, Any] = {
            "arm_id": arm.arm_id,
            "strategy": arm.strategy,
            "universe": arm.universe_label,
            "limit": arm.universe_limit,
        }
        adir = arms_dir / arm.arm_id.replace("/", "_")
        adir.mkdir(parents=True, exist_ok=True)
        try:
            # Screen
            rs = run_window(
                arm.strategy,
                first=int(args.screen_first),
                last=int(args.screen_last),
                data_root=Path(args.data_root),
                ticker_file=arm.ticker_file,
                universe_limit=arm.universe_limit,
                min_train_rows=int(args.min_train_rows),
            )
            eq_s = rs.get("equity")
            tr_s = rs.get("trades") if isinstance(rs.get("trades"), pd.DataFrame) else pd.DataFrame()
            m_s = _metrics(eq_s, tr_s) if isinstance(eq_s, pd.Series) else {"error": "empty"}
            g_s = _gates(m_s)
            row["screen"] = {**m_s, "gates": g_s}
            if isinstance(eq_s, pd.Series):
                _eq_norm(eq_s).to_csv(adir / "equity_screen.csv", header=["equity"])
            if isinstance(tr_s, pd.DataFrame) and not tr_s.empty:
                tr_s.to_csv(adir / "trades_screen.csv", index=False)

            # Confirm
            rc = run_window(
                arm.strategy,
                first=int(args.confirm_first),
                last=int(args.confirm_last),
                data_root=Path(args.data_root),
                ticker_file=arm.ticker_file,
                universe_limit=arm.universe_limit,
                min_train_rows=int(args.min_train_rows),
            )
            eq_c = rc.get("equity")
            tr_c = rc.get("trades") if isinstance(rc.get("trades"), pd.DataFrame) else pd.DataFrame()
            m_c = _metrics(eq_c, tr_c) if isinstance(eq_c, pd.Series) else {"error": "empty"}
            g_c = _gates(m_c)
            xs = _spy_excess(eq_c, Path(args.data_root)) if isinstance(eq_c, pd.Series) else None
            row["confirm"] = {**m_c, "gates": g_c, "excess_spy_total": xs}
            row["honest_score"] = _honest_score(m_c, xs)
            if isinstance(eq_c, pd.Series):
                _eq_norm(eq_c).to_csv(adir / "equity_confirm.csv", header=["equity"])
            if isinstance(tr_c, pd.DataFrame) and not tr_c.empty:
                tr_c.to_csv(adir / "trades_confirm.csv", index=False)
                # co-occurrence graph on confirm trades
                edges = trade_cooccurrence_graph(tr_c)
                hubs = hub_scores(edges)
                (graphs_dir / f"{arm.arm_id}_cooccur.html").write_text(
                    graph_to_html(
                        edges,
                        title=f"Trade co-occurrence confirm — {arm.arm_id}",
                        hubs=hubs,
                    ),
                    encoding="utf-8",
                )
                row["graph"] = graph_summary_dict(edges)

            # Full stitch
            if isinstance(eq_s, pd.Series) and isinstance(eq_c, pd.Series):
                eq_f = _stitch(eq_s, eq_c)
                m_f = _metrics(eq_f, None)
                g_f = _gates(m_f)
                row["full"] = {**m_f, "gates": g_f}
                eq_f.to_csv(adir / "equity_full.csv", header=["equity"])
            else:
                row["full"] = {"error": "missing_segment"}

            print(
                f"  screen_cagr={m_s.get('cagr')} confirm_cagr={m_c.get('cagr')} "
                f"confirm_pass={g_c.get('pass')} score={row.get('honest_score')}",
                flush=True,
            )
        except Exception as e:
            row["error"] = f"{type(e).__name__}:{e}"
            print(f"  ERROR {row['error']}", flush=True)

        (adir / "metrics.json").write_text(
            json.dumps(row, indent=2, default=str), encoding="utf-8"
        )
        state["rows"].append(row)
        state["done"].append(arm.arm_id)
        done_set.add(arm.arm_id)
        state["elapsed_sec"] = time.time() - t0
        save_progress(prog_path, state)

    if state.get("stop_reason") is None:
        state["stop_reason"] = "complete"
    state["finished"] = datetime.now(timezone.utc).isoformat()
    save_progress(prog_path, state)

    # Rank by confirm
    rows = list(state["rows"])
    ranked = sorted(
        [r for r in rows if not r.get("error")],
        key=lambda r: float(r.get("honest_score") or -999),
        reverse=True,
    )
    confirm_passers = [
        r
        for r in ranked
        if (r.get("confirm") or {}).get("gates", {}).get("pass")
    ]
    research_pass = [
        r
        for r in confirm_passers
        if (r.get("full") or {}).get("gates", {}).get("pass")
    ]

    # Dashboard equity top3 confirm scores
    chart_map: Dict[str, pd.Series] = {}
    for r in ranked[:5]:
        p = arms_dir / r["arm_id"].replace("/", "_") / "equity_confirm.csv"
        if p.is_file():
            eq = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
            chart_map[r["arm_id"]] = eq
    dash = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Redesign v2</title>",
        "<style>body{font-family:system-ui;background:#0b1020;color:#e8ecf5;padding:24px}",
        "table{border-collapse:collapse} td,th{border:1px solid #334;padding:6px 10px}",
        "a{color:#8ecae6}</style></head><body>",
        "<h1>Redesign v2 mega — screen/confirm</h1>",
        f"<p>Generated {datetime.now(timezone.utc).isoformat()} · "
        f"stop={state.get('stop_reason')} · arms_done={len(state['done'])}/{len(arms)}</p>",
        equity_chart_html(chart_map, title="Confirm equity (top honest_score)"),
        "<h2>Leaderboard (confirm honest_score)</h2>",
        "<table><tr><th>arm</th><th>confirm CAGR</th><th>MDD</th><th>pass</th>"
        "<th>screen CAGR</th><th>full CAGR</th><th>score</th></tr>",
    ]
    for r in ranked:
        c = r.get("confirm") or {}
        s = r.get("screen") or {}
        f = r.get("full") or {}
        dash.append(
            f"<tr><td><code>{r.get('arm_id')}</code></td>"
            f"<td>{100*float(c.get('cagr') or 0):.1f}%</td>"
            f"<td>{100*float(c.get('max_drawdown') or 0):.1f}%</td>"
            f"<td>{(c.get('gates') or {}).get('pass')}</td>"
            f"<td>{100*float(s.get('cagr') or 0):.1f}%</td>"
            f"<td>{100*float(f.get('cagr') or 0):.1f}%</td>"
            f"<td>{float(r.get('honest_score') or 0):.3f}</td></tr>"
        )
    dash.append("</table>")
    dash.append("<h2>Graphs</h2><ul>")
    for g in sorted(graphs_dir.glob("*.html")):
        dash.append(f"<li><a href='graphs/{g.name}'>{g.name}</a></li>")
    dash.append("</ul><p>Research only. Not financial advice.</p></body></html>")
    (out / "dashboard.html").write_text("\n".join(dash), encoding="utf-8")

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "stop_reason": state.get("stop_reason"),
        "n_done": len(state["done"]),
        "n_arms": len(arms),
        "confirm_passers": [r["arm_id"] for r in confirm_passers],
        "research_pass": [r["arm_id"] for r in research_pass],
        "ranked": ranked,
        "paper_freeze": "turbo_highvol_minalloc",
        "disclaimer": "Research only. Not financial advice.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# Redesign v2 mega — SUMMARY",
        "",
        "> **Research only.** Paper freeze unchanged.",
        "",
        f"- Stop: **{state.get('stop_reason')}** · done {len(state['done'])}/{len(arms)}",
        f"- Screen **{args.screen_first}–{args.screen_last}** · Confirm **{args.confirm_first}–{args.confirm_last}**",
        f"- Gates confirm: CAGR>{GATE_CAGR:.0%} · MDD≥{GATE_MDD:.0%} · n≥{GATE_TRADES}",
        f"- Confirm passers: `{', '.join(summary['confirm_passers']) or 'none'}`",
        f"- Research PASS (confirm∩full): `{', '.join(summary['research_pass']) or 'none'}`",
        "",
        "## Leaderboard",
        "",
        "| arm | confirm CAGR | MDD | pass | screen CAGR | full CAGR | score |",
        "|-----|--------------|-----|------|-------------|-----------|-------|",
    ]
    for r in ranked:
        c = r.get("confirm") or {}
        s = r.get("screen") or {}
        f = r.get("full") or {}
        lines.append(
            f"| `{r.get('arm_id')}` | {100*float(c.get('cagr') or 0):.1f}% | "
            f"{100*float(c.get('max_drawdown') or 0):.1f}% | "
            f"{(c.get('gates') or {}).get('pass')} | "
            f"{100*float(s.get('cagr') or 0):.1f}% | "
            f"{100*float(f.get('cagr') or 0):.1f}% | "
            f"{float(r.get('honest_score') or 0):.3f} |"
        )
    lines += [
        "",
        f"[Dashboard](dashboard.html)",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    dlines = [
        "# Redesign v2 — Decision",
        "",
        f"**Confirm passers:** {summary['confirm_passers'] or 'none'}",
        f"**Research PASS (confirm∩full):** {summary['research_pass'] or 'none'}",
        "",
        f"**Verdict:** {'PASS candidate(s) exist' if research_pass else 'FAIL — no research PASS'}",
        "",
        "**Paper freeze:** turbo_highvol_minalloc **unchanged**",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "DECISION.md").write_text("\n".join(dlines), encoding="utf-8")
    print(f"Wrote {out / 'SUMMARY.md'} research_pass={summary['research_pass']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
