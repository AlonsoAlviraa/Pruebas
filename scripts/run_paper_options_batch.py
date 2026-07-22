#!/usr/bin/env python3
"""Paper options multi-strategy batch (proxy BS on free OHLCV).

VIRTUAL capital only. Marks labeled proxy_bs — not exchange option fills.

Supports:
  - risk gates + CVaR / Calmar in SUMMARY (OPT-PR3)
  - multi-window runs (OPT-PR5)
  - synthetic crash stress (OPT-PR5)
  - optional Yahoo live chain snapshot for *today* only (OPT-PR6)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.cloud.free_data import SEED_DIR, build_cloud_feed
from paper_live.options.replay_options import book_delta_report, run_options_batch
from paper_live.options.risk import OptionsRiskConfig
from paper_live.options.strategies import OptionStrategySpec
from paper_live.options.stress import StressSpec, build_stressed_feed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("options_batch")

DEFAULT_WINDOWS: List[Tuple[str, str, str]] = [
    ("2022", "2022-01-03", "2022-12-30"),
    ("2023", "2023-01-03", "2023-12-29"),
    ("2024", "2024-01-02", "2024-12-31"),
    ("2025", "2025-01-02", "2025-12-31"),
    ("2026YTD", "2026-01-02", "2099-12-31"),
]


def _specs_from_zoo(path: Path) -> Tuple[List[OptionStrategySpec], float, OptionsRiskConfig]:
    z = json.loads(path.read_text(encoding="utf-8"))
    global_risk = OptionsRiskConfig.from_mapping(z.get("risk") or {})
    out: List[OptionStrategySpec] = []
    for s in z.get("strategies") or []:
        # Leave risk fields None when absent so _risk_for_spec can apply kind floors
        out.append(
            OptionStrategySpec(
                id=str(s["id"]),
                label=str(s.get("label") or s["id"]),
                kind=str(s["kind"]),
                underlying=str(s.get("underlying") or "SPY"),
                dte_days=int(s.get("dte_days") or 30),
                otm_pct=float(s.get("otm_pct") or 0.05),
                wing_otm_pct=float(s.get("wing_otm_pct") or 0.15),
                premium_mult=float(s.get("premium_mult") or 1.15),
                contracts=int(s.get("contracts") or 1),
                max_portfolio_dd=(
                    float(s["max_portfolio_dd"]) if s.get("max_portfolio_dd") is not None else None
                ),
                max_single_day_drop=(
                    float(s["max_single_day_drop"])
                    if s.get("max_single_day_drop") is not None
                    else None
                ),
                max_margin_fraction=(
                    float(s["max_margin_fraction"])
                    if s.get("max_margin_fraction") is not None
                    else None
                ),
                hard_kill_enabled=(
                    bool(s["hard_kill_enabled"]) if s.get("hard_kill_enabled") is not None else None
                ),
                meta=dict(s.get("meta") or {}),
                notes=str(s.get("notes") or ""),
            )
        )
    return out, float(z.get("capital0") or 100_000.0), global_risk


def _clamp_window(
    days: Sequence[date], start: date, end: date
) -> Tuple[date, date, bool]:
    """Return (clamped_start, clamped_end, clamped_flag)."""
    if not days:
        raise SystemExit("No feed days")
    s = next((d for d in days if d >= start), days[0])
    e = next((d for d in reversed(days) if d <= end), days[-1])
    if s > e:
        s, e = days[0], days[-1]
    clamped = s != start or e != end
    return s, e, clamped


def _bh(feed, ticker: str, start: date, end: date) -> Optional[float]:
    b0 = feed.bar(ticker, start)
    b1 = feed.bar(ticker, end)
    if b0 and b1 and float(b0.close) > 0:
        return float(b1.close) / float(b0.close) - 1.0
    return None


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2%}"


def _run_one_window(
    feed,
    specs: Sequence[OptionStrategySpec],
    *,
    start: date,
    end: date,
    capital0: float,
    risk: OptionsRiskConfig,
    data_label: str = "proxy_bs",
) -> Dict[str, Any]:
    spy_bh = _bh(feed, "SPY", start, end)
    qqq_bh = _bh(feed, "QQQ", start, end)
    results = run_options_batch(
        feed,
        specs,
        start=start,
        end=end,
        capital0=capital0,
        risk=risk,
        data_label=data_label,
        spy_bh=spy_bh,
        qqq_bh=qqq_bh,
    )
    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "capital0": capital0,
        "data_label": data_label,
        "benchmarks": {"spy_bh": spy_bh, "qqq_bh": qqq_bh},
        "strategies": [r.to_dict() for r in results],
        "results_obj": results,
    }


def _ranking_table(results: Sequence[Any], spy_bh: Optional[float]) -> List[str]:
    ranking = sorted(results, key=lambda r: r.total_return, reverse=True)
    lines = [
        "| Rank | ID | Kind | Return | MaxDD | CVaR5% | WorstMo | Calmar | Kill | DefRisk | Opens | DTE rolls | vsSPY |",
        "|------|-----|------|--------|-------|--------|---------|--------|------|---------|-------|-----------|-------|",
    ]
    for i, r in enumerate(ranking, 1):
        vs = r.vs_spy_bh
        if vs is None and spy_bh is not None:
            vs = r.total_return - spy_bh
        cal = f"{r.calmar_like:.2f}" if r.calmar_like is not None else "n/a"
        wm = getattr(r, "worst_month", None)
        n_open = getattr(r, "n_opens", r.n_rolls)
        n_dte = getattr(r, "n_dte_rolls", 0)
        lines.append(
            f"| {i} | `{r.strategy_id}` | {r.kind} | {_fmt_pct(r.total_return)} | "
            f"{_fmt_pct(r.max_dd)} | {_fmt_pct(r.cvar_5pct)} | {_fmt_pct(wm)} | {cal} | "
            f"{'YES' if r.hard_kill else 'no'} | {'yes' if r.defined_risk else 'no'} | "
            f"{n_open} | {n_dte} | {_fmt_pct(vs)} |"
        )
    return lines


def _write_summary_md(
    *,
    as_of: str,
    capital0: float,
    primary: Dict[str, Any],
    multi: Optional[List[Dict[str, Any]]] = None,
    stress: Optional[Dict[str, Any]] = None,
    live_chain: Optional[Dict[str, Any]] = None,
    risk: OptionsRiskConfig,
    data_sources: Dict[str, Any],
    specs: Sequence[OptionStrategySpec],
) -> str:
    win = primary["window"]
    ben = primary.get("benchmarks") or {}
    spy_bh = ben.get("spy_bh")
    qqq_bh = ben.get("qqq_bh")
    results = primary["results_obj"]
    lines = [
        f"# Paper options multi-strategy — `{as_of}`",
        "",
        f"**Window:** {win['start']} → {win['end']} · **Capital:** VIRTUAL ${capital0:,.0f}",
        "",
        f"**Data label:** `{primary.get('data_label', 'proxy_bs')}` "
        "(Black–Scholes on HV/IV proxy) — **NOT exchange fills**",
        "",
        f"**SPY B&H:** {_fmt_pct(spy_bh)} · **QQQ B&H:** {_fmt_pct(qqq_bh)}",
        "",
        "## Risk gates (short premium)",
        "",
        f"- Global defaults: max_portfolio_dd **{risk.max_portfolio_dd:.0%}**, "
        f"max_single_day_drop **{risk.max_single_day_drop:.0%}**, "
        f"max_margin_fraction **{risk.max_margin_fraction:.0%}**, "
        f"hard_kill **{risk.hard_kill_enabled}**",
        f"- CVaR alpha (report only): **{risk.cvar_alpha:.0%}** worst daily returns",
        "- **Per-strategy overrides apply** (see table below and each `risk_config` in JSON).",
        "",
        "| ID | max_dd | max_day | margin_frac | hard_kill |",
        "|----|--------|---------|-------------|-----------|",
    ]
    for r in results:
        rc = r.risk_config or {}
        lines.append(
            f"| `{r.strategy_id}` | {_fmt_pct(rc.get('max_portfolio_dd'))} | "
            f"{_fmt_pct(rc.get('max_single_day_drop'))} | "
            f"{_fmt_pct(rc.get('max_margin_fraction'))} | "
            f"{rc.get('hard_kill_enabled')} |"
        )
    lines += [
        "",
        "### Assumptions (model marks)",
        "",
        "- IV: **vix_surface** when VIX (±VIX3M) in feed; else **proxy_hv** = HV20 × premium_mult.",
        "- Marks via European Black–Scholes; mild put skew on surface; no dividends / borrow.",
        "- Bid haircut on sells (meta `bid_haircut`, default 5%) — not NBBO.",
        "- Premium seller mgmt: 50% credit TP, 2× credit SL, max 1 roll (meta-overridable).",
        "- Assignment proxy at expiry / deep ITM (labeled `assignment_proxy`).",
        "- Put credit spread margin = strike width × 100 × contracts (defined risk).",
        "- CSP collateral = short put strike × 100 × contracts.",
        "- Contract size is **strict** vs `max_margin_fraction` (no silent 1-lot fallback).",
        "- Single-day hard kill only on consecutive marked sessions "
        "(missing bars clear the day-drop baseline — no multi-day gap false kill).",
        "- **do not treat as live fill quality**.",
        "",
        "## Ranking (primary window)",
        "",
    ]
    lines += _ranking_table(results, spy_bh)
    book = book_delta_report(results)
    lines += [
        "",
        "## Book delta (approx BS)",
        "",
        f"- sum Δ_end = **{book.get('sum_delta_end')}** · mean Δ_end = **{book.get('mean_delta_end')}**",
        f"- label: `{book.get('label')}` — {book.get('note')}",
        "",
    ]

    if multi:
        lines += ["", "## Multi-window summary", ""]
        lines += [
            "| Window | Requested | Actual | Clamped | Best ID | Best Ret | Worst MaxDD | SPY B&H | QQQ B&H |",
            "|--------|-----------|--------|---------|---------|----------|-------------|---------|---------|",
        ]
        for block in multi:
            wname = block.get("name") or "?"
            w = block["window"]
            req = block.get("requested") or {}
            clamped = block.get("clamped", False)
            strat = block.get("strategies") or []
            if strat:
                best = max(strat, key=lambda s: s.get("total_return") or -1e9)
                worst_dd = min(strat, key=lambda s: s.get("max_dd") or 0)
                best_id = best.get("strategy_id")
                best_ret = best.get("total_return")
                wdd = worst_dd.get("max_dd")
            else:
                best_id, best_ret, wdd = "n/a", None, None
            bmk = block.get("benchmarks") or {}
            req_s = f"{req.get('start', '?')}→{req.get('end', '?')}"
            lines.append(
                f"| {wname} | {req_s} | {w['start']}→{w['end']} | "
                f"{'yes' if clamped else 'no'} | `{best_id}` | "
                f"{_fmt_pct(best_ret)} | {_fmt_pct(wdd)} | "
                f"{_fmt_pct(bmk.get('spy_bh'))} | {_fmt_pct(bmk.get('qqq_bh'))} |"
            )
        lines += [
            "",
            "Per-window strategy detail is in `multi_window.json` "
            "(`requested_*`, `clamped` fields included).",
        ]

    if stress:
        lines += [
            "",
            "## Synthetic crash stress",
            "",
            f"**Label:** `{stress.get('meta', {}).get('label', 'stress')}` · "
            f"**data_label:** `proxy_bs_stress`",
            "",
            f"Shock: {stress.get('meta', {}).get('shock_pct')} over "
            f"{stress.get('meta', {}).get('n_days')} days "
            f"({stress.get('meta', {}).get('crash_start')} → "
            f"{stress.get('meta', {}).get('crash_end')})",
            f"VIX spike mult: {stress.get('meta', {}).get('vix_spike_mult', 'n/a')} · "
            f"spiked: {stress.get('meta', {}).get('vix_tickers_spiked', [])}",
            "",
        ]
        s_results = stress.get("results_obj") or []
        if s_results:
            lines += _ranking_table(s_results, (stress.get("benchmarks") or {}).get("spy_bh"))
        lines += [
            "",
            "Stress injects a forced ~−30% underlying path **and** spikes VIX/VIX3M panels "
            "for the surface; marks stay model BS. "
            "Depression persists after crash_end (no recovery jump on the stressed feed).",
        ]

    if live_chain is not None:
        lines += [
            "",
            "## Live chain snapshot (today only, OPT-PR6)",
            "",
            f"**data_label:** `{live_chain.get('data_label', 'n/a')}`",
            "",
        ]
        for und, summ in (live_chain.get("summaries") or {}).items():
            lines.append(
                f"- **{und}** OK: spot={summ.get('spot')} nearest {summ.get('side')} "
                f"K={summ.get('nearest_strike')} mid={summ.get('nearest_mid')} "
                f"IV={summ.get('nearest_iv')} exp={summ.get('expiry')}"
            )
        fails = live_chain.get("failures") or []
        for f in fails:
            lines.append(f"- **FAIL** `{f}` — no synthetic chain invented.")
        if not (live_chain.get("summaries") or {}) and not fails:
            lines.append(
                f"- Chain fetch failed: `{live_chain.get('error')}` — "
                "**no synthetic chain invented**."
            )
        lines += [
            "",
            "Live chain does **not** rewrite historical `proxy_bs` marks.",
        ]

    lines += [
        "",
        "## Design / papers",
        "",
        "See `docs/design/2026-07-22_paper_options_strategies.md` (VRP, covered call, CSP, spreads).",
        "",
        f"**Data sources (OHLCV):** `{json.dumps(data_sources, default=str)[:500]}`",
        "",
        "---",
        f"_Generated {datetime.now(timezone.utc).isoformat()} · paper only · proxy_bs · VIRTUAL capital_",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Paper options batch (proxy BS). "
            "Default zoo: paper_live/cloud/zoo_options.json. "
            "TA-gated zoo: paper_live/cloud/zoo_options_ta.json "
            "(meta require_uptrend / require_low_atr / volume / RSI gates)."
        )
    )
    ap.add_argument("--out", default="reports/paper_options")
    ap.add_argument(
        "--zoo",
        default="paper_live/cloud/zoo_options.json",
        help=(
            "Options strategy zoo. "
            "Examples: paper_live/cloud/zoo_options.json, "
            "paper_live/cloud/zoo_options_ta.json, "
            "paper_live/cloud/zoo_options_50.json"
        ),
    )
    ap.add_argument("--start", default="2025-10-29")
    ap.add_argument("--end", default=None)
    ap.add_argument(
        "--multi-window",
        action="store_true",
        help="Run default multi-year windows (2022–2026YTD) in addition to --start/--end",
    )
    ap.add_argument(
        "--windows",
        default=None,
        help="Custom windows name:start:end,name2:start2:end2",
    )
    ap.add_argument(
        "--stress",
        action="store_true",
        help="Run synthetic ~-30%% crash stress on primary window",
    )
    ap.add_argument(
        "--stress-shock",
        type=float,
        default=-0.30,
        help="Total underlying shock for stress (default -0.30)",
    )
    ap.add_argument(
        "--live-chain",
        action="store_true",
        help="Fetch Yahoo options chain for today (SPY/QQQ); never fakes on failure",
    )
    ap.add_argument(
        "--live-chain-tickers",
        default="SPY,QQQ",
        help="Comma-separated underlyings for --live-chain",
    )
    args = ap.parse_args()

    zoo_path = Path(args.zoo)
    specs, capital0, risk = _specs_from_zoo(zoo_path)
    tickers = sorted(
        {sp.underlying.upper() for sp in specs} | {"SPY", "QQQ", "VIX", "VIX3M"}
    )
    feed, sources = build_cloud_feed(
        tickers,
        seed_dir=SEED_DIR,
        lookback_calendar_days=1600,
        require_real=True,
        min_real_tickers=2,
    )
    days = feed.days
    if not days:
        raise SystemExit("No feed days")

    req_start = __import__("pandas").Timestamp(args.start).date()
    req_end = (
        __import__("pandas").Timestamp(args.end).date()
        if args.end
        else days[-1]
    )
    start, end, primary_clamped = _clamp_window(days, req_start, req_end)
    if primary_clamped:
        logger.warning(
            "Primary window clamped: requested %s→%s actual %s→%s",
            req_start,
            req_end,
            start,
            end,
        )

    primary = _run_one_window(
        feed, specs, start=start, end=end, capital0=capital0, risk=risk
    )
    primary["requested"] = {
        "start": req_start.isoformat(),
        "end": req_end.isoformat(),
    }
    primary["clamped"] = primary_clamped

    multi_payload: List[Dict[str, Any]] = []
    windows: List[Tuple[str, str, str]] = []
    if args.windows:
        for part in args.windows.split(","):
            bits = part.strip().split(":")
            if len(bits) != 3:
                raise SystemExit(f"Bad --windows segment: {part}")
            windows.append((bits[0], bits[1], bits[2]))
    elif args.multi_window:
        windows = list(DEFAULT_WINDOWS)

    for name, ws, we in windows:
        w_req_s = __import__("pandas").Timestamp(ws).date()
        w_req_e = __import__("pandas").Timestamp(we).date()
        w_start, w_end, w_clamped = _clamp_window(days, w_req_s, w_req_e)
        if w_clamped:
            logger.warning(
                "Window %s clamped: requested %s→%s actual %s→%s",
                name,
                w_req_s,
                w_req_e,
                w_start,
                w_end,
            )
        block = _run_one_window(
            feed, specs, start=w_start, end=w_end, capital0=capital0, risk=risk
        )
        serial = {
            "name": name,
            "requested": {
                "start": w_req_s.isoformat(),
                "end": w_req_e.isoformat(),
            },
            "window": block["window"],
            "clamped": w_clamped,
            "capital0": capital0,
            "data_label": "proxy_bs",
            "benchmarks": block["benchmarks"],
            "strategies": block["strategies"],
        }
        multi_payload.append(serial)

    stress_block: Optional[Dict[str, Any]] = None
    if args.stress:
        st = StressSpec(shock_pct=float(args.stress_shock))
        try:
            sfeed, smeta = build_stressed_feed(
                feed, start=start, end=end, tickers=tickers, stress=st
            )
            stress_block = _run_one_window(
                sfeed,
                specs,
                start=start,
                end=end,
                capital0=capital0,
                risk=risk,
                data_label="proxy_bs_stress",
            )
            stress_block["meta"] = smeta
        except Exception as e:
            logger.warning("Stress run failed: %s", e)
            stress_block = {
                "meta": {"error": str(e), "label": st.label},
                "window": primary["window"],
                "benchmarks": {},
                "strategies": [],
                "results_obj": [],
            }

    live_chain_payload: Optional[Dict[str, Any]] = None
    if args.live_chain:
        from paper_live.options.yahoo_chain import (
            fetch_yahoo_option_chain,
            summarize_chain_vs_proxy,
        )

        unds = [x.strip().upper() for x in args.live_chain_tickers.split(",") if x.strip()]
        snaps: Dict[str, Any] = {}
        summaries: Dict[str, Any] = {}
        failures: List[str] = []
        n_ok = 0
        for und in unds:
            snap = fetch_yahoo_option_chain(und, raise_on_error=False)
            snaps[und] = snap.to_dict()
            if snap.ok:
                n_ok += 1
                summaries[und] = summarize_chain_vs_proxy(snap, otm_pct=0.05, side="put")
            else:
                failures.append(f"{und}: {snap.error}")
        n_tot = len(unds)
        if n_ok == 0:
            top_label = "yahoo_chain_failed"
        elif n_ok < n_tot:
            top_label = "yahoo_chain_partial"
        else:
            top_label = "yahoo_chain"
        live_chain_payload = {
            "ok": n_ok > 0,
            "partial": 0 < n_ok < n_tot,
            "data_label": top_label,
            "summaries": summaries,
            "failures": failures,
            "chains": snaps,
            "error": None if n_ok == n_tot else "; ".join(failures) if failures else None,
            "note": "Today-only validation; historical marks remain proxy_bs.",
        }

    out_root = Path(args.out)
    latest = out_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    primary_book = book_delta_report(primary["results_obj"])
    # Prefer run-level labels from results when present
    run_labels = sorted({r.data_label for r in primary["results_obj"]})
    primary_label = run_labels[0] if len(run_labels) == 1 else (
        "mixed" if run_labels else "proxy_bs"
    )
    payload: Dict[str, Any] = {
        "as_of": as_of,
        "window": primary["window"],
        "requested": primary.get("requested"),
        "clamped": primary.get("clamped"),
        "capital0": capital0,
        "data_label": primary_label,
        "data_sources": sources,
        "risk": risk.to_dict(),
        "benchmarks": primary["benchmarks"],
        "strategies": primary["strategies"],
        "book_delta": primary_book,
        "disclaimer": (
            "Model BS marks (vix_surface or proxy_hv IV) — not real option fills. "
            "Virtual capital only. Bid haircut + assignment_proxy + seller mgmt applied. "
            "CVaR/Calmar from daily equity marks. Margin sizing is strict (no 1-lot fallback)."
        ),
    }
    if multi_payload:
        payload["multi_window"] = multi_payload
    if stress_block is not None:
        payload["stress"] = {
            "meta": stress_block.get("meta"),
            "window": stress_block.get("window"),
            "data_label": "proxy_bs_stress",
            "benchmarks": stress_block.get("benchmarks"),
            "strategies": stress_block.get("strategies"),
        }
    if live_chain_payload is not None:
        payload["live_chain"] = {
            k: v for k, v in live_chain_payload.items() if k != "chains"
        }
        (latest / "live_chain.json").write_text(
            json.dumps(live_chain_payload, indent=2, default=str), encoding="utf-8"
        )

    md = _write_summary_md(
        as_of=as_of,
        capital0=capital0,
        primary=primary,
        multi=multi_payload or None,
        stress=stress_block,
        live_chain=live_chain_payload,
        risk=risk,
        data_sources=sources,
        specs=specs,
    )
    (latest / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (latest / "SUMMARY.md").write_text(md, encoding="utf-8")
    if multi_payload:
        (latest / "multi_window.json").write_text(
            json.dumps(multi_payload, indent=2, default=str), encoding="utf-8"
        )
    if stress_block is not None:
        (latest / "stress.json").write_text(
            json.dumps(
                {
                    "meta": stress_block.get("meta"),
                    "window": stress_block.get("window"),
                    "data_label": "proxy_bs_stress",
                    "benchmarks": stress_block.get("benchmarks"),
                    "strategies": stress_block.get("strategies"),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    print(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {latest / 'SUMMARY.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
