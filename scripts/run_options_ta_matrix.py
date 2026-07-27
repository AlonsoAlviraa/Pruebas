#!/usr/bin/env python3
"""OPT_TA multi-window + stress matrix (P0 blind-spot closeout).

Runs ``zoo_options_ta.json`` (and optional names zoo) across:
  - 2022 bear, 2023, 2024, 2025 study window
  - synthetic −30% stress on primary window

Writes pack under ``reports/paper_options_ta_matrix/`` with SUMMARY tables.
Uses free Yahoo OHLCV + VIX/VIX3M when available (labels honest).

VIRTUAL capital only.
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
from paper_live.options.book import book_delta_report_beta, build_sleeve_portfolio
from paper_live.options.replay_options import book_delta_report, run_options_batch
from paper_live.options.risk import OptionsRiskConfig
from paper_live.options.scorecard import write_scorecard
from paper_live.options.strategies import OptionStrategySpec
from paper_live.options.stress import StressSpec, build_stressed_feed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("options_ta_matrix")

# Default study windows (blind-spot P0)
MATRIX_WINDOWS: List[Tuple[str, str, str]] = [
    ("2022_bear", "2022-01-03", "2022-12-30"),
    ("2023", "2023-01-03", "2023-12-29"),
    ("2024", "2024-01-02", "2024-12-31"),
    ("2025_study", "2025-10-29", "2099-12-31"),
]

# Always pull vol surface proxies + common underlyings
VOL_TICKERS = ("VIX", "VIX3M")
BENCH_TICKERS = ("SPY", "QQQ")


def _specs_from_zoo(path: Path) -> Tuple[List[OptionStrategySpec], float, OptionsRiskConfig]:
    z = json.loads(path.read_text(encoding="utf-8"))
    global_risk = OptionsRiskConfig.from_mapping(z.get("risk") or {})
    out: List[OptionStrategySpec] = []
    for s in z.get("strategies") or []:
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


def _run_window(
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
    iv_sources = sorted({getattr(r, "iv_source", "") or "" for r in results})
    labels = sorted({r.data_label for r in results})
    book = book_delta_report(results)
    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "capital0": capital0,
        "data_label": labels[0] if len(labels) == 1 else "mixed",
        "iv_sources": iv_sources,
        "benchmarks": {"spy_bh": spy_bh, "qqq_bh": qqq_bh},
        "strategies": [r.to_dict() for r in results],
        "results_obj": results,
        "book_delta": book,
    }


def _ranking_table(results: Sequence[Any], spy_bh: Optional[float]) -> List[str]:
    ranking = sorted(results, key=lambda r: r.total_return, reverse=True)
    lines = [
        "| Rank | ID | Kind | Und | Return | MaxDD | CVaR5% | Opens | DTE rolls | TP/SL/TE | Δend | vsSPY | Kill |",
        "|------|-----|------|-----|--------|-------|--------|-------|-----------|----------|------|-------|------|",
    ]
    for i, r in enumerate(ranking, 1):
        vs = r.vs_spy_bh
        if vs is None and spy_bh is not None:
            vs = r.total_return - spy_bh
        dlt = getattr(r, "approx_delta_end", None)
        dlt_s = f"{dlt:.0f}" if dlt is not None else "n/a"
        tpsl = (
            f"{getattr(r, 'n_tp', 0)}/{getattr(r, 'n_sl', 0)}/"
            f"{getattr(r, 'n_time_exit', 0)}"
        )
        n_open = getattr(r, "n_opens", r.n_rolls)
        n_dte = getattr(r, "n_dte_rolls", 0)
        lines.append(
            f"| {i} | `{r.strategy_id}` | {r.kind} | {r.underlying} | "
            f"{_fmt_pct(r.total_return)} | {_fmt_pct(r.max_dd)} | {_fmt_pct(r.cvar_5pct)} | "
            f"{n_open} | {n_dte} | {tpsl} | {dlt_s} | {_fmt_pct(vs)} | "
            f"{'YES' if r.hard_kill else 'no'} |"
        )
    return lines


def _window_md(name: str, block: Dict[str, Any]) -> str:
    win = block["window"]
    ben = block.get("benchmarks") or {}
    results = block.get("results_obj") or []
    clamped = block.get("clamped", False)
    req = block.get("requested") or {}
    lines = [
        f"### Window `{name}`",
        "",
        f"- **Actual:** {win['start']} → {win['end']}",
        f"- **Requested:** {req.get('start', '?')} → {req.get('end', '?')}",
        f"- **Clamped:** {'**yes** (history short — labeled honestly)' if clamped else 'no'}",
        f"- **data_label / IV:** `{block.get('data_label')}` · sources={block.get('iv_sources')}",
        f"- **SPY B&H:** {_fmt_pct(ben.get('spy_bh'))} · **QQQ B&H:** {_fmt_pct(ben.get('qqq_bh'))}",
        "",
    ]
    if results:
        lines += _ranking_table(results, ben.get("spy_bh"))
    else:
        lines.append("_No strategy results._")
    bd = block.get("book_delta") or {}
    lines += [
        "",
        f"**Book delta (approx):** sum_end={bd.get('sum_delta_end')} · "
        f"mean_end={bd.get('mean_delta_end')} · label=`{bd.get('label', 'n/a')}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="OPT_TA multi-window + stress matrix")
    ap.add_argument(
        "--zoo",
        default="paper_live/cloud/zoo_options_ta.json",
        help="Primary TA zoo",
    )
    ap.add_argument(
        "--names-zoo",
        default="paper_live/cloud/zoo_options_ta_names.json",
        help="Single-name zoo (default: zoo_options_ta_names.json). Pass empty string to skip.",
    )
    ap.add_argument(
        "--no-names",
        action="store_true",
        help="Skip single-name zoo even if default path exists",
    )
    ap.add_argument(
        "--no-scorecard",
        action="store_true",
        help="Skip promote/watch/kill scorecard write",
    )
    ap.add_argument(
        "--chain-diag",
        action="store_true",
        help="Optional: Yahoo chain mid vs model BS today (network; diagnostic only)",
    )
    ap.add_argument(
        "--out",
        default="reports/paper_options_ta_matrix",
    )
    ap.add_argument(
        "--primary-start",
        default="2025-10-29",
        help="Primary study window start (also stress base)",
    )
    ap.add_argument("--primary-end", default=None)
    ap.add_argument(
        "--stress-shock",
        type=float,
        default=-0.30,
        help="Synthetic crash on primary window (default -0.30)",
    )
    ap.add_argument(
        "--no-stress",
        action="store_true",
        help="Skip synthetic stress leg",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=1600,
        help="Calendar lookback for free data (need 2022+)",
    )
    ap.add_argument(
        "--min-real-tickers",
        type=int,
        default=2,
        help="Min real tickers for build_cloud_feed",
    )
    args = ap.parse_args()

    zoo_path = Path(args.zoo)
    specs, capital0, risk = _specs_from_zoo(zoo_path)
    names_zoo_arg = None if args.no_names else (args.names_zoo or None)
    if names_zoo_arg in ("", "none", "None"):
        names_zoo_arg = None
    if names_zoo_arg:
        npath = Path(names_zoo_arg)
        if npath.is_file():
            n_specs, _, _ = _specs_from_zoo(npath)
            # append non-duplicate ids
            seen = {s.id for s in specs}
            for s in n_specs:
                if s.id not in seen:
                    specs.append(s)
                    seen.add(s.id)
        else:
            logger.warning("names-zoo not found: %s — skipping", npath)
            names_zoo_arg = None

    unds = sorted({sp.underlying.upper() for sp in specs} | set(BENCH_TICKERS) | set(VOL_TICKERS))
    logger.info("Building feed tickers=%s lookback=%d", unds, args.lookback_days)
    feed, sources = build_cloud_feed(
        unds,
        seed_dir=SEED_DIR,
        lookback_calendar_days=int(args.lookback_days),
        require_real=True,
        min_real_tickers=int(args.min_real_tickers),
    )
    days = feed.days
    if not days:
        raise SystemExit("No feed days")

    # If VIX missing from free fetch, label will fall back to proxy_hv automatically
    has_vix = any(str(sources.get(t, "")).startswith("yahoo") or sources.get(t) not in (
        None,
        "missing",
        "synthetic",
    ) for t in ("VIX", "^VIX", "VIX3M", "^VIX3M") if t in sources)
    if "VIX" not in feed._raw and "^VIX" not in feed._raw:  # noqa: SLF001 — diagnostic only
        logger.warning(
            "VIX not in feed panels — IV marks will label proxy_hv. sources=%s",
            {k: sources.get(k) for k in unds},
        )
        has_vix = False
    else:
        has_vix = True

    import pandas as pd

    windows = list(MATRIX_WINDOWS)
    # Ensure primary study is present
    p_start = pd.Timestamp(args.primary_start).date()
    p_end = pd.Timestamp(args.primary_end).date() if args.primary_end else days[-1]

    multi_blocks: List[Dict[str, Any]] = []
    for name, ws, we in windows:
        w_req_s = pd.Timestamp(ws).date()
        w_req_e = pd.Timestamp(we).date() if we != "2099-12-31" else days[-1]
        if name == "2025_study":
            w_req_s = p_start
            w_req_e = p_end
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
        block = _run_window(
            feed, specs, start=w_start, end=w_end, capital0=capital0, risk=risk
        )
        block["name"] = name
        block["requested"] = {
            "start": w_req_s.isoformat(),
            "end": w_req_e.isoformat(),
        }
        block["clamped"] = w_clamped
        multi_blocks.append(block)

    # Primary = 2025_study if present else last
    primary = next((b for b in multi_blocks if b.get("name") == "2025_study"), multi_blocks[-1])

    stress_block: Optional[Dict[str, Any]] = None
    if not args.no_stress:
        st = StressSpec(shock_pct=float(args.stress_shock))
        try:
            stress_tickers = sorted(
                {sp.underlying.upper() for sp in specs} | set(BENCH_TICKERS)
            )
            sfeed, smeta = build_stressed_feed(
                feed,
                start=date.fromisoformat(primary["window"]["start"]),
                end=date.fromisoformat(primary["window"]["end"]),
                tickers=stress_tickers,
                stress=st,
            )
            stress_block = _run_window(
                sfeed,
                specs,
                start=date.fromisoformat(primary["window"]["start"]),
                end=date.fromisoformat(primary["window"]["end"]),
                capital0=capital0,
                risk=risk,
                data_label="proxy_bs_stress",
            )
            stress_block["meta"] = smeta
            stress_block["name"] = "stress_primary"
            stress_block["clamped"] = primary.get("clamped", False)
            stress_block["requested"] = primary.get("requested")
        except Exception as e:
            logger.warning("Stress run failed: %s", e)
            stress_block = {
                "name": "stress_primary",
                "meta": {"error": str(e), "label": st.label},
                "window": primary["window"],
                "benchmarks": {},
                "strategies": [],
                "results_obj": [],
                "book_delta": {},
            }

    out_root = Path(args.out)
    latest = out_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Serialise multi (drop results_obj)
    multi_serial = []
    for b in multi_blocks:
        multi_serial.append(
            {
                "name": b.get("name"),
                "requested": b.get("requested"),
                "window": b.get("window"),
                "clamped": b.get("clamped"),
                "data_label": b.get("data_label"),
                "iv_sources": b.get("iv_sources"),
                "benchmarks": b.get("benchmarks"),
                "book_delta": b.get("book_delta"),
                "strategies": b.get("strategies"),
            }
        )

    payload: Dict[str, Any] = {
        "as_of": as_of,
        "zoo": str(zoo_path),
        "names_zoo": names_zoo_arg,
        "capital0": capital0,
        "has_vix": has_vix,
        "data_sources": sources,
        "risk": risk.to_dict(),
        "windows": multi_serial,
        "primary": {
            "name": primary.get("name"),
            "window": primary.get("window"),
            "clamped": primary.get("clamped"),
            "data_label": primary.get("data_label"),
            "benchmarks": primary.get("benchmarks"),
            "book_delta": primary.get("book_delta"),
            "strategies": primary.get("strategies"),
        },
        "disclaimer": (
            "Model BS marks; IV from vix_surface when VIX in feed else proxy_hv. "
            "Bid haircut + assignment_proxy + management rules applied. "
            "VIRTUAL capital. Not exchange fills."
        ),
    }
    if stress_block is not None:
        payload["stress"] = {
            "name": stress_block.get("name"),
            "meta": stress_block.get("meta"),
            "window": stress_block.get("window"),
            "data_label": stress_block.get("data_label"),
            "benchmarks": stress_block.get("benchmarks"),
            "book_delta": stress_block.get("book_delta"),
            "strategies": stress_block.get("strategies"),
        }

    # SUMMARY.md
    lines = [
        f"# OPT_TA multi-window matrix — `{as_of}`",
        "",
        f"**Zoo:** `{zoo_path}`"
        + (f" + names `{names_zoo_arg}`" if names_zoo_arg else ""),
        f"**Capital:** VIRTUAL ${capital0:,.0f}",
        f"**VIX in feed:** {has_vix}",
        f"**N strategies:** {len(specs)}",
        "",
        "## Data quality labels",
        "",
        "- Marks: Black–Scholes (`proxy_bs` math)",
        "- IV: `vix_surface` when VIX (±VIX3M) available, else `proxy_hv` (HV×premium_mult)",
        "- Fills: bid haircut on sells (default 5%); not NBBO",
        "- Assignment: `assignment_proxy` at expiry / deep ITM",
        "- Management: 50% credit TP, 2× credit SL, max 1 **DTE roll** per structure (meta-overridable)",
        "- Counters: **Opens** = every successful entry; **DTE rolls** = roll-only (capped by max_rolls)",
        "- Stress: equity path shock **and** VIX/VIX3M spike (not spot-only)",
        "",
        "## Windows overview",
        "",
        "| Window | Requested | Actual | Clamped | Best ID | Best Ret | Worst MaxDD | SPY B&H | Book Δend |",
        "|--------|-----------|--------|---------|---------|----------|-------------|---------|-----------|",
    ]
    for b in multi_blocks:
        name = b.get("name") or "?"
        w = b["window"]
        req = b.get("requested") or {}
        clamped = b.get("clamped", False)
        strat = b.get("strategies") or []
        if strat:
            best = max(strat, key=lambda s: s.get("total_return") or -1e9)
            worst_dd = min(strat, key=lambda s: s.get("max_dd") or 0)
            best_id = best.get("strategy_id")
            best_ret = best.get("total_return")
            wdd = worst_dd.get("max_dd")
        else:
            best_id, best_ret, wdd = "n/a", None, None
        bmk = b.get("benchmarks") or {}
        bd = b.get("book_delta") or {}
        req_s = f"{req.get('start', '?')}→{req.get('end', '?')}"
        lines.append(
            f"| {name} | {req_s} | {w['start']}→{w['end']} | "
            f"{'yes' if clamped else 'no'} | `{best_id}` | "
            f"{_fmt_pct(best_ret)} | {_fmt_pct(wdd)} | "
            f"{_fmt_pct(bmk.get('spy_bh'))} | {bd.get('sum_delta_end', 'n/a')} |"
        )

    lines += ["", "## Per-window detail", ""]
    for b in multi_blocks:
        lines.append(_window_md(str(b.get("name")), b))

    if stress_block is not None:
        lines += [
            "## Synthetic crash stress (primary window)",
            "",
            f"**Label:** `{stress_block.get('meta', {}).get('label', 'stress')}` · "
            f"shock={stress_block.get('meta', {}).get('shock_pct')}",
            "",
            _window_md("stress_primary", stress_block),
        ]

    lines += [
        "---",
        f"_Generated {datetime.now(timezone.utc).isoformat()} · paper only · VIRTUAL_",
        f"**Data sources:** `{json.dumps(sources, default=str)[:800]}`",
        "",
    ]
    md = "\n".join(lines)

    (latest / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (latest / "SUMMARY.md").write_text(md, encoding="utf-8")
    (latest / "multi_window.json").write_text(
        json.dumps(multi_serial, indent=2, default=str), encoding="utf-8"
    )
    if stress_block is not None:
        (latest / "stress.json").write_text(
            json.dumps(
                {
                    k: v
                    for k, v in payload.get("stress", {}).items()
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    # Per-window markdown stubs
    for b in multi_blocks:
        name = b.get("name") or "window"
        (latest / f"window_{name}.md").write_text(_window_md(str(name), b), encoding="utf-8")

    # Beta-weighted book + sleeve portfolio on primary window
    try:
        primary_results = primary.get("results_obj") or []
        p_end_d = date.fromisoformat(primary["window"]["end"])
        if primary_results:
            bw = book_delta_report_beta(primary_results, feed, p_end_d)
            sleeve = build_sleeve_portfolio(primary_results, capital0=capital0)
            payload["primary"]["book_delta_beta"] = bw
            payload["primary"]["sleeve_portfolio"] = {
                k: v
                for k, v in sleeve.to_dict().items()
                if k != "equity_curve"
            }
            # attach full curve separately (can be large)
            (latest / "sleeve_equity.json").write_text(
                json.dumps(sleeve.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
            lines_extra = [
                "",
                "## Book risk (primary window)",
                "",
                f"- **Raw Δ sum:** {bw.get('sum_raw_delta_end')}",
                f"- **Beta-weighted Δ sum:** {bw.get('sum_beta_weighted_delta')} "
                f"(label=`{bw.get('label')}`)",
                f"- **Sleeve portfolio:** return={_fmt_pct(sleeve.total_return)} "
                f"maxDD={_fmt_pct(sleeve.max_dd)} weights={sleeve.weights} "
                f"members={sleeve.members}",
                "",
            ]
            md = md.rstrip() + "\n" + "\n".join(lines_extra) + "\n"
            (latest / "SUMMARY.md").write_text(md, encoding="utf-8")
            (latest / "summary.json").write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
    except Exception as e:
        logger.warning("Book beta / sleeve failed: %s", e)

    # Optional chain diagnostic (today only — never rewrites history)
    if args.chain_diag:
        try:
            from paper_live.options.chain_diag import diagnose_chain_vs_model
            from paper_live.options.vol_surface import resolve_vix_level, VIX_TICKERS, VIX3M_TICKERS

            last = days[-1]
            vix = resolve_vix_level(feed, last, aliases=VIX_TICKERS)
            vix3m = resolve_vix_level(feed, last, aliases=VIX3M_TICKERS)
            diag = diagnose_chain_vs_model(
                ("SPY", "QQQ", "AAPL"),
                vix=vix,
                vix3m=vix3m,
            )
            payload["chain_diag"] = diag
            (latest / "chain_diag.json").write_text(
                json.dumps(diag, indent=2, default=str), encoding="utf-8"
            )
            (latest / "summary.json").write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            logger.info("chain_diag ok=%s label=%s", diag.get("ok"), diag.get("label"))
        except Exception as e:
            logger.warning("chain_diag failed: %s", e)
            (latest / "chain_diag.json").write_text(
                json.dumps(
                    {"ok": False, "label": "yahoo_chain_failed", "error": str(e)},
                    indent=2,
                ),
                encoding="utf-8",
            )

    # Scorecard promote / watch / kill
    if not args.no_scorecard:
        try:
            sc_path = out_root / "SCORECARD.md"
            write_scorecard(
                latest / "summary.json",
                out_md=sc_path,
                out_json=out_root / "SCORECARD.json",
                config_path=ROOT / "paper_live" / "cloud" / "scorecard_options_config.json",
            )
            logger.info("Scorecard → %s", sc_path)
        except Exception as e:
            logger.warning("Scorecard failed: %s", e)

    logger.info("Wrote pack to %s", latest)
    print(f"Wrote {latest / 'SUMMARY.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
