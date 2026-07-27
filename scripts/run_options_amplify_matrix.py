#!/usr/bin/env python3
"""Multi-year options AMPLIFY matrix (debit/PMCC) with mean multi-year ranking.

Uses free Yahoo OHLCV + VIX surface BS marks (proxy_bs|vix_surface).
Optionally uses data pack cache from download_options_research_data.py.

VIRTUAL only. Not OPRA historical fills.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.cloud.free_data import SEED_DIR, build_cloud_feed
from paper_live.options.replay_options import run_options_batch
from paper_live.options.risk import OptionsRiskConfig
from paper_live.options.strategies import OptionStrategySpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("options_amplify")

WINDOWS: List[Tuple[str, str, str]] = [
    ("2022", "2022-01-03", "2022-12-30"),
    ("2023", "2023-01-03", "2023-12-29"),
    ("2024", "2024-01-02", "2024-12-31"),
    ("2025_study", "2025-01-02", "2099-12-31"),
]


def _specs(path: Path) -> Tuple[List[OptionStrategySpec], float, OptionsRiskConfig]:
    z = json.loads(path.read_text(encoding="utf-8"))
    risk = OptionsRiskConfig.from_mapping(z.get("risk") or {})
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
                wing_otm_pct=float(s.get("wing_otm_pct") or 0.12),
                premium_mult=float(s.get("premium_mult") or 1.15),
                contracts=int(s.get("contracts") or 3),
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
    return out, float(z.get("capital0") or 100_000.0), risk


def _bh(feed, ticker: str, start: date, end: date) -> Optional[float]:
    b0, b1 = feed.bar(ticker, start), feed.bar(ticker, end)
    if b0 and b1 and float(b0.close) > 0:
        return float(b1.close) / float(b0.close) - 1.0
    return None


def _geo(rs: Sequence[float]) -> Optional[float]:
    if not rs:
        return None
    g = 1.0
    for r in rs:
        g *= 1.0 + float(r)
        if g <= 0:
            return -1.0
    return g ** (1.0 / len(rs)) - 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", default="paper_live/cloud/zoo_options_amplify.json")
    ap.add_argument("--out", default="reports/options_amplify")
    ap.add_argument("--lookback-days", type=int, default=2000)
    ap.add_argument("--max-strategies", type=int, default=None)
    ap.add_argument("--cache-dir", default=None, help="Reuse ohlcv cache pack")
    ap.add_argument(
        "--source",
        choices=("eodhd", "yahoo", "auto"),
        default="eodhd",
        help="Market data source (default eodhd)",
    )
    ap.add_argument("--eodhd-from", default="2020-01-01")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    # ensure zoo exists
    zoo_path = Path(args.zoo)
    if not zoo_path.is_file():
        logger.info("Building amplify zoo…")
        from scripts.build_options_amplify_zoo import main as build_zoo

        build_zoo()

    specs, capital0, risk = _specs(zoo_path)
    if args.max_strategies:
        specs = specs[: int(args.max_strategies)]

    unds = sorted({s.underlying.upper() for s in specs} | {"SPY", "QQQ", "IWM", "VIX", "VIX3M"})
    cache = Path(args.cache_dir) if args.cache_dir else Path(args.out) / "data_cache"
    logger.info("Feed tickers=%d n_strats=%d source=%s", len(unds), len(specs), args.source)
    sources: dict = {}
    feed = None
    if args.synthetic:
        feed, sources = build_cloud_feed(
            unds,
            cache_dir=cache,
            seed_dir=SEED_DIR,
            lookback_calendar_days=int(args.lookback_days),
            force_synthetic=True,
            require_real=False,
            min_real_tickers=0,
        )
    elif args.source in ("eodhd", "auto"):
        try:
            from paper_live.data.eodhd_client import build_eodhd_feed, get_token

            get_token()
            feed, sources = build_eodhd_feed(
                unds,
                start=str(args.eodhd_from),
                cache_dir=cache / "eodhd",
                min_history=50,
            )
            logger.info("Using EODHD EOD feed")
        except Exception as e:
            if args.source == "eodhd":
                raise SystemExit(f"EODHD required but failed: {e}")
            logger.warning("EODHD failed (%s); falling back to Yahoo", e)
            feed = None
    if feed is None:
        feed, sources = build_cloud_feed(
            unds,
            cache_dir=cache,
            seed_dir=SEED_DIR,
            lookback_calendar_days=int(args.lookback_days),
            force_synthetic=False,
            require_real=True,
            min_real_tickers=3,
        )
        logger.info("Using Yahoo/cloud feed")
    days = list(feed.days)
    if not days:
        raise SystemExit("no days")

    # windows clamped
    wins: List[Tuple[str, date, date]] = []
    for name, ws, we in WINDOWS:
        req_s = date.fromisoformat(ws)
        req_e = days[-1] if we.startswith("2099") else date.fromisoformat(we)
        s = next((d for d in days if d >= req_s), days[0])
        e = next((d for d in reversed(days) if d <= req_e), days[-1])
        wins.append((name, s, e))
        logger.info("Window %s %s→%s", name, s, e)

    # per strategy year returns
    from collections import defaultdict

    year_map: Dict[str, Dict[str, Any]] = defaultdict(dict)
    meta_map: Dict[str, Dict[str, Any]] = {}

    for name, start_d, end_d in wins:
        spy_bh = _bh(feed, "SPY", start_d, end_d)
        qqq_bh = _bh(feed, "QQQ", start_d, end_d)
        iwm_bh = _bh(feed, "IWM", start_d, end_d)
        logger.info(
            "%s BH SPY=%s QQQ=%s",
            name,
            spy_bh,
            qqq_bh,
        )
        results = run_options_batch(
            feed,
            specs,
            start=start_d,
            end=end_d,
            capital0=capital0,
            risk=risk,
            data_label="proxy_bs",
            spy_bh=spy_bh,
            qqq_bh=qqq_bh,
        )
        for r in results:
            year_map[r.strategy_id][name] = {
                "total_return": r.total_return,
                "max_dd": r.max_dd,
                "n_opens": r.n_opens,
                "hard_kill": r.hard_kill,
                "vs_spy": r.vs_spy_bh,
                "vs_qqq": r.vs_qqq_bh,
                "data_label": r.data_label,
                "kind": r.kind,
                "underlying": r.underlying,
            }
            meta_map[r.strategy_id] = {
                "label": r.label,
                "kind": r.kind,
                "underlying": r.underlying,
            }
        # store bench on a special key
        year_map["_BENCH"][name] = {
            "SPY": spy_bh,
            "QQQ": qqq_bh,
            "IWM": iwm_bh,
        }

    # aggregate
    rows = []
    bench_years = year_map.get("_BENCH") or {}
    qqq_means = [v["QQQ"] for v in bench_years.values() if v.get("QQQ") is not None]
    spy_means = [v["SPY"] for v in bench_years.values() if v.get("SPY") is not None]
    qqq_mean = sum(qqq_means) / len(qqq_means) if qqq_means else None
    spy_mean = sum(spy_means) / len(spy_means) if spy_means else None

    for sid, ydict in year_map.items():
        if sid == "_BENCH":
            continue
        rets = []
        dds = []
        pos = 0
        kills = 0
        yr = {}
        for name, _, _ in wins:
            cell = ydict.get(name)
            if not cell:
                continue
            tr = float(cell["total_return"])
            rets.append(tr)
            yr[name] = tr
            if cell.get("max_dd") is not None:
                dds.append(float(cell["max_dd"]))
            if tr > 0:
                pos += 1
            if cell.get("hard_kill"):
                kills += 1
        mean_ret = sum(rets) / len(rets) if rets else None
        geo = _geo(rets)
        worst_dd = min(dds) if dds else None
        xs_spy = (mean_ret - spy_mean) if mean_ret is not None and spy_mean is not None else None
        xs_qqq = (mean_ret - qqq_mean) if mean_ret is not None and qqq_mean is not None else None
        m = meta_map.get(sid) or {}
        is_income = str(m.get("kind") or "") in (
            "covered_call",
            "cash_secured_put",
            "put_credit_spread",
            "call_credit_spread",
            "iron_condor",
        )
        rows.append(
            {
                "strategy_id": sid,
                "label": m.get("label"),
                "kind": m.get("kind"),
                "underlying": m.get("underlying"),
                "family": "income_control" if is_income else "amplify",
                "mean_ret": mean_ret,
                "geo_ret": geo,
                "worst_dd": worst_dd,
                "mean_xs_spy": xs_spy,
                "mean_xs_qqq": xs_qqq,
                "n_positive_years": pos,
                "hard_kill_years": kills,
                "year_returns": yr,
                "year_detail": ydict,
            }
        )

    rows.sort(key=lambda r: -(r.get("mean_ret") if r.get("mean_ret") is not None else -1e9))

    # verdicts amplify only
    promote, watch, kill = [], [], []
    for r in rows:
        if r.get("family") == "income_control" or r.get("kind") == "cash":
            r["verdict"] = "HOLD_CTRL"
            continue
        mr = r.get("mean_ret")
        if mr is None:
            r["verdict"] = "KILL_AMP"
            kill.append(r)
            continue
        beat_qqq = qqq_mean is not None and mr >= qqq_mean + 0.03
        beat_spy = spy_mean is not None and (r.get("mean_xs_spy") or -1) >= 0.05
        dd_ok = r.get("worst_dd") is None or r["worst_dd"] > -0.70
        pos_ok = int(r.get("n_positive_years") or 0) >= 2
        if (beat_qqq or beat_spy) and dd_ok and pos_ok and int(r.get("hard_kill_years") or 0) == 0:
            r["verdict"] = "PROMOTE_AMP"
            promote.append(r)
        elif mr > (spy_mean or 0) and (not pos_ok or not dd_ok):
            r["verdict"] = "WATCH_AMP"
            watch.append(r)
        elif mr < (spy_mean or 0) - 0.05:
            r["verdict"] = "KILL_AMP"
            kill.append(r)
        else:
            r["verdict"] = "WATCH_AMP"
            watch.append(r)

    out = Path(args.out)
    latest = out / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "zoo": str(zoo_path),
        "n_strategies": len(specs),
        "data_sources": sources,
        "windows": [{"name": n, "start": s.isoformat(), "end": e.isoformat()} for n, s, e in wins],
        "benchmarks_mean": {"spy": spy_mean, "qqq": qqq_mean},
        "strategies": rows,
        "promote": promote,
        "watch": watch,
        "kill": kill,
        "disclaimer": (
            "proxy_bs|vix_surface marks. No OPRA historical chains. "
            "Chain packs today-only if downloaded separately. VIRTUAL."
        ),
    }
    (latest / "full_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    def pct(x):
        try:
            return f"{float(x):.2%}"
        except Exception:
            return "n/a"

    lines = [
        f"# Options AMPLIFY matrix — mean multi-year rank",
        "",
        f"**N strategies:** {len(specs)} · **Label:** `proxy_bs|vix_surface`",
        f"**SPY mean BH:** {pct(spy_mean)} · **QQQ mean BH:** {pct(qqq_mean)}",
        "",
        "## Data honesty",
        "",
        "- Underlying/VIX path: **EODHD EOD** when `--source eodhd` (default).",
        "- Option marks: Black–Scholes on VIX surface / HV (`proxy_bs|vix_surface`) unless EODHD UnicornBay options subscribed.",
        "- EODHD US options marketplace is a **separate** add-on; 403 → `eodhd_options_not_subscribed`.",
        "- Download pack: `scripts/download_eodhd_options_research.py`.",
        "",
        f"## PROMOTE_AMP ({len(promote)})",
        "",
    ]
    if promote:
        for p in promote[:15]:
            lines.append(
                f"- `{p['strategy_id']}` mean={pct(p['mean_ret'])} "
                f"xsQQQ={pct(p.get('mean_xs_qqq'))} DD={pct(p.get('worst_dd'))} "
                f"years={p.get('year_returns')}"
            )
    else:
        lines.append("_None beat QQQ mean+3pp or SPY+5pp with DD/pos filters._")

    lines += [
        "",
        "## Ranking by mean_ret (amplify + controls)",
        "",
        "| Rank | ID | Kind | Und | Mean | Geo | xsSPY | xsQQQ | DD | +yrs | Verdict |",
        "|------|----|------|-----|------|-----|-------|-------|----|------|---------|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | `{r['strategy_id']}` | {r.get('kind')} | {r.get('underlying')} | "
            f"{pct(r.get('mean_ret'))} | {pct(r.get('geo_ret'))} | {pct(r.get('mean_xs_spy'))} | "
            f"{pct(r.get('mean_xs_qqq'))} | {pct(r.get('worst_dd'))} | "
            f"{r.get('n_positive_years')} | {r.get('verdict')} |"
        )
    lines += ["", "---", payload["disclaimer"], ""]
    (latest / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "RESCORE.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "RESCORE.json").write_text(
        json.dumps(
            {"promote": promote, "watch": watch, "kill": kill, "benchmarks_mean": payload["benchmarks_mean"]},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info(
        "PROMOTE_AMP=%d WATCH=%d KILL=%d → %s",
        len(promote),
        len(watch),
        len(kill),
        latest / "SUMMARY.md",
    )
    print(
        json.dumps(
            {
                "n": len(specs),
                "promote": len(promote),
                "watch": len(watch),
                "kill": len(kill),
                "top": [
                    {"id": r["strategy_id"], "mean": r.get("mean_ret"), "verdict": r.get("verdict")}
                    for r in rows[:8]
                ],
                "summary": str(latest / "SUMMARY.md"),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
