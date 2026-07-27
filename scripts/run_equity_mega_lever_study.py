#!/usr/bin/env python3
"""Mega equity strategy study: thousands of sleeves + signal-scaled broker leverage.

- Combinatorial long-only equity signals (trend/mom/RSI/breakout/top-k…)
- Leverage ≤2× only when signal strength is high (broker/Reg-T style)
- Financing on (L−1) + IBKR-like commissions/slippage on rebalances
- Multi-year returns vs SPY BH (and cash-aware notes)
- Data: EODHD EOD underlyings

VIRTUAL only. Not financial advice.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.equity.cost_drag import CostDragConfig
from paper_live.equity.grid_zoo import write_equity_grid_zoo
from paper_live.equity.signal_backtest import run_equity_spec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eq_mega")

_W_FEED = None
_W_COST: Optional[CostDragConfig] = None
_W_CAPITAL0 = 100_000.0
_W_START: Optional[date] = None
_W_END: Optional[date] = None


def _specs_from_zoo(path: Path, max_n: Optional[int] = None) -> Tuple[List[Dict[str, Any]], float]:
    z = json.loads(path.read_text(encoding="utf-8"))
    raw = list(z.get("strategies") or [])
    if max_n and len(raw) > max_n:
        idxs = np.linspace(0, len(raw) - 1, num=max_n, dtype=int)
        seen = set()
        picked = []
        for i in idxs:
            if int(i) not in seen:
                seen.add(int(i))
                picked.append(raw[int(i)])
        raw = picked[:max_n]
    return raw, float(z.get("capital0") or 100_000.0)


def _bh(feed, ticker: str, start: date, end: date) -> Optional[float]:
    days = feed.session_days(start, end)
    if not days:
        return None
    b0 = feed.bar(ticker, days[0])
    b1 = feed.bar(ticker, days[-1])
    if b0 is None or b1 is None or float(b0.close) <= 0:
        return None
    return float(b1.close) / float(b0.close) - 1.0


def _worker_init(unds, eodhd_from, cache_dir, capital0, start_iso, end_iso, cost_dict):
    global _W_FEED, _W_COST, _W_CAPITAL0, _W_START, _W_END
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from paper_live.data.eodhd_client import build_eodhd_feed

    feed, _ = build_eodhd_feed(
        unds, start=eodhd_from, cache_dir=Path(cache_dir), min_history=60
    )
    _W_FEED = feed
    _W_COST = CostDragConfig(**cost_dict) if cost_dict else CostDragConfig()
    _W_CAPITAL0 = capital0
    _W_START = date.fromisoformat(start_iso) if start_iso else None
    _W_END = date.fromisoformat(end_iso) if end_iso else None


def _worker_run(spec: Dict[str, Any]) -> Dict[str, Any]:
    assert _W_FEED is not None
    r = run_equity_spec(
        _W_FEED,
        spec,
        start=_W_START,
        end=_W_END,
        capital0=_W_CAPITAL0,
        cost=_W_COST,
    )
    return r.to_dict()


def main() -> int:
    ap = argparse.ArgumentParser(description="Equity mega lever study")
    ap.add_argument("--zoo", default="paper_live/cloud/zoo_equity_grid.json")
    ap.add_argument("--max-strategies", type=int, default=1500)
    ap.add_argument("--build-grid", action="store_true")
    ap.add_argument("--from-year", type=int, default=2015)
    ap.add_argument("--to-year", type=int, default=2025)
    ap.add_argument("--eodhd-from", default="2010-01-01")
    ap.add_argument("--out", default="reports/equity_mega_lever")
    ap.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 4) - 2))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.max_strategies = min(args.max_strategies, 80)
        args.from_year = max(args.from_year, 2018)

    zoo_path = Path(args.zoo)
    if args.build_grid or not zoo_path.is_file():
        logger.info("Building equity grid max=%d", max(args.max_strategies * 2, 2000))
        write_equity_grid_zoo(zoo_path, max_strategies=max(args.max_strategies * 2, 2000))

    specs, capital0 = _specs_from_zoo(zoo_path, max_n=args.max_strategies)
    unds = set()
    for s in specs:
        u = str(s.get("underlying") or "SPY").upper()
        if u != "BASKET":
            unds.add(u)
        for t in (s.get("meta") or {}).get("basket") or []:
            unds.add(str(t).upper())
    unds |= {"SPY", "QQQ", "IWM"}
    unds = sorted(unds)
    kinds = sorted({str(s.get("kind")) for s in specs})
    logger.info(
        "Strategies=%d unds=%d kinds=%s years=%d-%d workers=%d",
        len(specs),
        len(unds),
        kinds,
        args.from_year,
        args.to_year,
        args.workers,
    )

    from paper_live.data.eodhd_client import build_eodhd_feed

    out_root = Path(args.out)
    latest = out_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    cache = out_root / "eodhd_cache"

    feed, sources = build_eodhd_feed(
        unds, start=args.eodhd_from, cache_dir=cache, min_history=60
    )
    days = list(feed.days)
    logger.info("Feed %s→%s n=%d", days[0], days[-1], len(days))

    start_d = date(args.from_year, 1, 2)
    end_d = date(args.to_year, 12, 31)
    s = next((d for d in days if d >= start_d), days[0])
    e = next((d for d in reversed(days) if d <= end_d), days[-1])

    # load cost model if present
    cost_path = ROOT / "paper_live" / "config" / "cost_model.json"
    cost = CostDragConfig()
    if cost_path.is_file():
        cost = CostDragConfig.from_cost_model_json(
            json.loads(cost_path.read_text(encoding="utf-8"))
        )
    cost_dict = cost.to_dict()

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    if args.workers <= 1:
        for i, sp in enumerate(specs, 1):
            r = run_equity_spec(
                feed, sp, start=s, end=e, capital0=capital0, cost=cost
            )
            results.append(r.to_dict())
            if i % 25 == 0 or i == len(specs):
                logger.info("Progress %d/%d ETA %.1f min", i, len(specs),
                            ((time.time()-t0)/max(i,1))*(len(specs)-i)/60)
    else:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(specs)),
            initializer=_worker_init,
            initargs=(
                unds,
                args.eodhd_from,
                str(cache.resolve()),
                capital0,
                s.isoformat(),
                e.isoformat(),
                cost_dict,
            ),
        ) as ex:
            futs = {ex.submit(_worker_run, sp): sp.get("id") for sp in specs}
            done = 0
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exn:
                    logger.exception("fail %s: %s", futs[fut], exn)
                    results.append(
                        {
                            "strategy_id": futs[fut],
                            "total_return": 0.0,
                            "max_dd": 0.0,
                            "error": str(exn),
                            "kind": "",
                            "underlying": "",
                        }
                    )
                done += 1
                if done % 50 == 0 or done == len(specs):
                    logger.info(
                        "Progress %d/%d ETA %.1f min",
                        done,
                        len(specs),
                        ((time.time() - t0) / max(done, 1)) * (len(specs) - done) / 60,
                    )

    (latest / "sleeve_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    # Rank by total return and by Calmar-like (prefer non-wiped first)
    scored = []
    for r in results:
        tr = float(r.get("total_return") or 0)
        dd = abs(float(r.get("max_dd") or 0)) + 1e-6
        wiped = bool(r.get("wiped"))
        scored.append(
            {
                **r,
                "score": tr / dd,
                "calmar_like": tr / dd,
                "rank_key_ret": (0 if wiped else 1, tr),
                "rank_key_cal": (0 if wiped else 1, tr / dd),
            }
        )
    by_ret = sorted(
        scored, key=lambda x: (x["rank_key_ret"][0], x["rank_key_ret"][1]), reverse=True
    )
    by_cal = sorted(
        scored, key=lambda x: (x["rank_key_cal"][0], x["rank_key_cal"][1]), reverse=True
    )
    n_wiped = sum(1 for r in results if r.get("wiped"))

    spy_bh = _bh(feed, "SPY", s, e)
    qqq_bh = _bh(feed, "QQQ", s, e)

    # annual SPY for year table
    year_spy: Dict[str, Optional[float]] = {}
    for y in range(args.from_year, args.to_year + 1):
        ys, ye = date(y, 1, 2), date(y, 12, 31)
        year_spy[str(y)] = _bh(feed, "SPY", ys, ye)

    # mean annual across strategies that have year_returns
    def mean_year_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        acc: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            for y, v in (r.get("year_returns") or {}).items():
                acc[y].append(float(v))
        return {y: float(np.mean(v)) for y, v in acc.items()}

    top20 = by_ret[:20]
    top20_cal = by_cal[:20]

    summary = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "n_strategies": len(specs),
        "kinds": kinds,
        "from": s.isoformat(),
        "to": e.isoformat(),
        "data": {"underlyings": "eodhd_eod", "leverage": "signal_scaled_broker_proxy"},
        "cost_profile": cost.to_dict(),
        "spy_bh": spy_bh,
        "qqq_bh": qqq_bh,
        "mean_strategy_return": float(
            np.mean([float(r.get("total_return") or 0) for r in results])
        )
        if results
        else None,
        "median_strategy_return": float(
            np.median([float(r.get("total_return") or 0) for r in results])
        )
        if results
        else None,
        "pct_beat_spy": float(
            np.mean(
                [
                    float(r.get("total_return") or 0) > float(spy_bh or 0)
                    for r in results
                ]
            )
        )
        if results and spy_bh is not None
        else None,
        "n_wiped": int(n_wiped),
        "pct_wiped": float(n_wiped / max(len(results), 1)),
        "top20_by_return": top20,
        "top20_by_calmar": top20_cal,
        "kind_counts": dict(Counter(str(r.get("kind")) for r in results)),
        "year_spy": year_spy,
        "disclaimer": (
            "VIRTUAL. Equity EODHD EOD. Leverage is broker-style proxy (≤2×) with "
            "financing + commissions/slippage. Not financial advice."
        ),
    }
    (latest / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    def pct(x):
        try:
            return f"{100*float(x):+.2f}%"
        except Exception:
            return "n/a"

    lines = [
        "# Equity mega study — signal leverage + broker costs",
        "",
        f"**Strategies:** {len(specs)} · **Window:** {s} → {e}",
        f"**Data:** EODHD EOD · **Leverage:** signal-scaled ≤2× + financing",
        f"**Costs:** IBKR-like commission/slippage on rebalance (see cost_profile)",
        "",
        "## Headline",
        "",
        f"| SPY BH | QQQ BH | Strat mean | Strat median | % beat SPY | % wiped (DD kill) |",
        f"|--------|--------|------------|--------------|------------|-------------------|",
        f"| {pct(spy_bh)} | {pct(qqq_bh)} | {pct(summary['mean_strategy_return'])} | "
        f"{pct(summary['median_strategy_return'])} | "
        f"{100*(summary['pct_beat_spy'] or 0):.1f}% | "
        f"{100*(summary.get('pct_wiped') or 0):.1f}% |",
        "",
        "## Top 15 by total return (non-wiped preferred)",
        "",
        "| Rank | ID | Kind | Und | Return | MaxDD | Mean L | Cost drag | Wiped |",
        "|------|----|------|-----|--------|-------|--------|-----------|-------|",
    ]
    for i, r in enumerate(top20[:15], 1):
        lines.append(
            f"| {i} | {r.get('strategy_id','')[:40]} | {r.get('kind')} | {r.get('underlying')} | "
            f"{pct(r.get('total_return'))} | {pct(r.get('max_dd'))} | "
            f"{float(r.get('mean_leverage') or 0):.2f} | {float(r.get('cost_drag_total') or 0):.4f} | "
            f"{'Y' if r.get('wiped') else ''} |"
        )
    lines += [
        "",
        "## Top 10 by Calmar-like (ret / |maxDD|)",
        "",
        "| Rank | ID | Kind | Return | MaxDD | Score |",
        "|------|----|------|--------|-------|-------|",
    ]
    for i, r in enumerate(top20_cal[:10], 1):
        lines.append(
            f"| {i} | {str(r.get('strategy_id'))[:40]} | {r.get('kind')} | "
            f"{pct(r.get('total_return'))} | {pct(r.get('max_dd'))} | "
            f"{float(r.get('calmar_like') or 0):.2f} |"
        )
    lines += [
        "",
        "## Method",
        "",
        "1. Grid of long-only equity rules (SMA, mom, RSI, breakout, top-k basket, …).",
        "2. **Signal strength** maps to leverage: weak → base (1×), strong → up to **2×**.",
        "3. **Financing** daily on (L−1); **commissions + slippage** on |ΔL| rebalances.",
        "4. Hard DD kill; rank by total return and Calmar-like vs SPY/QQQ BH.",
        "",
        "---",
        summary["disclaimer"],
        "",
    ]
    (latest / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "DONE n=%d mean=%s spy=%s → %s",
        len(specs),
        summary["mean_strategy_return"],
        spy_bh,
        latest / "SUMMARY.md",
    )
    print(
        json.dumps(
            {
                "n_strategies": len(specs),
                "spy_bh": spy_bh,
                "qqq_bh": qqq_bh,
                "mean": summary["mean_strategy_return"],
                "median": summary["median_strategy_return"],
                "pct_beat_spy": summary["pct_beat_spy"],
                "top1": top20[0] if top20 else None,
                "summary": str(latest / "SUMMARY.md"),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
