#!/usr/bin/env python3
"""Paper options multi-strategy batch (proxy BS on free OHLCV).

VIRTUAL capital only. Marks labeled proxy_bs — not exchange option fills.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.cloud.free_data import SEED_DIR, build_cloud_feed
from paper_live.options.replay_options import run_options_batch
from paper_live.options.strategies import OptionStrategySpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("options_batch")


def _specs_from_zoo(path: Path) -> list[OptionStrategySpec]:
    z = json.loads(path.read_text(encoding="utf-8"))
    out: list[OptionStrategySpec] = []
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
                meta=dict(s.get("meta") or {}),
                notes=str(s.get("notes") or ""),
            )
        )
    return out, float(z.get("capital0") or 100_000.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper options batch (proxy BS)")
    ap.add_argument("--out", default="reports/paper_options")
    ap.add_argument("--zoo", default="paper_live/cloud/zoo_options.json")
    ap.add_argument("--start", default="2025-10-29")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    zoo_path = Path(args.zoo)
    specs, capital0 = _specs_from_zoo(zoo_path)
    tickers = sorted({sp.underlying.upper() for sp in specs} | {"SPY", "QQQ"})
    feed, sources = build_cloud_feed(
        tickers,
        seed_dir=SEED_DIR,
        lookback_calendar_days=500,
        require_real=True,
        min_real_tickers=2,
    )
    days = feed.days
    if not days:
        raise SystemExit("No feed days")
    start = __import__("pandas").Timestamp(args.start).date()
    end = (
        __import__("pandas").Timestamp(args.end).date()
        if args.end
        else days[-1]
    )
    start = next((d for d in days if d >= start), days[0])
    end = next((d for d in reversed(days) if d <= end), days[-1])

    results = run_options_batch(feed, specs, start=start, end=end, capital0=capital0)
    spy0 = feed.bar("SPY", start)
    spy1 = feed.bar("SPY", end)
    spy_bh = None
    if spy0 and spy1 and float(spy0.close) > 0:
        spy_bh = float(spy1.close) / float(spy0.close) - 1.0

    out_root = Path(args.out)
    latest = out_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload = {
        "as_of": as_of,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "capital0": capital0,
        "data_label": "proxy_bs",
        "data_sources": sources,
        "benchmarks": {"spy_bh": spy_bh},
        "strategies": [r.to_dict() for r in results],
        "disclaimer": "proxy_bs marks — not real option fills. Virtual capital only.",
    }
    ranking = sorted(results, key=lambda r: r.total_return, reverse=True)
    lines = [
        f"# Paper options multi-strategy — `{as_of}`",
        "",
        f"**Window:** {start} → {end} · **Capital:** VIRTUAL ${capital0:,.0f}",
        "",
        f"**Data label:** `proxy_bs` (Black–Scholes on HV/IV proxy) — **NOT exchange fills**",
        "",
        f"**SPY B&H:** {spy_bh:.2%}" if spy_bh is not None else "**SPY B&H:** n/a",
        "",
        "## Ranking",
        "",
        "| Rank | ID | Kind | Return | MaxDD | Rolls | Und |",
        "|------|-----|------|--------|-------|-------|-----|",
    ]
    for i, r in enumerate(ranking, 1):
        lines.append(
            f"| {i} | `{r.strategy_id}` | {r.kind} | {r.total_return:.2%} | "
            f"{r.max_dd:.2%} | {r.n_rolls} | {r.underlying} |"
        )
    lines += [
        "",
        "## Design / papers",
        "",
        "See `docs/design/2026-07-22_paper_options_strategies.md` (VRP, covered call, CSP, spreads).",
        "",
        "---",
        f"_Generated {datetime.now(timezone.utc).isoformat()} · paper only · proxy_bs_",
        "",
    ]
    (latest / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (latest / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {latest / 'SUMMARY.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
