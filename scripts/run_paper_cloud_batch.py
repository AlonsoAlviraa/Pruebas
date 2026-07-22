#!/usr/bin/env python3
"""Free multi-strategy paper cloud batch (for GitHub Actions).

Virtual capital only. Yahoo (real) OHLCV primary; synthetic only with --synthetic.
Use --start 2026-01-01 to study YTD 2026 (or any OOS window).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.cloud.batch import run_cloud_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper cloud multi-strategy batch (free)")
    ap.add_argument("--out", default="reports/paper_cloud")
    ap.add_argument("--zoo", default=None, help="Path to strategy_zoo.json")
    ap.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    ap.add_argument("--lookback-days", type=int, default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--keep-ledgers", action="store_true")
    args = ap.parse_args()

    result = run_cloud_batch(
        out_root=args.out,
        zoo_path=args.zoo,
        force_synthetic=bool(args.synthetic),
        lookback_days=args.lookback_days,
        start_date=args.start,
        end_date=args.end,
        keep_ledgers=bool(args.keep_ledgers),
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    print(f"\nWrote study pack under: {result.out_dir}", file=sys.stderr)
    print(f"Latest: {Path(args.out) / 'latest' / 'SUMMARY.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
