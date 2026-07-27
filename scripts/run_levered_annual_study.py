#!/usr/bin/env python3
"""Run levered multi-year study; rank by mean return; PROMOTE really-good sleeves.

VIRTUAL capital. Labels: levered_proxy / etf_levered_proxy.
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

from paper_live.cloud.levered_annual import run_levered_annual_study

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Levered annual alpha study")
    ap.add_argument("--out", default="reports/levered_annual")
    ap.add_argument("--zoo", default="paper_live/cloud/zoo_levered_alpha.json")
    ap.add_argument("--lookback-days", type=int, default=2000)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-strategies", type=int, default=None)
    args = ap.parse_args()

    result = run_levered_annual_study(
        out_root=Path(args.out),
        zoo_path=Path(args.zoo),
        lookback_days=int(args.lookback_days),
        force_synthetic=bool(args.synthetic),
        max_strategies=args.max_strategies,
    )
    print(json.dumps({
        "paths": result.get("paths"),
        "n_promote": result.get("meta", {}).get("n_promote"),
        "n_watch": result.get("meta", {}).get("n_watch"),
        "qqq_bh_mean": result.get("meta", {}).get("qqq_bh_mean"),
        "top_promote": [
            {"id": p.get("strategy_id"), "mean_ret": p.get("mean_ret")}
            for p in (result.get("promote") or [])[:5]
        ],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
