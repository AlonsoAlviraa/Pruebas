#!/usr/bin/env python3
"""Mega annual alpha study CLI.

Search equity + options zoos calendar-year by year for strategies that beat
max(SPY, QQQ, IWM) B&H by +3 percentage points.

Virtual capital only. Options marks: proxy_bs | vix_surface (never OPRA).

Examples
--------
# Full Yahoo study (2022–2025_study):
python scripts/run_mega_annual_alpha_study.py --out reports/mega_annual_alpha

# Faster smoke (synthetic, capped):
python scripts/run_mega_annual_alpha_study.py --synthetic --max-equity 5 --max-options 2 \\
  --lookback-days 900 --out reports/mega_annual_alpha_synth

# Equity only:
python scripts/run_mega_annual_alpha_study.py --skip-options --max-equity 20
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

from paper_live.cloud.mega_annual_alpha import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_ZOO,
    run_mega_annual_alpha_study,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mega_annual_alpha")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Mega annual alpha study: strategies vs max(SPY,QQQ,IWM)+3pp per year. "
            "VIRTUAL capital · paper equity · options proxy_bs|vix_surface."
        )
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Report root (default reports/mega_annual_alpha)",
    )
    ap.add_argument(
        "--zoo",
        default=str(DEFAULT_ZOO),
        help="Mega zoo JSON (equity_strategies + options_strategies)",
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic OHLCV (tests / offline). Never claim real alpha.",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=1800,
        help="Calendar lookback for free data (need ~2022+)",
    )
    ap.add_argument(
        "--max-equity",
        type=int,
        default=None,
        help="Cap number of equity strategies",
    )
    ap.add_argument(
        "--max-options",
        type=int,
        default=None,
        help="Cap number of options strategies",
    )
    ap.add_argument(
        "--max-strategies",
        type=int,
        default=None,
        help="If set, apply same cap to equity (and leave options unless --max-options set)",
    )
    ap.add_argument("--skip-options", action="store_true")
    ap.add_argument("--skip-equity", action="store_true")
    ap.add_argument(
        "--min-opens",
        type=int,
        default=0,
        help="Min opens/trades for winner filter (0 = off)",
    )
    ap.add_argument(
        "--min-real-tickers",
        type=int,
        default=5,
        help="Min real tickers when not --synthetic",
    )
    ap.add_argument(
        "--keep-ledgers",
        action="store_true",
        help="Retain per-strategy ledger dirs (slow; for debugging fills)",
    )
    ap.add_argument(
        "--lean-ledger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In-memory ledger, no JSONL audit (default on). Use --no-lean-ledger for full audit.",
    )
    args = ap.parse_args()

    max_eq = args.max_equity
    max_opt = args.max_options
    if args.max_strategies is not None:
        if max_eq is None:
            max_eq = int(args.max_strategies)
        if max_opt is None and not args.skip_options:
            # keep a modest options control set unless user capped
            max_opt = min(10, int(args.max_strategies))

    try:
        result = run_mega_annual_alpha_study(
            out_root=args.out,
            zoo_path=args.zoo,
            force_synthetic=bool(args.synthetic),
            lookback_days=int(args.lookback_days),
            max_equity=max_eq,
            max_options=max_opt,
            skip_options=bool(args.skip_options),
            skip_equity=bool(args.skip_equity),
            min_opens=int(args.min_opens),
            keep_ledgers=bool(args.keep_ledgers),
            lean_ledger=bool(args.lean_ledger),
            min_real_tickers=int(args.min_real_tickers),
        )
    except Exception as e:
        logger.error("Study failed: %s", e)
        print(json.dumps({"error": str(e), "mode": "paper"}, indent=2), file=sys.stderr)
        return 1

    n_years = len(result.windows_meta) or 1
    strict_key = f"{n_years}/{n_years}"
    strict = result.tiers.get(strict_key) or []
    print(
        json.dumps(
            {
                **result.to_dict(),
                "strict_winner_ids": [s.strategy_id for s in strict],
                "n_strict_winners": len(strict),
            },
            indent=2,
            default=str,
        )
    )
    summary = Path(args.out) / "latest" / "SUMMARY.md"
    print(f"\nWrote pack: {summary}", file=sys.stderr)
    print(f"Strict winners ({strict_key}): {len(strict)}", file=sys.stderr)
    if not strict:
        print(
            "NOTE: Zero strategies cleared +3pp over best index every year "
            "(valid scientific result). See tiers in SUMMARY.md.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
