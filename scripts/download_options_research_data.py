#!/usr/bin/env python3
"""Download *as much free options research data as available*.

What this can download without paid OPRA:
  1) Yahoo OHLCV for many underlyings + VIX/VIX3M (historical multi-year)
  2) Yahoo options **chain snapshots for today** (not historical chains)

Labels:
  - OHLCV: yahoo
  - chain: yahoo_chain (point-in-time only)
  - Historical option marks in backtests remain proxy_bs|vix_surface

Never invents chain history.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.cloud.free_data import SEED_DIR, build_cloud_feed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_options_research")

# Broad liquid US names + indices for amplify research
DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX", "AVGO",
    "JPM", "XOM", "UNH", "V", "MA", "COST",
    "VIX", "VIX3M",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Download free options research data pack")
    ap.add_argument("--out", default="reports/options_data_pack")
    ap.add_argument("--lookback-days", type=int, default=2000)
    ap.add_argument("--tickers", default="", help="Comma list; default large universe")
    ap.add_argument("--no-chain", action="store_true", help="Skip Yahoo chain snapshots")
    ap.add_argument(
        "--chain-tickers",
        default="SPY,QQQ,IWM,AAPL,NVDA,MSFT,META,AMZN,GOOGL,TSLA",
        help="Underlyings for live chain snapshot",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "ohlcv_cache"
    cache.mkdir(parents=True, exist_ok=True)

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else list(DEFAULT_UNIVERSE)
    )
    logger.info("Downloading OHLCV for %d tickers lookback=%d", len(tickers), args.lookback_days)
    feed, sources = build_cloud_feed(
        tickers,
        cache_dir=cache,
        seed_dir=SEED_DIR,
        lookback_calendar_days=int(args.lookback_days),
        force_synthetic=False,
        require_real=True,
        min_real_tickers=min(5, len(tickers)),
    )

    ohlcv_manifest = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "lookback_days": args.lookback_days,
        "tickers": tickers,
        "data_sources": sources,
        "n_days": len(feed.days),
        "first_day": feed.days[0].isoformat() if feed.days else None,
        "last_day": feed.days[-1].isoformat() if feed.days else None,
        "label": "yahoo_ohlcv",
        "note": "Underlying prices only. Not option chain history.",
    }
    (out / "ohlcv_manifest.json").write_text(
        json.dumps(ohlcv_manifest, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "OHLCV ready: %d days %s→%s sources_ok=%d",
        ohlcv_manifest["n_days"],
        ohlcv_manifest["first_day"],
        ohlcv_manifest["last_day"],
        sum(1 for v in sources.values() if str(v).startswith("yahoo")),
    )

    chain_pack: Dict[str, Any] = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "data_label": "yahoo_chain",
        "note": "Point-in-time only. NOT historical OPRA. Failures labeled yahoo_chain_failed.",
        "underlyings": {},
    }
    if not args.no_chain:
        from paper_live.options.yahoo_chain import fetch_yahoo_option_chain

        chain_tickers = [t.strip().upper() for t in args.chain_tickers.split(",") if t.strip()]
        for t in chain_tickers:
            try:
                snap = fetch_yahoo_option_chain(t)
                d = snap.to_dict()
                # keep summary smaller: drop full lists if huge already truncated
                chain_pack["underlyings"][t] = {
                    "ok": d.get("ok"),
                    "spot": d.get("spot"),
                    "n_calls": d.get("n_calls"),
                    "n_puts": d.get("n_puts"),
                    "expirations": (d.get("expirations") or [])[:12],
                    "data_label": d.get("data_label"),
                    "error": d.get("error"),
                    # sample near-ATM quotes for research
                    "sample_calls": (d.get("calls") or [])[:15],
                    "sample_puts": (d.get("puts") or [])[:15],
                }
                logger.info(
                    "Chain %s ok=%s calls=%s puts=%s",
                    t,
                    d.get("ok"),
                    d.get("n_calls"),
                    d.get("n_puts"),
                )
            except Exception as e:
                chain_pack["underlyings"][t] = {
                    "ok": False,
                    "data_label": "yahoo_chain_failed",
                    "error": str(e),
                }
                logger.warning("Chain %s failed: %s", t, e)
            time.sleep(0.4)
        (out / "chain_snapshots.json").write_text(
            json.dumps(chain_pack, indent=2, default=str), encoding="utf-8"
        )
        # also per-ticker files
        cdir = out / "chains"
        cdir.mkdir(exist_ok=True)
        for t, body in chain_pack["underlyings"].items():
            (cdir / f"{t}.json").write_text(
                json.dumps(body, indent=2, default=str), encoding="utf-8"
            )

    # optional surface error vs model
    try:
        from paper_live.options.chain_diag import diagnose_chain_vs_model
        from paper_live.options.vol_surface import resolve_vix_level, VIX_TICKERS, VIX3M_TICKERS

        last = feed.days[-1]
        vix = resolve_vix_level(feed, last, aliases=VIX_TICKERS)
        vix3m = resolve_vix_level(feed, last, aliases=VIX3M_TICKERS)
        diag = diagnose_chain_vs_model(
            [t for t in ("SPY", "QQQ", "AAPL", "NVDA") if t in chain_pack.get("underlyings", {}) or True],
            vix=vix,
            vix3m=vix3m,
        )
        (out / "chain_vs_model_diag.json").write_text(
            json.dumps(diag, indent=2, default=str), encoding="utf-8"
        )
    except Exception as e:
        logger.warning("chain_diag skip: %s", e)

    summary = {
        "ohlcv_manifest": str(out / "ohlcv_manifest.json"),
        "chain_snapshots": str(out / "chain_snapshots.json") if not args.no_chain else None,
        "cache_dir": str(cache),
        "disclaimer": (
            "Historical backtests still use proxy_bs|vix_surface marks. "
            "Chain pack is TODAY only. Not OPRA history."
        ),
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
