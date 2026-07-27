#!/usr/bin/env python3
"""Download EODHD research pack for options amplify studies.

Primary: EOD OHLCV for many US underlyings + VIX/VIX3M (label eodhd_eod).
Secondary: attempt US options marketplace chains (UnicornBay). If 403, document
``eodhd_options_not_subscribed`` — do not invent.

Env: EODHD_API_TOKEN or EODHD_API_KEY
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

from paper_live.data.eodhd_client import (
    build_eodhd_feed,
    fetch_options_marketplace,
    get_token,
    probe_options_subscription,
    user_info,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eodhd_options_research")

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX", "AVGO",
    "JPM", "XOM", "UNH", "V", "MA", "COST", "CRM", "ORCL", "ADBE", "INTC", "BA",
    "VIX", "VIX3M",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="EODHD download for options research")
    ap.add_argument("--out", default="reports/eodhd_options_pack")
    ap.add_argument("--from", dest="from_", default="2020-01-01")
    ap.add_argument("--to", default=None)
    ap.add_argument("--tickers", default="")
    ap.add_argument("--skip-options-probe", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    cache = out / "ohlcv_eodhd"
    out.mkdir(parents=True, exist_ok=True)

    try:
        tok = get_token()
        info = user_info()
        logger.info(
            "EODHD user subscription=%s dailyLimit=%s",
            info.get("subscriptionType"),
            info.get("dailyRateLimit"),
        )
        (out / "eodhd_user.json").write_text(
            json.dumps(
                {
                    k: info.get(k)
                    for k in (
                        "subscriptionType",
                        "subscriptionMode",
                        "dailyRateLimit",
                        "apiRequests",
                        "apiRequestsDate",
                        "email",
                    )
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        logger.error("EODHD auth/user failed: %s", e)
        return 2

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else list(DEFAULT_UNIVERSE)
    )
    feed, sources = build_eodhd_feed(
        tickers,
        start=args.from_,
        end=args.to,
        cache_dir=cache,
        min_history=40,
    )
    manifest = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "eodhd",
        "label": "eodhd_eod",
        "from": args.from_,
        "to": args.to,
        "tickers": tickers,
        "sources": sources,
        "n_days": len(feed.days),
        "first_day": feed.days[0].isoformat() if feed.days else None,
        "last_day": feed.days[-1].isoformat() if feed.days else None,
        "cache_dir": str(cache),
        "note": "Underlying EOD from EODHD. Options chain requires UnicornBay marketplace add-on.",
    }
    (out / "ohlcv_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "EODHD OHLCV: %d tickers ok, days %s→%s",
        sum(1 for v in sources.values() if v == "eodhd_eod"),
        manifest["first_day"],
        manifest["last_day"],
    )

    options_probe: dict = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "marketplace": "unicornbay_us_options",
        "underlyings": {},
        "subscribed": None,
        "endpoints": None,
    }
    if not args.skip_options_probe:
        # Full three-endpoint subscription probe (fail closed on 403)
        sub = probe_options_subscription()
        options_probe["subscribed"] = bool(sub.get("subscribed"))
        options_probe["endpoints"] = sub
        logger.info(
            "UnicornBay subscription probe subscribed=%s label=%s",
            sub.get("subscribed"),
            sub.get("data_label"),
        )
        # Spot-check a few underlyings only if subscribed (avoid 403 spam)
        if options_probe["subscribed"]:
            for t in ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]:
                res = fetch_options_marketplace(t, limit=5)
                options_probe["underlyings"][t] = res.to_dict()
                logger.info(
                    "Options contracts %s ok=%s n=%s",
                    t,
                    res.ok,
                    res.n_rows,
                )
        else:
            options_probe["underlyings"]["SPY"] = sub.get("contracts_spy")
        (out / "options_marketplace_probe.json").write_text(
            json.dumps(options_probe, indent=2, default=str), encoding="utf-8"
        )

    summary = {
        "ohlcv_manifest": str(out / "ohlcv_manifest.json"),
        "options_probe": str(out / "options_marketplace_probe.json")
        if not args.skip_options_probe
        else None,
        "options_subscribed": options_probe.get("subscribed"),
        "disclaimer": (
            "EODHD EOD used for underlyings/VIX. "
            "US options EOD chains require UnicornBay marketplace subscription. "
            "Amplify backtests use proxy_bs|vix_surface when chains unavailable."
        ),
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
