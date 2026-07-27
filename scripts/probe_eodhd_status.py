#!/usr/bin/env python3
"""Probe EODHD account: EOD underlyings + UnicornBay options endpoints."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.data.eodhd_client import (
    BASE,
    _http_get,
    fetch_eod,
    get_token,
    user_info,
)


def _try(name: str, url: str) -> dict:
    try:
        raw = _http_get(url, retries=1)
        data = json.loads(raw.decode("utf-8"))
        n = 0
        if isinstance(data, dict):
            n = len(data.get("data") or data.get("results") or [])
        elif isinstance(data, list):
            n = len(data)
        return {"name": name, "ok": True, "http": 200, "n": n, "preview": str(data)[:240]}
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:240]
        except Exception:
            pass
        return {"name": name, "ok": False, "http": e.code, "error": body or str(e)}
    except Exception as e:
        return {"name": name, "ok": False, "http": None, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    tok = get_token()
    out: dict = {"token_len": len(tok), "user": {}, "eod": {}, "options_endpoints": []}
    try:
        info = user_info()
        out["user"] = {
            k: info.get(k)
            for k in (
                "subscriptionType",
                "subscriptionMode",
                "dailyRateLimit",
                "apiRequests",
                "apiRequestsDate",
                "name",
                "email",
            )
        }
    except Exception as e:
        out["user"] = {"error": str(e)}
        print(json.dumps(out, indent=2))
        return 2

    for t in ("SPY", "VIX", "QQQ"):
        try:
            df = fetch_eod(t, start="2020-01-02", end="2020-01-15")
            out["eod"][t] = {
                "ok": not df.empty,
                "rows": int(len(df)),
                "label": "eodhd_eod",
                "first": str(df["date"].iloc[0]) if not df.empty else None,
                "last": str(df["date"].iloc[-1]) if not df.empty else None,
            }
        except Exception as e:
            out["eod"][t] = {"ok": False, "error": str(e)}

    # Three UnicornBay endpoints from EODHD docs
    probes = [
        (
            "underlying-symbols",
            f"{BASE}/mp/unicornbay/options/underlying-symbols?"
            + urlencode({"api_token": tok, "page[limit]": "3"}),
        ),
        (
            "contracts SPY",
            f"{BASE}/mp/unicornbay/options/contracts?"
            + urlencode(
                {
                    "api_token": tok,
                    "filter[underlying_symbol]": "SPY",
                    "page[limit]": "3",
                    "compact": "1",
                }
            ),
        ),
        (
            "eod SPY",
            f"{BASE}/mp/unicornbay/options/eod?"
            + urlencode(
                {
                    "api_token": tok,
                    "filter[underlying_symbol]": "SPY",
                    "page[limit]": "3",
                    "compact": "1",
                }
            ),
        ),
    ]
    for name, url in probes:
        out["options_endpoints"].append(_try(name, url))

    opts_ok = any(x.get("ok") for x in out["options_endpoints"])
    out["summary"] = {
        "eod_underlyings": "OK" if all(v.get("ok") for v in out["eod"].values()) else "PARTIAL",
        "unicornbay_options": "OK" if opts_ok else "NOT_SUBSCRIBED_OR_FAILED",
        "can_claim_real_option_marks": bool(opts_ok),
        "next_step_if_403": (
            "Subscribe US Stock Options API: https://eodhd.com/lp/us-stock-options-api "
            "(marketplace add-on ~29.99/mo). Then wire options/eod into run_options_strategy "
            "and set CHAIN_PRICING_ENGINE_AVAILABLE=True only when that path is live."
        ),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0 if out["summary"]["eod_underlyings"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
