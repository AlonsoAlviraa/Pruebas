"""Deep-ish Yahoo fundamentals (no ANTIGUO dependency) → data/{T}_fundamentals.csv.

Note: free Yahoo timeseries often returns only ~4–5 quarters / ~4 annuals.
EODHD Fundamentals (403 on current plan) is preferred for multi-year OOS.

Schema: as_of, period, eps, revenue, net_income, available_at, source=yahoo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.features import list_tickers  # noqa: E402

ENDPOINT = (
    "https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}"
)
UA = "Mozilla/5.0 (compatible; trad-research/1.0)"


def _request(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _series_to_map(result: list, name_substr: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for entry in result:
        meta = entry.get("meta") or {}
        types = meta.get("type") or []
        if not types:
            continue
        series_name = types[0]
        if name_substr.lower() not in series_name.lower():
            continue
        values = entry.get(series_name) or []
        for payload in values:
            if not isinstance(payload, dict):
                continue
            as_of = payload.get("asOfDate")
            if not as_of:
                continue
            raw = payload.get("reportedValue")
            if isinstance(raw, dict):
                raw = raw.get("raw")
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            out[str(as_of)[:10]] = val
    return out


def fetch_yahoo_fundamentals(ticker: str, *, lag_days: int = 45) -> pd.DataFrame:
    now = int(datetime.now(tz=timezone.utc).timestamp())
    base = ENDPOINT.format(ticker=ticker)
    # quarterly + annual
    types = (
        "quarterlyDilutedEPS,quarterlyTotalRevenue,"
        "annualDilutedEPS,annualTotalRevenue"
    )
    params = {
        "type": types,
        "period1": "0",
        "period2": str(now),
        "lang": "en-US",
        "region": "US",
    }
    data = _request(base + "?" + urlencode(params))
    result = (data.get("timeseries") or {}).get("result") or []
    q_eps = _series_to_map(result, "quarterlyDilutedEPS")
    q_rev = _series_to_map(result, "quarterlyTotalRevenue")
    # annual names contain annualDilutedEPS / annualTotalRevenue
    a_eps = {k: v for k, v in _series_to_map(result, "DilutedEPS").items() if k in _series_to_map(result, "annualDilutedEPS") or True}
    # cleaner annual extract
    a_eps = {}
    a_rev = {}
    for entry in result:
        meta = entry.get("meta") or {}
        types_l = meta.get("type") or []
        if not types_l:
            continue
        sn = types_l[0]
        values = entry.get(sn) or []
        for payload in values:
            if not isinstance(payload, dict):
                continue
            as_of = str(payload.get("asOfDate") or "")[:10]
            if not as_of:
                continue
            raw = payload.get("reportedValue")
            if isinstance(raw, dict):
                raw = raw.get("raw")
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if sn == "annualDilutedEPS":
                a_eps[as_of] = val
            elif sn == "annualTotalRevenue":
                a_rev[as_of] = val
            elif sn == "quarterlyDilutedEPS":
                q_eps[as_of] = val
            elif sn == "quarterlyTotalRevenue":
                q_rev[as_of] = val

    lag = pd.Timedelta(days=int(lag_days))
    rows: List[Dict[str, Any]] = []
    for d, eps in q_eps.items():
        as_of = pd.Timestamp(d, tz="UTC")
        rows.append(
            {
                "as_of": as_of,
                "period": "Q",
                "eps": eps,
                "revenue": q_rev.get(d, float("nan")),
                "net_income": float("nan"),
                "available_at": as_of + lag,
                "source": "yahoo",
            }
        )
    for d, eps in a_eps.items():
        as_of = pd.Timestamp(d, tz="UTC")
        rows.append(
            {
                "as_of": as_of,
                "period": "A",
                "eps": eps,
                "revenue": a_rev.get(d, float("nan")),
                "net_income": float("nan"),
                "available_at": as_of + lag,
                "source": "yahoo",
            }
        )
    # revenue-only quarters
    for d, rev in q_rev.items():
        if d in q_eps:
            continue
        as_of = pd.Timestamp(d, tz="UTC")
        rows.append(
            {
                "as_of": as_of,
                "period": "Q",
                "eps": float("nan"),
                "revenue": rev,
                "net_income": float("nan"),
                "available_at": as_of + lag,
                "source": "yahoo",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["as_of", "period", "eps", "revenue", "net_income", "available_at", "source"]
        )
    df = pd.DataFrame(rows).sort_values(["period", "as_of"]).reset_index(drop=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker-file", type=Path, default=ROOT / "good_tickers_filtrados.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lag-days", type=int, default=45)
    ap.add_argument("--sleep", type=float, default=0.12)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    limit = None if int(args.limit) <= 0 else int(args.limit)
    tickers = list_tickers(Path(args.ticker_file), data_root, limit=limit)
    ok = 0
    depths = []
    errors: Dict[str, str] = {}
    for i, t in enumerate(tickers, 1):
        fp = data_root / f"{t}_fundamentals.csv"
        if fp.is_file() and not args.force:
            try:
                old = pd.read_csv(fp)
                nq = int((old.get("period") == "Q").sum()) if "period" in old.columns else len(old)
                if nq >= 4 and "available_at" in old.columns:
                    ok += 1
                    depths.append({"ticker": t, "n_rows": len(old), "source": "cache"})
                    continue
            except Exception:
                pass
        try:
            df = fetch_yahoo_fundamentals(t, lag_days=int(args.lag_days))
            if df.empty:
                errors[t] = "empty"
                continue
            df.to_csv(fp, index=False)
            ok += 1
            nq = int((df["period"] == "Q").sum())
            na = int((df["period"] == "A").sum())
            depths.append(
                {
                    "ticker": t,
                    "n_q": nq,
                    "n_a": na,
                    "min_as_of": str(df["as_of"].min()),
                    "max_as_of": str(df["as_of"].max()),
                }
            )
            if i % 20 == 0:
                print(f"  {i}/{len(tickers)} ok={ok}", flush=True)
        except Exception as e:
            errors[t] = f"{type(e).__name__}:{e}"
        time.sleep(float(args.sleep))

    nq5 = sum(1 for d in depths if d.get("n_q", d.get("n_rows", 0)) >= 5)
    cov = {
        "provider": "yahoo_urllib",
        "note": "EODHD fundamentals 403; Yahoo free depth often ~5Q + ~4 annual — prefer EODHD fund add-on for long OOS.",
        "n_requested": len(tickers),
        "n_ok": ok,
        "n_with_ge_5_quarters": nq5,
        "n_errors": len(errors),
        "depths_sample": depths[:20],
        "error_sample": dict(list(errors.items())[:15]),
    }
    outp = ROOT / "reports" / "yahoo_fundamentals_coverage.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(cov, indent=2), encoding="utf-8")
    print(f"OK={ok}/{len(tickers)} ≥5Q≈{nq5} → {outp}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
