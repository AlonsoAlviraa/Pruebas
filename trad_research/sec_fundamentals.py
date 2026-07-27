"""Free SEC EDGAR companyfacts → quarterly fundamentals (PIT via filed date).

No API key. Requires User-Agent per SEC fair access.
Design: plan v3 growth 50-100 free; docs/design 2026-07-24 growth.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_UA = "TRAD Research Bot trad-research@local (educational research)"

# Prefer diluted EPS; cascade revenue tags (XBRL naming varies by era/issuer)
EPS_TAGS = (
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
    "EarningsPerShareBasicAndDiluted",
)
REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
    "SalesRevenueGoodsNet",
    "RevenueNotFromContractWithCustomer",
)
NET_INCOME_TAGS = (
    "NetIncomeLoss",
    "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
)

_FRAME_Q = re.compile(r"^CY(\d{4})Q([1-4])$")


def _headers(user_agent: Optional[str] = None) -> Dict[str, str]:
    return {
        "User-Agent": (user_agent or DEFAULT_UA).strip(),
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov" if "data.sec.gov" in "" else "www.sec.gov",
    }


def http_get_json(url: str, *, user_agent: Optional[str] = None, timeout: int = 60) -> Any:
    ua = (user_agent or DEFAULT_UA).strip()
    # Host header must match URL host
    host = "data.sec.gov" if "data.sec.gov" in url else "www.sec.gov"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": ua,
            "Accept-Encoding": "gzip, deflate",
            "Host": host,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        # urllib may auto-decompress gzip depending on build; try both
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            import gzip

            text = gzip.decompress(raw).decode("utf-8")
        return json.loads(text)


def load_ticker_cik_map(
    *,
    cache_path: Optional[Path] = None,
    user_agent: Optional[str] = None,
    force: bool = False,
) -> Dict[str, str]:
    """Return {TICKER: zero-padded 10-digit CIK}."""
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.is_file() and not force:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data:
            return {str(k).upper(): str(v).zfill(10) for k, v in data.items()}

    payload = http_get_json(SEC_TICKERS_URL, user_agent=user_agent)
    # shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}, ...}
    out: Dict[str, str] = {}
    if isinstance(payload, dict):
        for _, row in payload.items():
            if not isinstance(row, dict):
                continue
            t = str(row.get("ticker") or "").upper().strip()
            cik = row.get("cik_str")
            if t and cik is not None:
                out[t] = str(int(cik)).zfill(10)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out, indent=0), encoding="utf-8")
    return out


def fetch_companyfacts(
    cik: str,
    *,
    cache_dir: Optional[Path] = None,
    user_agent: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    cik10 = str(int(str(cik).lstrip("0") or "0")).zfill(10)
    cache_dir = Path(cache_dir) if cache_dir else None
    fp = (cache_dir / f"CIK{cik10}.json") if cache_dir else None
    if fp is not None and fp.is_file() and not force:
        return json.loads(fp.read_text(encoding="utf-8"))
    url = SEC_FACTS_URL.format(cik=cik10)
    # companyfacts lives on data.sec.gov
    ua = user_agent or DEFAULT_UA
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": ua,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = resp.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            import gzip

            text = gzip.decompress(raw).decode("utf-8")
        data = json.loads(text)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(data), encoding="utf-8")  # type: ignore[union-attr]
    return data


def _unit_rows(concept: Dict[str, Any]) -> List[Dict[str, Any]]:
    units = concept.get("units") or {}
    rows: List[Dict[str, Any]] = []
    for unit_name, vals in units.items():
        if not isinstance(vals, list):
            continue
        for v in vals:
            if not isinstance(v, dict):
                continue
            r = dict(v)
            r["_unit"] = unit_name
            rows.append(r)
    return rows


def _is_quarter_row(row: Dict[str, Any]) -> bool:
    """Prefer pure quarterly frames; reject multi-quarter YTD blobs when possible."""
    frame = str(row.get("frame") or "")
    if _FRAME_Q.match(frame):
        return True
    # Without frame: duration ~1 quarter by start/end
    try:
        start = pd.Timestamp(row.get("start"))
        end = pd.Timestamp(row.get("end"))
        days = (end - start).days
        if 70 <= days <= 100:
            return True
    except Exception:
        pass
    # 10-Q with fp Q1-Q4 sometimes omit frame
    form = str(row.get("form") or "")
    fp = str(row.get("fp") or "")
    if form in ("10-Q", "10-Q/A") and fp in ("Q1", "Q2", "Q3", "Q4"):
        return True
    return False


def _pick_tag_rows(facts_gaap: Dict[str, Any], tags: Sequence[str]) -> List[Dict[str, Any]]:
    for tag in tags:
        if tag not in facts_gaap:
            continue
        rows = _unit_rows(facts_gaap[tag])
        qrows = [r for r in rows if _is_quarter_row(r) and r.get("val") is not None]
        if len(qrows) >= 8:
            return qrows
        if qrows:
            # keep looking for a richer tag
            best = qrows
        else:
            best = []
    # fallback: first tag any rows
    for tag in tags:
        if tag in facts_gaap:
            rows = _unit_rows(facts_gaap[tag])
            qrows = [r for r in rows if _is_quarter_row(r) and r.get("val") is not None]
            if qrows:
                return qrows
    return []


def parse_companyfacts_to_quarterly(payload: Dict[str, Any]) -> pd.DataFrame:
    """Normalize companyfacts JSON → quarterly rows with available_at=filed."""
    empty = pd.DataFrame(
        columns=["as_of", "period", "eps", "revenue", "net_income", "available_at", "source"]
    )
    if not isinstance(payload, dict):
        return empty
    gaap = (payload.get("facts") or {}).get("us-gaap") or {}
    if not gaap:
        return empty

    eps_rows = _pick_tag_rows(gaap, EPS_TAGS)
    rev_rows = _pick_tag_rows(gaap, REVENUE_TAGS)
    ni_rows = _pick_tag_rows(gaap, NET_INCOME_TAGS)

    def key_of(r: Dict[str, Any]) -> str:
        # end date is period end
        return str(r.get("end") or "")[:10]

    by_end: Dict[str, Dict[str, Any]] = {}

    def ingest(rows: List[Dict[str, Any]], field: str) -> None:
        # prefer earliest filed for first-reported (PIT); if same end, min filed
        best: Dict[str, Tuple[pd.Timestamp, float, pd.Timestamp]] = {}
        for r in rows:
            end = key_of(r)
            if not end:
                continue
            try:
                filed = pd.Timestamp(r.get("filed"), tz="UTC")
            except Exception:
                continue
            try:
                val = float(r["val"])
            except (TypeError, ValueError, KeyError):
                continue
            as_of = pd.Timestamp(end, tz="UTC")
            prev = best.get(end)
            if prev is None or filed < prev[0]:
                best[end] = (filed, val, as_of)
        for end, (filed, val, as_of) in best.items():
            bucket = by_end.setdefault(
                end,
                {
                    "as_of": as_of,
                    "period": "Q",
                    "eps": float("nan"),
                    "revenue": float("nan"),
                    "net_income": float("nan"),
                    "available_at": filed,
                    "source": "sec",
                },
            )
            bucket[field] = val
            # available_at = max filed among fields so all three known
            prev_av = pd.Timestamp(bucket["available_at"])
            if filed > prev_av:
                bucket["available_at"] = filed

    ingest(eps_rows, "eps")
    ingest(rev_rows, "revenue")
    ingest(ni_rows, "net_income")

    if not by_end:
        return empty
    df = pd.DataFrame(list(by_end.values())).sort_values("as_of").reset_index(drop=True)
    # drop rows with neither eps nor revenue nor NI
    mask = df[["eps", "revenue", "net_income"]].notna().any(axis=1)
    df = df.loc[mask].reset_index(drop=True)
    return df


def fundamentals_from_cik(
    cik: str,
    *,
    cache_dir: Optional[Path] = None,
    user_agent: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    payload = fetch_companyfacts(
        cik, cache_dir=cache_dir, user_agent=user_agent, force=force
    )
    return parse_companyfacts_to_quarterly(payload)


def download_ticker_fundamentals(
    ticker: str,
    *,
    cik_map: Dict[str, str],
    data_root: Path,
    cache_dir: Path,
    user_agent: Optional[str] = None,
    force: bool = False,
    sleep_s: float = 0.2,
) -> Tuple[bool, str, int]:
    """Write data/{T}_fundamentals.csv. Returns (ok, message, n_quarters)."""
    t = ticker.upper().strip()
    cik = cik_map.get(t)
    if not cik:
        return False, "no_cik", 0
    try:
        df = fundamentals_from_cik(
            cik, cache_dir=cache_dir, user_agent=user_agent, force=force
        )
        time.sleep(sleep_s)
        if df.empty:
            return False, "empty_parse", 0
        out = Path(data_root) / f"{t}_fundamentals.csv"
        df.to_csv(out, index=False)
        return True, "ok", int(len(df))
    except Exception as e:
        return False, f"{type(e).__name__}:{e}", 0
