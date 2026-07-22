"""Yahoo Finance options chain snapshot (today / near-term only).

LABEL: successful fetches use ``data_label='yahoo_chain'``.
Failures raise or return an explicit error result — **never** invent chain
quotes and label them as real.

Virtual capital / research only; free unofficial endpoint, may be rate-limited.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

YAHOO_OPTIONS = (
    "https://query2.finance.yahoo.com/v7/finance/options/{ticker}"
)
YAHOO_OPTIONS_ALT = (
    "https://query1.finance.yahoo.com/v7/finance/options/{ticker}"
)
YAHOO_OPTIONS_EXP = (
    "https://query2.finance.yahoo.com/v7/finance/options/{ticker}?date={exp_ts}"
)


class YahooChainError(Exception):
    """Raised when chain cannot be fetched or parsed (no silent fake data)."""


@dataclass
class OptionQuote:
    contract_symbol: str
    strike: float
    expiry: str  # ISO date
    option_type: str  # call | put
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    mid: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    implied_volatility: Optional[float] = None
    in_the_money: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChainSnapshot:
    """Point-in-time options chain; never fabricate on failure."""

    underlying: str
    as_of_utc: str
    spot: Optional[float]
    expirations: List[str] = field(default_factory=list)
    calls: List[OptionQuote] = field(default_factory=list)
    puts: List[OptionQuote] = field(default_factory=list)
    data_label: str = "yahoo_chain"
    source: str = "yahoo_v7_options"
    notes: List[str] = field(default_factory=list)
    ok: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "underlying": self.underlying,
            "as_of_utc": self.as_of_utc,
            "spot": self.spot,
            "expirations": list(self.expirations),
            "n_calls": len(self.calls),
            "n_puts": len(self.puts),
            "calls": [c.to_dict() for c in self.calls[:50]],
            "puts": [p.to_dict() for p in self.puts[:50]],
            "data_label": self.data_label,
            "source": self.source,
            "notes": list(self.notes),
            "ok": self.ok,
            "error": self.error,
            "mode": "paper",
            "capital_label": "VIRTUAL",
        }


def _http_get(url: str, *, timeout: int = 30, retries: int = 2) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": DEFAULT_UA,
                    "Accept": "application/json,*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (URLError, HTTPError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise YahooChainError(f"HTTP GET failed {url}: {last_err}")


def _mid(bid: Optional[float], ask: Optional[float], last: Optional[float]) -> Optional[float]:
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return 0.5 * (float(bid) + float(ask))
    if last is not None and last > 0:
        return float(last)
    if bid is not None and bid > 0:
        return float(bid)
    if ask is not None and ask > 0:
        return float(ask)
    return None


def _parse_contract(row: Dict[str, Any], option_type: str) -> Optional[OptionQuote]:
    try:
        strike = float(row.get("strike"))
    except (TypeError, ValueError):
        return None
    exp_ts = row.get("expiration")
    try:
        exp_iso = datetime.fromtimestamp(int(exp_ts), tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        exp_iso = str(row.get("expiration") or "")

    bid = row.get("bid")
    ask = row.get("ask")
    last = row.get("lastPrice")
    try:
        bid_f = float(bid) if bid is not None else None
    except (TypeError, ValueError):
        bid_f = None
    try:
        ask_f = float(ask) if ask is not None else None
    except (TypeError, ValueError):
        ask_f = None
    try:
        last_f = float(last) if last is not None else None
    except (TypeError, ValueError):
        last_f = None

    iv = row.get("impliedVolatility")
    try:
        iv_f = float(iv) if iv is not None else None
    except (TypeError, ValueError):
        iv_f = None

    return OptionQuote(
        contract_symbol=str(row.get("contractSymbol") or ""),
        strike=strike,
        expiry=exp_iso,
        option_type=option_type,
        bid=bid_f,
        ask=ask_f,
        last=last_f,
        mid=_mid(bid_f, ask_f, last_f),
        volume=_optional_float(row.get("volume")),
        open_interest=_optional_float(row.get("openInterest")),
        implied_volatility=iv_f,
        in_the_money=bool(row.get("inTheMoney")) if row.get("inTheMoney") is not None else None,
    )


def _optional_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_chain_payload(j: Dict[str, Any], ticker: str) -> ChainSnapshot:
    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = (j.get("optionChain") or {}).get("result") or []
    if not result:
        err = (j.get("optionChain") or {}).get("error")
        raise YahooChainError(f"Empty optionChain for {ticker}: {err}")

    res0 = result[0]
    quote = res0.get("quote") or {}
    spot = quote.get("regularMarketPrice") or quote.get("postMarketPrice")
    try:
        spot_f = float(spot) if spot is not None else None
    except (TypeError, ValueError):
        spot_f = None

    exp_ts_list = res0.get("expirationDates") or []
    expirations: List[str] = []
    for ts in exp_ts_list:
        try:
            expirations.append(
                datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()
            )
        except (TypeError, ValueError, OSError):
            continue

    options_blocks = res0.get("options") or []
    calls: List[OptionQuote] = []
    puts: List[OptionQuote] = []
    for block in options_blocks:
        for row in block.get("calls") or []:
            q = _parse_contract(row, "call")
            if q is not None:
                calls.append(q)
        for row in block.get("puts") or []:
            q = _parse_contract(row, "put")
            if q is not None:
                puts.append(q)

    if not calls and not puts:
        raise YahooChainError(f"No contracts parsed for {ticker}")

    return ChainSnapshot(
        underlying=ticker.upper(),
        as_of_utc=as_of,
        spot=spot_f,
        expirations=expirations,
        calls=calls,
        puts=puts,
        data_label="yahoo_chain",
        source="yahoo_v7_options",
        notes=[
            "Real Yahoo options chain snapshot (unofficial free endpoint).",
            "For today/near-term validation only — not historical proxy_bs marks.",
        ],
        ok=True,
        error=None,
    )


def fetch_yahoo_option_chain(
    ticker: str,
    *,
    expiration: Optional[date] = None,
    timeout: int = 30,
    raise_on_error: bool = False,
) -> ChainSnapshot:
    """
    Fetch current options chain for ``ticker`` from Yahoo.

    On failure: if ``raise_on_error``, raise ``YahooChainError``;
    otherwise return ``ChainSnapshot(ok=False, data_label='yahoo_chain_failed', ...)``
    — never returns synthetic strikes labeled as real.
    """
    t = ticker.upper().strip()
    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    urls: List[str] = []
    if expiration is not None:
        exp_ts = int(
            datetime(expiration.year, expiration.month, expiration.day, tzinfo=timezone.utc).timestamp()
        )
        urls.append(YAHOO_OPTIONS_EXP.format(ticker=t, exp_ts=exp_ts))
    urls.extend(
        [
            YAHOO_OPTIONS.format(ticker=t),
            YAHOO_OPTIONS_ALT.format(ticker=t),
        ]
    )

    last_err: Optional[str] = None
    for url in urls:
        try:
            raw = _http_get(url, timeout=timeout)
            j = json.loads(raw.decode("utf-8", errors="replace"))
            return _parse_chain_payload(j, t)
        except YahooChainError as e:
            last_err = str(e)
            logger.warning("Yahoo chain parse fail %s: %s", t, e)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            last_err = f"parse error: {e}"
            logger.warning("Yahoo chain JSON fail %s: %s", t, e)
        except Exception as e:  # network etc. already wrapped; keep safety
            last_err = str(e)
            logger.warning("Yahoo chain fail %s: %s", t, e)

    msg = last_err or "unknown failure"
    if raise_on_error:
        raise YahooChainError(msg)
    return ChainSnapshot(
        underlying=t,
        as_of_utc=as_of,
        spot=None,
        expirations=[],
        calls=[],
        puts=[],
        data_label="yahoo_chain_failed",
        source="yahoo_v7_options",
        notes=[
            "Chain fetch failed — no synthetic contracts invented.",
            "Historical marks remain proxy_bs if used elsewhere.",
        ],
        ok=False,
        error=msg,
    )


def summarize_chain_vs_proxy(
    snap: ChainSnapshot,
    *,
    otm_pct: float = 0.05,
    side: str = "put",
) -> Dict[str, Any]:
    """
    Lightweight today-only validation: nearest OTM strike mid vs spot.

    Does not rewrite historical proxy_bs backtests.
    """
    if not snap.ok or snap.spot is None or snap.spot <= 0:
        return {
            "ok": False,
            "data_label": snap.data_label,
            "error": snap.error or "no spot",
            "underlying": snap.underlying,
        }
    spot = float(snap.spot)
    target = spot * (1.0 - abs(otm_pct)) if side == "put" else spot * (1.0 + abs(otm_pct))
    book = snap.puts if side == "put" else snap.calls
    if not book:
        return {
            "ok": False,
            "data_label": snap.data_label,
            "error": f"no {side}s",
            "underlying": snap.underlying,
            "spot": spot,
        }
    nearest = min(book, key=lambda q: abs(q.strike - target))
    return {
        "ok": True,
        "data_label": "yahoo_chain",
        "underlying": snap.underlying,
        "spot": spot,
        "side": side,
        "target_otm_pct": otm_pct,
        "nearest_strike": nearest.strike,
        "nearest_mid": nearest.mid,
        "nearest_iv": nearest.implied_volatility,
        "expiry": nearest.expiry,
        "n_expirations": len(snap.expirations),
        "n_contracts": len(book),
        "as_of_utc": snap.as_of_utc,
    }
