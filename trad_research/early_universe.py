"""Build causal early-window L0 ticker lists from available history coverage.

Modern highvol80 cache often starts ~2014-06 — cannot support 2005–2014 OOS.
This module selects names with long EOD history for S1b early falsification.

Causality
---------
* Membership as-of ``as_of`` (default = first OOS start): ticker must have history
  starting on/before ``history_start_need`` and still be listed on ``as_of``
  (last bar ≥ as_of). Do **not** require survival past the OOS end (no future
  survivorship filter on last_need after OOS).
* ADV$ window ends **strictly before** first OOS (``adv_end`` < as_of).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Exclude pure index/ETF proxies from stock L0 (still usable as benches)
DEFAULT_EXCLUDE = frozenset(
    {
        "SPY",
        "QQQ",
        "IVV",
        "IWM",
        "DIA",
        "VOO",
        "VTI",
        "XLB",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLU",
        "XLV",
        "XLY",
        "XLC",
        "XLRE",
    }
)


def _history_path(data_root: Path, ticker: str) -> Path:
    return Path(data_root) / f"{ticker}_history.csv"


def ticker_date_span(data_root: Path, ticker: str) -> Optional[Tuple[str, str, int]]:
    p = _history_path(data_root, ticker)
    if not p.is_file():
        return None
    try:
        d0 = pd.read_csv(p, usecols=[0])
        if d0.empty:
            return None
        first = str(d0.iloc[0, 0])[:10]
        last = str(d0.iloc[-1, 0])[:10]
        return first, last, len(d0)
    except Exception:
        return None


def mean_dollar_volume(
    data_root: Path,
    ticker: str,
    *,
    start: str = "2009-01-01",
    end: str = "2009-12-31",
) -> float:
    p = _history_path(data_root, ticker)
    if not p.is_file():
        return 0.0
    try:
        df = pd.read_csv(p)
        df.columns = [str(c).lower() for c in df.columns]
        dcol = df.columns[0]
        df[dcol] = pd.to_datetime(df[dcol], utc=True, errors="coerce")
        if "close" not in df.columns or "volume" not in df.columns:
            return 0.0
        m = (df[dcol] >= start) & (df[dcol] <= end)
        sub = df.loc[m]
        if len(sub) < 40:
            return 0.0
        return float((sub["close"] * sub["volume"]).mean())
    except Exception:
        return 0.0


def _default_adv_window(as_of: str) -> Tuple[str, str]:
    """ADV ends day before as_of; starts ~2y earlier (causal liquidity rank)."""
    as_ts = pd.Timestamp(as_of)
    end = (as_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (as_ts - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    return start, end


def build_early_window_universe(
    data_root: Path | str = "data",
    *,
    as_of: str = "2010-01-01",
    history_start_need: str = "2005-06-01",
    min_adv_usd: float = 50_000.0,
    max_names: int = 40,
    exclude: Optional[Sequence[str]] = None,
    adv_window_start: Optional[str] = None,
    adv_window_end: Optional[str] = None,
    # Legacy kwargs (ignored for membership when as_of is set; kept for CLI compat)
    first_need: Optional[str] = None,
    last_need: Optional[str] = None,
) -> List[str]:
    """Causal L0 as-of first OOS: listed on as_of, history from history_start_need, ADV pre-OOS.

    Parameters
    ----------
    as_of :
        First OOS date (inclusive). Ticker must have last bar ≥ as_of (listed then).
        No requirement to survive past OOS end.
    history_start_need :
        Require first bar on/before this date (warm-up / train depth).
    adv_window_* :
        If None, ADV uses [as_of-2y, as_of-1d] strictly before OOS.
    last_need :
        **Deprecated / diagnostic only.** If provided and after as_of, it is
        **not** used for membership (would be look-ahead survivorship). Logged
        in meta via ensure_* only when explicitly requested as diagnostic.
    """
    data_root = Path(data_root)
    ban = set(DEFAULT_EXCLUDE)
    if exclude:
        ban |= {str(x).upper() for x in exclude}

    hist_need = first_need or history_start_need
    adv_start, adv_end = (
        (adv_window_start, adv_window_end)
        if adv_window_start and adv_window_end
        else _default_adv_window(as_of)
    )
    # Hard causality: ADV end must be < as_of
    if adv_end >= as_of:
        adv_end = (pd.Timestamp(as_of) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    rows: List[Tuple[str, float]] = []
    for p in data_root.glob("*_history.csv"):
        t = p.name.replace("_history.csv", "")
        if "." in t or t.upper() in ban:
            continue
        span = ticker_date_span(data_root, t)
        if span is None:
            continue
        first, last, _n = span
        # Causal membership: history deep enough + still listed as-of first OOS
        if first > hist_need:
            continue
        if last < as_of:
            continue
        # Do not require last >= last_need (post-OOS survivorship)
        adv = mean_dollar_volume(data_root, t, start=adv_start, end=adv_end)
        if adv < float(min_adv_usd):
            continue
        rows.append((t.upper(), adv))

    rows.sort(key=lambda x: -x[1])
    return [t for t, _ in rows[: int(max_names)]]


def build_early_window_universe_meta(
    data_root: Path | str = "data",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Same as build_early_window_universe but returns tickers + causality meta."""
    as_of = kwargs.get("as_of", "2010-01-01")
    adv_s = kwargs.get("adv_window_start")
    adv_e = kwargs.get("adv_window_end")
    if not adv_s or not adv_e:
        adv_s, adv_e = _default_adv_window(str(as_of))
    if str(adv_e) >= str(as_of):
        adv_e = (pd.Timestamp(as_of) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tickers = build_early_window_universe(data_root, **kwargs)
    return {
        "tickers": tickers,
        "as_of": as_of,
        "adv_window_start": adv_s,
        "adv_window_end": adv_e,
        "history_start_need": kwargs.get("history_start_need")
        or kwargs.get("first_need")
        or "2005-06-01",
        "no_post_oos_survivorship": True,
        "last_need_ignored_for_membership": kwargs.get("last_need"),
    }


def write_universe_file(tickers: Sequence[str], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tickers) + ("\n" if tickers else ""), encoding="utf-8")
    return path


def _config_fingerprint(
    *,
    as_of: str,
    history_start_need: str,
    max_names: int,
    min_adv_usd: float = 50_000.0,
) -> str:
    import hashlib
    import json

    payload = {
        "as_of": str(as_of),
        "history_start_need": str(history_start_need),
        "max_names": int(max_names),
        "min_adv_usd": float(min_adv_usd),
        "schema": "early_l0_v2_causal",
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _meta_path(universe_path: Path) -> Path:
    return universe_path.with_suffix(universe_path.suffix + ".meta.json")


def ensure_early_universe_file(
    path: Path | str = "universe_early_longhist.txt",
    *,
    data_root: Path | str = "data",
    max_names: int = 40,
    rebuild: bool = False,
    as_of: str = "2010-01-01",
    history_start_need: str = "2005-06-01",
    min_adv_usd: float = 50_000.0,
) -> Path:
    """Create early L0 file if missing, rebuild=True, or config fingerprint differs.

    Sidecar ``*.meta.json`` stores as_of / history_start_need / max_names hash so a
    stale pre-fix file is never silently reused under a new causal config.
    """
    import json

    path = Path(path)
    fp = _config_fingerprint(
        as_of=as_of,
        history_start_need=history_start_need,
        max_names=max_names,
        min_adv_usd=min_adv_usd,
    )
    meta_p = _meta_path(path)
    if path.is_file() and meta_p.is_file() and not rebuild:
        try:
            old = json.loads(meta_p.read_text(encoding="utf-8"))
            if old.get("fingerprint") == fp:
                return path
        except Exception:
            pass
    # rebuild required
    tickers = build_early_window_universe(
        data_root,
        as_of=as_of,
        history_start_need=history_start_need,
        max_names=max_names,
        min_adv_usd=min_adv_usd,
    )
    write_universe_file(tickers, path)
    meta = {
        "fingerprint": fp,
        "as_of": as_of,
        "history_start_need": history_start_need,
        "max_names": max_names,
        "min_adv_usd": min_adv_usd,
        "n_tickers": len(tickers),
        "schema": "early_l0_v2_causal",
    }
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
