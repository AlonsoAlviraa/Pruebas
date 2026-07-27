"""Causal crash / oversold entry overlays for highvol-style strategies.

Research-only: index RSI (Wilder), drawdown-from-peak, optional SMA distance.
Fail-closed: only bars ≤ t; missing index history → crash flag False.

Does not invent OPRA/options. Does not claim live edge.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trad_research.features import _wilder_rsi


@dataclass
class CrashEntryConfig:
    """Overlay knobs applied on top of a base turbo_highvol-style strategy.

    When ``enabled`` is False, all crash maps are empty/off (no behaviour change).
    """

    enabled: bool = False
    # rsi | dd | rsi_or_dd | rsi_and_dd | rsi_recover
    mode: str = "rsi"
    # Prefer SPY then QQQ; combine with any/all
    index_names: Tuple[str, ...] = ("SPY", "QQQ")
    combine: str = "any"  # any | all | first (first available index only)
    rsi_period: int = 14
    rsi_threshold: float = 30.0
    dd_threshold: float = -0.15  # e.g. -0.15 = 15% off peak
    # rsi_recover: crash when RSI < thr AND RSI rising vs prior bar
    require_rsi_rising: bool = False
    # Optional: also require index below SMA (causal)
    require_below_sma50: bool = False
    require_below_sma200: bool = False

    # --- Actions when crash flag is True ---
    relax_regime: bool = True  # allow new entries even if hard regime is risk-off
    crash_min_confidence: Optional[float] = 0.22  # lower conf threshold for entries
    crash_relax_trend: bool = True  # ignore hard close>SMA50 on crash days
    crash_min_dist_sma200: Optional[float] = -0.25  # looser SMA200 distance
    crash_score_boost: float = 1.15  # multiply score for ranking on crash days
    force_soft_hard_regime: bool = False  # if True, soft_hard_regime during crash only

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WinRateFilterConfig:
    """Research filters aimed at lifting win rate without rewriting the engine.

    ATR policy (SSOT with BacktestConfig.max_atr_pct_entry):
      - **Non-crash days:** apply ``max_atr_pct_tight`` (signal) and entry gate.
      - **Crash days:** skip tight ATR so strategy ``max_atr_pct`` (looser) remains.

    soft_trend_non_crash:
      When True, hard close>SMA50 is **not** applied at signal gen; this filter
      applies SMA50|SMA20 on non-crash days only (true soft replacement, not a no-op).
    """

    # Skip entries when atr_norm above this on non-crash days (tighter than max_atr_pct)
    max_atr_pct_tight: Optional[float] = None
    # Require meta confidence (only if meta model attached on cfg)
    min_meta_conf: Optional[float] = None
    # Days to block re-entry on same ticker after a hard_stop exit
    hard_stop_cooldown_days: int = 0
    # When NOT in crash mode: soft trend (SMA50 | SMA20) *replaces* hard SMA50
    soft_trend_non_crash: bool = False
    # Raise min p_buy / score floor outside crash (None = no change). Wired in filters.
    non_crash_min_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IndexCrashMetrics:
    """Per-index causal metrics; dates are UTC-normalized timestamps."""

    name: str
    dates: pd.DatetimeIndex
    close: np.ndarray
    rsi: np.ndarray
    dd_from_peak: np.ndarray
    below_sma50: np.ndarray
    below_sma200: np.ndarray
    rsi_rising: np.ndarray

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": self.dates,
                "close": self.close,
                "rsi": self.rsi,
                "dd_from_peak": self.dd_from_peak,
                "below_sma50": self.below_sma50,
                "below_sma200": self.below_sma200,
                "rsi_rising": self.rsi_rising,
                "index": self.name,
            }
        )


def load_index_ohlcv(
    data_root: Path,
    name: str,
) -> Optional[pd.DataFrame]:
    """Load ``{name}_history.csv``; return sorted date/close frame or None."""
    p = Path(data_root) / f"{name}_history.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return None
    return df


def compute_index_crash_metrics(
    close: pd.Series,
    dates: pd.Series,
    *,
    name: str = "INDEX",
    rsi_period: int = 14,
) -> IndexCrashMetrics:
    """Causal RSI, peak drawdown, SMA flags from a close series (no future bars)."""
    c = pd.to_numeric(close, errors="coerce").astype(float)
    d = pd.to_datetime(dates, utc=True)
    rsi = _wilder_rsi(c, int(rsi_period))
    # Expanding peak using only ≤ t
    peak = c.cummax()
    dd = (c / peak.replace(0.0, np.nan)) - 1.0
    sma50 = c.rolling(50, min_periods=25).mean()
    sma200 = c.rolling(200, min_periods=100).mean()
    below50 = (c < sma50).fillna(False).to_numpy(dtype=bool)
    below200 = (c < sma200).fillna(False).to_numpy(dtype=bool)
    rsi_arr = rsi.to_numpy(dtype=float)
    rsi_prev = np.roll(rsi_arr, 1)
    rsi_prev[0] = np.nan
    rising = np.isfinite(rsi_arr) & np.isfinite(rsi_prev) & (rsi_arr > rsi_prev)
    return IndexCrashMetrics(
        name=name,
        dates=pd.DatetimeIndex(d),
        close=c.to_numpy(dtype=float),
        rsi=rsi_arr,
        dd_from_peak=dd.to_numpy(dtype=float),
        below_sma50=below50,
        below_sma200=below200,
        rsi_rising=rising,
    )


def _flag_from_metrics(
    m: IndexCrashMetrics,
    cfg: CrashEntryConfig,
) -> np.ndarray:
    thr = float(cfg.rsi_threshold)
    dd_thr = float(cfg.dd_threshold)
    rsi_low = np.isfinite(m.rsi) & (m.rsi < thr)
    dd_deep = np.isfinite(m.dd_from_peak) & (m.dd_from_peak <= dd_thr)

    mode = (cfg.mode or "rsi").lower().strip()
    if mode == "rsi":
        flag = rsi_low
    elif mode == "dd":
        flag = dd_deep
    elif mode == "rsi_or_dd":
        flag = rsi_low | dd_deep
    elif mode == "rsi_and_dd":
        flag = rsi_low & dd_deep
    elif mode in ("rsi_recover", "rsi_rising"):
        flag = rsi_low & m.rsi_rising
    else:
        flag = rsi_low

    if cfg.require_rsi_rising:
        flag = flag & m.rsi_rising
    if cfg.require_below_sma50:
        flag = flag & m.below_sma50
    if cfg.require_below_sma200:
        flag = flag & m.below_sma200
    return flag.astype(bool)


def build_crash_entry_map(
    data_root: Path | str,
    cfg: CrashEntryConfig,
) -> Tuple[Dict[pd.Timestamp, bool], Dict[str, Any]]:
    """Build date → crash_on map. Fail-closed: missing data → False / empty.

    Returns (map, meta) where meta documents which indices and thresholds used.
    """
    meta: Dict[str, Any] = {
        "enabled": bool(cfg.enabled),
        "mode": cfg.mode,
        "indices_used": [],
        "n_crash_days": 0,
        "config": cfg.to_dict(),
    }
    if not cfg.enabled:
        return {}, meta

    root = Path(data_root)
    metrics_list: List[IndexCrashMetrics] = []
    for name in cfg.index_names:
        raw = load_index_ohlcv(root, name)
        if raw is None:
            continue
        m = compute_index_crash_metrics(
            raw["close"],
            raw["date"],
            name=name,
            rsi_period=cfg.rsi_period,
        )
        metrics_list.append(m)
        meta["indices_used"].append(name)
        if cfg.combine == "first":
            break

    if not metrics_list:
        meta["error"] = "no_index_history"
        return {}, meta

    # Per-index flags → align on union of dates (asof prior for missing)
    flags_by_idx: List[Dict[pd.Timestamp, bool]] = []
    for m in metrics_list:
        f = _flag_from_metrics(m, cfg)
        flags_by_idx.append({d: bool(v) for d, v in zip(m.dates, f)})

    all_dates = sorted(set().union(*[set(m.keys()) for m in flags_by_idx]))
    out: Dict[pd.Timestamp, bool] = {}
    for day in all_dates:
        vals = []
        for fmap in flags_by_idx:
            if day in fmap:
                vals.append(fmap[day])
            else:
                # Causal asof: last known ≤ day
                prior = [d for d in fmap if d <= day]
                if prior:
                    vals.append(fmap[max(prior)])
                else:
                    vals.append(False)  # fail-closed
        if cfg.combine == "all":
            out[day] = bool(all(vals)) if vals else False
        else:
            # any or first
            out[day] = bool(any(vals)) if vals else False

    meta["n_crash_days"] = int(sum(1 for v in out.values() if v))
    meta["n_dates"] = len(out)
    return out, meta


def crash_on_day(
    day: pd.Timestamp,
    crash_map: Optional[Dict[pd.Timestamp, bool]],
) -> bool:
    """Lookup crash flag with causal asof; fail-closed if map empty/missing."""
    if not crash_map:
        return False
    day = pd.Timestamp(day)
    if day.tzinfo is None:
        day = day.tz_localize("UTC")
    else:
        day = day.tz_convert("UTC")
    if day in crash_map:
        return bool(crash_map[day])
    prior = [d for d in crash_map if d <= day]
    if not prior:
        return False
    return bool(crash_map[max(prior)])


def apply_crash_signal_overlay(
    df: pd.DataFrame,
    base_sig: pd.Series,
    base_score: pd.Series,
    p_buy: np.ndarray,
    cfg_bt: Any,
    crash_map: Optional[Dict[pd.Timestamp, bool]],
    crash_cfg: Optional[CrashEntryConfig],
) -> Tuple[pd.Series, pd.Series]:
    """Union base signals with crash-relaxed entries on crash days only.

    Uses only columns already on ``df`` (causal features). Does not peek ahead.
    """
    if not crash_cfg or not crash_cfg.enabled or not crash_map:
        return base_sig, base_score

    sig = np.asarray(base_sig, dtype=bool).copy()
    score = np.asarray(base_score, dtype=float).copy()
    dates = pd.to_datetime(df["date"], utc=True)

    # Crash-day relaxed mask from raw ML score
    conf = (
        float(crash_cfg.crash_min_confidence)
        if crash_cfg.crash_min_confidence is not None
        else float(getattr(cfg_bt, "min_confidence", 0.3))
    )
    crash_sig = p_buy >= conf

    if crash_cfg.crash_relax_trend:
        # Soft trend: SMA50 OR causal SMA20 if present; else no hard trend gate
        close = df["close"].to_numpy(dtype=float)
        trend_ok = np.ones(len(df), dtype=bool)
        if "sma_50" in df.columns:
            sma50 = df["sma_50"].to_numpy(dtype=float)
            sma20 = pd.Series(close).rolling(20, min_periods=10).mean().to_numpy()
            trend_ok = (close > sma50) | (close > sma20)
        crash_sig = crash_sig & trend_ok
    elif bool(getattr(cfg_bt, "require_trend", True)) and "sma_50" in df.columns:
        crash_sig = crash_sig & (
            df["close"].to_numpy(dtype=float) > df["sma_50"].to_numpy(dtype=float)
        )

    dist_floor = crash_cfg.crash_min_dist_sma200
    if dist_floor is not None and "dist_sma_200" in df.columns:
        crash_sig = crash_sig & (
            df["dist_sma_200"].to_numpy(dtype=float) >= float(dist_floor)
        )

    # Crash days: use strategy max_atr_pct only (loose). Do NOT apply max_atr_pct_tight
    # / max_atr_pct_entry here — those are non-crash-only (see WinRateFilterConfig).
    if "atr_norm" in df.columns and getattr(cfg_bt, "max_atr_pct", None) is not None:
        crash_sig = crash_sig & (
            df["atr_norm"].to_numpy(dtype=float) <= float(cfg_bt.max_atr_pct)
        )

    boost = float(crash_cfg.crash_score_boost or 1.0)
    for i, day in enumerate(dates):
        if not crash_on_day(day, crash_map):
            continue
        if crash_sig[i]:
            sig[i] = True
            score[i] = max(score[i], float(p_buy[i]) * boost)
        elif boost != 1.0 and sig[i]:
            score[i] = float(score[i]) * boost

    return pd.Series(sig, index=base_sig.index), pd.Series(score, index=base_score.index)


def apply_winrate_signal_filters(
    df: pd.DataFrame,
    sig: pd.Series,
    score: pd.Series,
    wr: Optional[WinRateFilterConfig],
    crash_map: Optional[Dict[pd.Timestamp, bool]] = None,
    *,
    non_crash_only: bool = True,
    p_buy: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Apply ATR tight / soft-trend / conf floor outside crash (research WR levers).

    When ``soft_trend_non_crash`` is True, caller must skip hard SMA50 at signal gen
    so this function can admit close>SMA20 with close<SMA50 (replacement, not no-op).
    """
    if wr is None:
        return sig, score
    s = np.asarray(sig, dtype=bool).copy()
    sc = np.asarray(score, dtype=float).copy()
    n = len(df)
    dates = pd.to_datetime(df["date"], utc=True) if "date" in df.columns else None
    p_arr = None if p_buy is None else np.asarray(p_buy, dtype=float)

    def _in_crash(i: int) -> bool:
        if dates is None or not crash_map:
            return False
        return crash_on_day(dates.iloc[i], crash_map)

    if wr.max_atr_pct_tight is not None and "atr_norm" in df.columns:
        atr = df["atr_norm"].to_numpy(dtype=float)
        thr = float(wr.max_atr_pct_tight)
        for i in range(n):
            if not s[i]:
                continue
            if non_crash_only and _in_crash(i):
                continue
            if not np.isfinite(atr[i]) or atr[i] > thr:
                s[i] = False

    if wr.soft_trend_non_crash and "close" in df.columns:
        close = df["close"].to_numpy(dtype=float)
        sma50 = (
            df["sma_50"].to_numpy(dtype=float)
            if "sma_50" in df.columns
            else np.full(n, np.nan)
        )
        sma20 = pd.Series(close).rolling(20, min_periods=10).mean().to_numpy()
        # Finite SMA50 required for that leg; NaN SMA50 fails that leg only
        soft_ok = (np.isfinite(sma50) & (close > sma50)) | (
            np.isfinite(sma20) & (close > sma20)
        )
        for i in range(n):
            if not s[i]:
                continue
            if _in_crash(i):
                continue
            if not soft_ok[i]:
                s[i] = False

    # non_crash_min_confidence: raise conf floor outside crash (uses p_buy if given else score)
    if wr.non_crash_min_confidence is not None:
        thr_nc = float(wr.non_crash_min_confidence)
        conf_src = p_arr if p_arr is not None else sc
        for i in range(n):
            if not s[i]:
                continue
            if non_crash_only and _in_crash(i):
                continue
            if not np.isfinite(conf_src[i]) or float(conf_src[i]) < thr_nc:
                s[i] = False

    return pd.Series(s, index=sig.index), pd.Series(sc, index=score.index)


def composite_rank_score(row: Dict[str, Any]) -> float:
    """Rank mega-study rows: win_rate + residual vs SPY + crash survival.

    Higher is better. Research heuristic only — not a live promotion gate.
    """
    wr = float(row.get("win_rate") or 0.0)
    excess = float(row.get("excess_total_vs_spy") or 0.0)
    # Crash window return: prefer less negative / positive during 2020 crash slice
    crash_ret = float(row.get("crash_2020_return") or 0.0)
    mdd = float(row.get("max_drawdown") or 0.0)
    # Normalize loosely
    score = (
        1.5 * wr
        + 0.35 * np.clip(excess, -2.0, 5.0)
        + 0.5 * np.clip(crash_ret, -1.0, 1.0)
        + 0.25 * np.clip(-mdd, 0.0, 1.0)  # smaller DD better
    )
    # Penalize tiny sample
    n = int(row.get("n_trades") or 0)
    if n < 20:
        score -= 0.5
    return float(score)


# Named presets for mega study / strategy registration
def preset_crash_rsi(thr: float = 30.0, conf: float = 0.22) -> CrashEntryConfig:
    return CrashEntryConfig(
        enabled=True,
        mode="rsi",
        rsi_threshold=thr,
        crash_min_confidence=conf,
        relax_regime=True,
        crash_relax_trend=True,
    )


def preset_crash_dd(dd: float = -0.15, conf: float = 0.22) -> CrashEntryConfig:
    return CrashEntryConfig(
        enabled=True,
        mode="dd",
        dd_threshold=dd,
        crash_min_confidence=conf,
        relax_regime=True,
        crash_relax_trend=True,
    )


def preset_crash_rsi_recover(thr: float = 30.0, conf: float = 0.22) -> CrashEntryConfig:
    return CrashEntryConfig(
        enabled=True,
        mode="rsi_recover",
        rsi_threshold=thr,
        require_rsi_rising=True,
        crash_min_confidence=conf,
        relax_regime=True,
        crash_relax_trend=True,
    )


def preset_wr_pack(
    *,
    max_atr: float = 0.16,
    cooldown: int = 10,
    soft_trend: bool = True,
) -> WinRateFilterConfig:
    return WinRateFilterConfig(
        max_atr_pct_tight=max_atr,
        hard_stop_cooldown_days=cooldown,
        soft_trend_non_crash=soft_trend,
    )
