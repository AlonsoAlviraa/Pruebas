"""Levered multi-year annual study runner (vectorized, fast).

Builds daily returns for index/regime/mom/dual/meta sleeves, applies leverage
proxy, ranks by mean multi-year return, selects PROMOTE/WATCH/KILL.

VIRTUAL only. Labels: levered_proxy / etf_levered_proxy / levered_wipe_proxy.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from paper_live.leverage.models import (
    LeverSpec,
    apply_leverage_to_returns,
    geometric_mean,
    max_upside_share,
    rank_by_mean_return,
    select_good_levered,
    year_returns_from_daily,
)

logger = logging.getLogger(__name__)

DEFAULT_ZOO = Path(__file__).resolve().parent / "zoo_levered_alpha.json"
DEFAULT_OUT = Path("reports/levered_annual")
WINDOWS: List[Tuple[str, str, str]] = [
    ("2022", "2022-01-03", "2022-12-30"),
    ("2023", "2023-01-03", "2023-12-29"),
    ("2024", "2024-01-02", "2024-12-31"),
    ("2025_study", "2025-01-02", "2099-12-31"),
]


def load_zoo(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or DEFAULT_ZOO)
    return json.loads(p.read_text(encoding="utf-8"))


def _close_series(feed: Any, ticker: str) -> pd.Series:
    try:
        hist = feed.history(ticker, through=feed.days[-1], include_through=True)
    except Exception:
        return pd.Series(dtype=float)
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    s = hist.set_index("date")["close"].astype(float)
    s.index = pd.to_datetime(s.index, utc=True)
    return s.sort_index()


def _returns(close: pd.Series) -> pd.Series:
    return close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _sma_mask(close: pd.Series, window: int = 200) -> pd.Series:
    sma = close.rolling(window, min_periods=max(20, window // 3)).mean()
    # causal: today's exposure uses SMA as of yesterday would be safer;
    # use through t (same-day close) is standard for daily research close-to-close.
    m = (close > sma).astype(float)
    m = m.where(sma.notna(), 0.0)
    return m.fillna(0.0)


def _vix_rank_series(vix: pd.Series, lookback: int = 252) -> pd.Series:
    def rank_at(i: int) -> float:
        if i < 10:
            return float("nan")
        lo = max(0, i - lookback + 1)
        tail = vix.iloc[lo : i + 1].dropna()
        if len(tail) < 10:
            return float("nan")
        cur = float(tail.iloc[-1])
        return float((tail <= cur).mean())

    vals = [rank_at(i) for i in range(len(vix))]
    return pd.Series(vals, index=vix.index, dtype=float)


def _top_mom_returns(
    panels: Mapping[str, pd.Series],
    *,
    index: pd.DatetimeIndex,
    top_k: int = 1,
    mom_lookback: int = 20,
    rebalance_every: int = 5,
) -> pd.Series:
    """Equal-weight top-k by trailing mom; hold between rebalances."""
    rets = pd.DataFrame({t: _returns(c.reindex(index).ffill()) for t, c in panels.items()})
    closes = pd.DataFrame({t: c.reindex(index).ffill() for t, c in panels.items()})
    out = pd.Series(0.0, index=index, dtype=float)
    weights = pd.Series(0.0, index=rets.columns, dtype=float)
    for i in range(len(index)):
        if i >= mom_lookback and (i % rebalance_every == 0 or weights.sum() <= 0):
            # trailing return mom_lookback days (causal: use closes up to i-1 for ranking?
            # use through i-1 for selection, earn r_i
            j = i - 1
            if j < mom_lookback:
                weights[:] = 0.0
            else:
                c0 = closes.iloc[j - mom_lookback]
                c1 = closes.iloc[j]
                mom = (c1 / c0 - 1.0).replace([np.inf, -np.inf], np.nan)
                mom = mom.dropna()
                if mom.empty:
                    weights[:] = 0.0
                else:
                    pick = mom.nlargest(int(top_k)).index.tolist()
                    weights[:] = 0.0
                    for t in pick:
                        weights[t] = 1.0 / len(pick)
        # earn today's returns on current weights
        out.iloc[i] = float((weights * rets.iloc[i]).sum())
    return out


def _dual_mom_returns(
    spy_r: pd.Series,
    qqq_r: pd.Series,
    spy_c: pd.Series,
    qqq_c: pd.Series,
    *,
    mom_lookback: int = 60,
    rebalance_every: int = 21,
) -> pd.Series:
    idx = spy_r.index.intersection(qqq_r.index)
    spy_r = spy_r.reindex(idx).fillna(0.0)
    qqq_r = qqq_r.reindex(idx).fillna(0.0)
    spy_c = spy_c.reindex(idx).ffill()
    qqq_c = qqq_c.reindex(idx).ffill()
    out = pd.Series(0.0, index=idx)
    choice = "QQQ"
    for i in range(len(idx)):
        if i >= mom_lookback and i % rebalance_every == 0:
            j = i - 1
            ms = float(spy_c.iloc[j] / spy_c.iloc[j - mom_lookback] - 1.0)
            mq = float(qqq_c.iloc[j] / qqq_c.iloc[j - mom_lookback] - 1.0)
            choice = "QQQ" if mq >= ms else "SPY"
        out.iloc[i] = float(qqq_r.iloc[i] if choice == "QQQ" else spy_r.iloc[i])
    return out


def _slice_year(
    rets: pd.Series,
    start: date,
    end: date,
) -> Tuple[np.ndarray, List[Any]]:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    sub = rets.loc[(rets.index >= s) & (rets.index <= e)]
    return sub.values.astype(float), list(sub.index)


def build_asset_return_map(feed: Any) -> Dict[str, pd.Series]:
    tickers = list(getattr(feed, "tickers", []) or [])
    out: Dict[str, pd.Series] = {}
    for t in tickers:
        c = _close_series(feed, t)
        if not c.empty:
            out[t.upper()] = c
    return out


def strategy_daily_returns(
    spec: Mapping[str, Any],
    closes: Mapping[str, pd.Series],
    *,
    calendar: Optional[pd.DatetimeIndex] = None,
) -> Tuple[pd.Series, str]:
    """Return (daily asset-or-blend returns before leverage, note)."""
    kind = str(spec.get("kind") or "index_hold")
    und = str(spec.get("underlying") or "QQQ").upper()

    if kind == "index_hold":
        if und not in closes:
            return pd.Series(dtype=float), f"missing:{und}"
        return _returns(closes[und]), f"hold:{und}"

    if kind == "regime_sma200":
        if und not in closes:
            return pd.Series(dtype=float), f"missing:{und}"
        r = _returns(closes[und])
        m = _sma_mask(closes[und], 200)
        # exposure applied later via mask in leverage; here return raw asset r + attach mask in runner
        r = r.copy()
        r.attrs["exposure_mask"] = m.reindex(r.index).fillna(0.0)
        return r, f"regime_sma200:{und}"

    if kind == "regime_vix_rank":
        if und not in closes or "VIX" not in closes:
            return pd.Series(dtype=float), "missing:VIX_or_und"
        r = _returns(closes[und])
        vr = _vix_rank_series(closes["VIX"])
        max_rank = float(spec.get("max_vix_rank") or 0.70)
        # full L if rank low; half if high
        m = pd.Series(0.5, index=r.index)
        aligned = vr.reindex(r.index).ffill()
        m = m.where(~(aligned <= max_rank), 1.0)
        m = m.where(aligned.notna(), 0.0)
        r = r.copy()
        r.attrs["exposure_mask"] = m.fillna(0.0)
        return r, f"regime_vix_rank:{und}"

    if kind == "top_mom":
        uni = [str(x).upper() for x in (spec.get("universe") or [])]
        panels = {t: closes[t] for t in uni if t in closes}
        if len(panels) < 2:
            return pd.Series(dtype=float), "top_mom_insufficient_universe"
        idx = calendar
        if idx is None:
            idx = panels[next(iter(panels))].index
            for c in panels.values():
                idx = idx.intersection(c.index)
        r = _top_mom_returns(
            panels,
            index=pd.DatetimeIndex(idx),
            top_k=int(spec.get("top_k") or 1),
            mom_lookback=int(spec.get("mom_lookback") or 20),
            rebalance_every=int(spec.get("rebalance_every") or 5),
        )
        return r, "top_mom"

    if kind == "dual_mom":
        if "SPY" not in closes or "QQQ" not in closes:
            return pd.Series(dtype=float), "dual_missing"
        r = _dual_mom_returns(
            _returns(closes["SPY"]),
            _returns(closes["QQQ"]),
            closes["SPY"],
            closes["QQQ"],
            mom_lookback=int(spec.get("mom_lookback") or 60),
            rebalance_every=int(spec.get("rebalance_every") or 21),
        )
        return r, "dual_mom"

    if kind == "core_sat":
        core = dict(spec.get("core") or {})
        sat = dict(spec.get("sat") or {})
        cw = float(spec.get("core_weight") or 0.7)
        core_spec = {
            "kind": "index_hold",
            "underlying": core.get("underlying") or "QQQ",
        }
        cr, _ = strategy_daily_returns(core_spec, closes, calendar=calendar)
        sat_spec = {
            "kind": sat.get("kind") or "top_mom",
            "top_k": sat.get("top_k", 1),
            "mom_lookback": sat.get("mom_lookback", 20),
            "rebalance_every": sat.get("rebalance_every", 5),
            "universe": sat.get("universe") or [],
        }
        sr, _ = strategy_daily_returns(sat_spec, closes, calendar=calendar)
        if cr.empty or sr.empty:
            return pd.Series(dtype=float), "core_sat_empty"
        idx = cr.index.intersection(sr.index)
        # Pre-leverage blend of *asset* returns; leverage applied on blend with L=1
        # then we apply core/sat leverage separately for honesty:
        # r = cw * (L_c * r_core) is wrong here — apply L in runner on each leg.
        blend = cw * cr.reindex(idx).fillna(0.0) + (1.0 - cw) * sr.reindex(idx).fillna(0.0)
        blend = blend.copy()
        blend.attrs["core_sat"] = {
            "core_w": cw,
            "core_L": float(core.get("leverage") or 1.5),
            "sat_L": float(sat.get("leverage") or 1.0),
            "core_r": cr.reindex(idx).fillna(0.0),
            "sat_r": sr.reindex(idx).fillna(0.0),
        }
        return blend, "core_sat"

    return pd.Series(dtype=float), f"unknown_kind:{kind}"


def run_one_strategy_years(
    spec: Mapping[str, Any],
    closes: Mapping[str, pd.Series],
    windows: Sequence[Tuple[str, str, str]],
    *,
    global_financing: float = 0.05,
    hard_dd_cap: float = -0.60,
    capital0: float = 100_000.0,
) -> Dict[str, Any]:
    """Run one levered strategy across calendar windows; return summary dict."""
    daily, note = strategy_daily_returns(spec, closes)
    sid = str(spec["id"])
    if daily.empty:
        return {
            "strategy_id": sid,
            "label": spec.get("label") or sid,
            "error": note or "empty",
            "mean_ret": None,
            "year_returns": {},
        }

    # core_sat special path: lever each leg then blend
    core_sat = daily.attrs.get("core_sat")
    year_rets: Dict[str, float] = {}
    year_dd: Dict[str, float] = {}
    year_wipe: Dict[str, bool] = {}
    year_label: Dict[str, str] = {}
    all_daily_parts: List[pd.Series] = []

    for name, ws, we in windows:
        end = we if we != "2099-12-31" else str(daily.index[-1].date())
        start_d = date.fromisoformat(ws)
        end_d = date.fromisoformat(end[:10])

        if core_sat:
            cr = core_sat["core_r"]
            sr = core_sat["sat_r"]
            cw = float(core_sat["core_w"])
            # Apply leverage to each leg path for this window
            r_c, d_c = _slice_year(cr, start_d, end_d)
            r_s, d_s = _slice_year(sr, start_d, end_d)
            if len(r_c) == 0:
                year_rets[name] = 0.0
                year_dd[name] = 0.0
                year_wipe[name] = False
                continue
            pc = apply_leverage_to_returns(
                r_c,
                dates=d_c,
                spec=LeverSpec(
                    leverage=float(core_sat["core_L"]),
                    financing_rate=float(spec.get("financing_rate", global_financing)),
                    hard_dd_cap=hard_dd_cap,
                    daily_reset=True,
                    label="etf_levered_proxy",
                ),
                capital0=1.0,
            )
            ps = apply_leverage_to_returns(
                r_s,
                dates=d_s,
                spec=LeverSpec(
                    leverage=float(core_sat["sat_L"]),
                    financing_rate=float(spec.get("financing_rate", global_financing)),
                    hard_dd_cap=hard_dd_cap,
                    daily_reset=True,
                    label="levered_proxy",
                ),
                capital0=1.0,
            )
            # blend equity paths daily
            n = min(len(pc.daily_returns), len(ps.daily_returns))
            blend_r = cw * pc.daily_returns[:n] + (1.0 - cw) * ps.daily_returns[:n]
            # compound
            g = 1.0
            peak = 1.0
            eq = 1.0
            mdd = 0.0
            for x in blend_r:
                eq *= 1.0 + float(x)
                peak = max(peak, eq)
                mdd = min(mdd, eq / peak - 1.0)
            year_rets[name] = eq - 1.0
            year_dd[name] = mdd
            year_wipe[name] = bool(pc.wiped or ps.wiped)
            year_label[name] = "levered_proxy_meta"
            all_daily_parts.append(pd.Series(blend_r, index=d_c[:n]))
            continue

        r_arr, dts = _slice_year(daily, start_d, end_d)
        if len(r_arr) == 0:
            year_rets[name] = 0.0
            year_dd[name] = 0.0
            year_wipe[name] = False
            continue

        mask = None
        if "exposure_mask" in getattr(daily, "attrs", {}):
            mser = daily.attrs["exposure_mask"]
            mser = mser.reindex(pd.DatetimeIndex(dts)).fillna(0.0)
            mask = mser.values.astype(float)

        L = float(spec.get("leverage") or 1.0)
        # benchmarks family: no financing
        fin = float(spec.get("financing_rate", global_financing))
        if str(spec.get("family") or "") == "benchmark":
            fin = 0.0
        lev_spec = LeverSpec(
            leverage=L,
            financing_rate=fin,
            hard_dd_cap=float(spec.get("hard_dd_cap") or hard_dd_cap),
            daily_reset=bool(spec.get("daily_reset", True)),
            vol_target=(
                float(spec["vol_target"]) if spec.get("vol_target") is not None else None
            ),
            vol_lookback=int(spec.get("vol_lookback") or 20),
            max_leverage=(
                float(spec["max_leverage"])
                if spec.get("max_leverage") is not None
                else None
            ),
            label="etf_levered_proxy" if L > 1.01 else "unlevered_proxy",
        )
        path = apply_leverage_to_returns(
            r_arr,
            dates=dts,
            spec=lev_spec,
            capital0=1.0,
            exposure_mask=mask,
        )
        year_rets[name] = path.total_return
        year_dd[name] = path.max_dd
        year_wipe[name] = path.wiped
        year_label[name] = path.data_label
        all_daily_parts.append(pd.Series(path.daily_returns, index=dts))

    rets_list = [year_rets[n] for n, _, _ in windows if n in year_rets]
    mean_ret = float(np.mean(rets_list)) if rets_list else None
    geo = geometric_mean(rets_list) if rets_list else None
    worst_dd = min(year_dd.values()) if year_dd else 0.0
    pos_years = sum(1 for v in rets_list if v > 0)
    wipe_years = sum(1 for v in year_wipe.values() if v)
    return {
        "strategy_id": sid,
        "label": str(spec.get("label") or sid),
        "family": str(spec.get("family") or ""),
        "leverage": float(spec.get("leverage") or 1.0),
        "kind": str(spec.get("kind") or ""),
        "note": note,
        "year_returns": year_rets,
        "year_max_dd": year_dd,
        "year_wipe": year_wipe,
        "year_data_label": year_label,
        "mean_ret": mean_ret,
        "geo_ret": geo,
        "worst_dd": worst_dd,
        "n_positive_years": pos_years,
        "wipe_years": wipe_years,
        "max_upside_share": max_upside_share(year_rets),
        "data_label": "levered_proxy",
        "mode": "paper",
        "capital_label": "VIRTUAL",
    }


def attach_benchmarks(
    row: Dict[str, Any],
    bench_year: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    """Add excess vs SPY/QQQ/best per year + means."""
    yr = row.get("year_returns") or {}
    xs_spy = []
    xs_qqq = []
    xs_best = []
    beat_spy_3 = 0
    beat_best_3 = 0
    for y, r in yr.items():
        b = bench_year.get(y) or {}
        spy = b.get("SPY")
        qqq = b.get("QQQ")
        iwm = b.get("IWM")
        vals = [v for v in (spy, qqq, iwm) if v is not None]
        best = max(vals) if vals else None
        if spy is not None:
            xs = float(r) - float(spy)
            xs_spy.append(xs)
            if float(r) >= float(spy) + 0.03:
                beat_spy_3 += 1
        if qqq is not None:
            xs_qqq.append(float(r) - float(qqq))
        if best is not None:
            xs_best.append(float(r) - float(best))
            if float(r) >= float(best) + 0.03:
                beat_best_3 += 1
    row = dict(row)
    row["mean_xs_spy"] = float(np.mean(xs_spy)) if xs_spy else None
    row["mean_xs_qqq"] = float(np.mean(xs_qqq)) if xs_qqq else None
    row["mean_xs_best"] = float(np.mean(xs_best)) if xs_best else None
    row["n_years_beat_spy_3pp"] = beat_spy_3
    row["n_years_beat_best_3pp"] = beat_best_3
    wdd = abs(float(row.get("worst_dd") or 0.0))
    mr = float(row.get("mean_ret") or 0.0)
    row["calmar_like"] = (mr / wdd) if wdd > 1e-9 else None
    return row


def bh_year_returns(
    close: pd.Series,
    windows: Sequence[Tuple[str, str, str]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, ws, we in windows:
        end = we if we != "2099-12-31" else str(close.index[-1].date())
        s = pd.Timestamp(ws, tz="UTC")
        e = pd.Timestamp(end[:10], tz="UTC")
        sub = close.loc[(close.index >= s) & (close.index <= e)]
        if len(sub) < 2:
            continue
        out[name] = float(sub.iloc[-1] / sub.iloc[0] - 1.0)
    return out


def write_pack(
    rows: Sequence[Mapping[str, Any]],
    *,
    promote: Sequence[Mapping[str, Any]],
    watch: Sequence[Mapping[str, Any]],
    kill: Sequence[Mapping[str, Any]],
    bench_year: Mapping[str, Mapping[str, float]],
    out_root: Path,
    meta: Mapping[str, Any],
) -> Dict[str, Path]:
    out_root = Path(out_root)
    latest = out_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    ranked = rank_by_mean_return(list(rows))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": dict(meta),
        "benchmarks_by_year": bench_year,
        "strategies": ranked,
        "promote": list(promote),
        "watch": list(watch),
        "kill": list(kill),
        "disclaimer": (
            "Levered research proxies with financing. VIRTUAL capital. "
            "Not real TQQQ/margin fills. Not financial advice."
        ),
    }
    (latest / "full_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (latest / "winners.json").write_text(
        json.dumps(
            {
                "promote": promote,
                "watch": watch,
                "kill": kill,
                "rank_by_mean_ret": [
                    {
                        "rank": r.get("rank_mean_ret"),
                        "id": r.get("strategy_id"),
                        "mean_ret": r.get("mean_ret"),
                        "geo_ret": r.get("geo_ret"),
                        "xs_spy": r.get("mean_xs_spy"),
                        "worst_dd": r.get("worst_dd"),
                        "L": r.get("leverage"),
                    }
                    for r in ranked
                    if not str(r.get("strategy_id", "")).startswith("BH_")
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    def pct(x: Any) -> str:
        try:
            return f"{float(x):.2%}"
        except Exception:
            return "n/a"

    lines = [
        f"# Levered annual study — `{payload['generated_at'][:10]}`",
        "",
        "**Capital:** VIRTUAL · **Labels:** levered_proxy / etf_levered_proxy",
        f"**Financing:** {meta.get('financing_rate', 0.05):.0%} on (L−1) · hard DD cap {meta.get('hard_dd_cap')}",
        "",
        "## Benchmarks (pure BH by year)",
        "",
        "| Year | SPY | QQQ | IWM |",
        "|------|-----|-----|-----|",
    ]
    for y in ["2022", "2023", "2024", "2025_study"]:
        b = bench_year.get(y) or {}
        lines.append(
            f"| {y} | {pct(b.get('SPY'))} | {pct(b.get('QQQ'))} | {pct(b.get('IWM'))} |"
        )

    lines += [
        "",
        "## Ranking by **mean return** (all years)",
        "",
        "| Rank | ID | L | MeanRet | GeoRet | xsSPY | xsQQQ | WorstDD | +yrs | UpsideConc | WipeY |",
        "|------|----|---|---------|--------|-------|-------|---------|------|------------|-------|",
    ]
    for r in ranked:
        if str(r.get("strategy_id", "")).startswith("BH_"):
            continue
        lines.append(
            f"| {r.get('rank_mean_ret')} | `{r.get('strategy_id')}` | {r.get('leverage')} | "
            f"{pct(r.get('mean_ret'))} | {pct(r.get('geo_ret'))} | {pct(r.get('mean_xs_spy'))} | "
            f"{pct(r.get('mean_xs_qqq'))} | {pct(r.get('worst_dd'))} | "
            f"{r.get('n_positive_years')} | {pct(r.get('max_upside_share'))} | {r.get('wipe_years')} |"
        )

    lines += ["", "## PROMOTE_LEV (really good by mean + filters)", ""]
    if promote:
        for p in promote[:10]:
            lines.append(
                f"- `{p.get('strategy_id')}` mean={pct(p.get('mean_ret'))} "
                f"xsSPY={pct(p.get('mean_xs_spy'))} DD={pct(p.get('worst_dd'))} "
                f"— {', '.join(p.get('reasons') or [])}"
            )
            yr = p.get("year_returns") or {}
            lines.append(
                f"  years: "
                + ", ".join(f"{k}={pct(v)}" for k, v in sorted(yr.items()))
            )
    else:
        lines.append("_None passed GOOD filters_")

    lines += ["", "## WATCH_LEV", ""]
    for p in list(watch)[:8]:
        lines.append(
            f"- `{p.get('strategy_id')}` mean={pct(p.get('mean_ret'))} "
            f"— {', '.join(p.get('reasons') or [])}"
        )

    lines += [
        "",
        "---",
        str(payload["disclaimer"]),
        "",
    ]
    md_path = latest / "SUMMARY.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    # also RESCORE alias
    rescore = out_root / "RESCORE.md"
    rescore.write_text("\n".join(lines), encoding="utf-8")
    (out_root / "RESCORE.json").write_text(
        json.dumps(
            {"promote": promote, "watch": watch, "kill": kill, "meta": meta},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {"latest": latest, "summary": md_path, "rescore": rescore}


def run_levered_annual_study(
    *,
    out_root: Path = DEFAULT_OUT,
    zoo_path: Optional[Path] = None,
    lookback_days: int = 2000,
    force_synthetic: bool = False,
    max_strategies: Optional[int] = None,
) -> Dict[str, Any]:
    from paper_live.cloud.free_data import SEED_DIR, build_cloud_feed

    zoo = load_zoo(zoo_path)
    capital0 = float(zoo.get("capital0") or 100_000.0)
    fin = float(zoo.get("financing_rate") or 0.05)
    hard = float(zoo.get("hard_dd_cap") or -0.60)
    specs = list(zoo.get("strategies") or [])
    if max_strategies is not None:
        specs = specs[: int(max_strategies)]

    tickers = {"SPY", "QQQ", "IWM", "VIX"}
    for s in specs:
        u = str(s.get("underlying") or "").upper()
        if u and u not in ("MULTI",):
            tickers.add(u)
        for t in s.get("universe") or []:
            tickers.add(str(t).upper())
        sat = (s.get("sat") or {}).get("universe") or []
        for t in sat:
            tickers.add(str(t).upper())
        core_u = ((s.get("core") or {}).get("underlying")) or None
        if core_u:
            tickers.add(str(core_u).upper())

    logger.info("Levered study tickers=%s n_strats=%d", sorted(tickers), len(specs))
    feed, sources = build_cloud_feed(
        sorted(tickers),
        cache_dir=Path(out_root) / "data_cache",
        seed_dir=SEED_DIR,
        lookback_calendar_days=int(lookback_days),
        force_synthetic=force_synthetic,
        require_real=not force_synthetic,
        min_real_tickers=3 if not force_synthetic else 0,
    )
    closes = build_asset_return_map(feed)
    # clamp windows to available
    days = list(feed.days)
    if not days:
        raise RuntimeError("no feed days")

    win_use: List[Tuple[str, str, str]] = []
    for name, ws, we in WINDOWS:
        if we == "2099-12-31":
            we = days[-1].isoformat()
        # clamp start
        req_s = date.fromisoformat(ws)
        req_e = date.fromisoformat(we[:10])
        s = next((d for d in days if d >= req_s), days[0])
        e = next((d for d in reversed(days) if d <= req_e), days[-1])
        win_use.append((name, s.isoformat(), e.isoformat()))
        if s != req_s or e != req_e:
            logger.warning("Window %s clamped %s→%s actual %s→%s", name, req_s, req_e, s, e)

    bench_year: Dict[str, Dict[str, float]] = {}
    for name, ws, we in win_use:
        bench_year[name] = {}
        for t in ("SPY", "QQQ", "IWM"):
            if t in closes:
                bh = bh_year_returns(closes[t], [(name, ws, we)])
                if name in bh:
                    bench_year[name][t] = bh[name]

    rows: List[Dict[str, Any]] = []
    for i, sp in enumerate(specs, 1):
        logger.info("[%d/%d] %s", i, len(specs), sp.get("id"))
        row = run_one_strategy_years(
            sp,
            closes,
            win_use,
            global_financing=fin,
            hard_dd_cap=hard,
            capital0=capital0,
        )
        row = attach_benchmarks(row, bench_year)
        rows.append(row)
        logger.info(
            "  mean=%s xs_spy=%s dd=%s",
            row.get("mean_ret"),
            row.get("mean_xs_spy"),
            row.get("worst_dd"),
        )

    # BH means for GOOD filter
    qqq_means = []
    spy_means = []
    for y, b in bench_year.items():
        if "QQQ" in b:
            qqq_means.append(b["QQQ"])
        if "SPY" in b:
            spy_means.append(b["SPY"])
    qqq_bh_mean = float(np.mean(qqq_means)) if qqq_means else None
    spy_bh_mean = float(np.mean(spy_means)) if spy_means else None

    # exclude pure BH from promote competition but keep in table
    trade_rows = [r for r in rows if not str(r.get("strategy_id", "")).startswith("BH_")]
    promote, watch, kill = select_good_levered(
        trade_rows,
        qqq_bh_mean=qqq_bh_mean,
        spy_bh_mean=spy_bh_mean,
    )

    meta = {
        "zoo": str(zoo_path or DEFAULT_ZOO),
        "financing_rate": fin,
        "hard_dd_cap": hard,
        "lookback_days": lookback_days,
        "force_synthetic": force_synthetic,
        "data_sources": sources,
        "windows": [{"name": n, "start": s, "end": e} for n, s, e in win_use],
        "qqq_bh_mean": qqq_bh_mean,
        "spy_bh_mean": spy_bh_mean,
        "n_promote": len(promote),
        "n_watch": len(watch),
        "n_kill": len(kill),
    }
    paths = write_pack(
        rows,
        promote=promote,
        watch=watch,
        kill=kill,
        bench_year=bench_year,
        out_root=Path(out_root),
        meta=meta,
    )
    logger.info(
        "PROMOTE=%d WATCH=%d KILL=%d → %s",
        len(promote),
        len(watch),
        len(kill),
        paths["summary"],
    )
    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "meta": meta,
        "promote": promote,
        "watch": watch,
        "n_strategies": len(rows),
    }
