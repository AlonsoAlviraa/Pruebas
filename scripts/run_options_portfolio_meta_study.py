#!/usr/bin/env python3
"""Long-history options sleeve study + meta-label portfolio selection.

- Grid of many options strategies (no NVDA×2 / QQQ×2)
- EODHD underlyings; marks_mode real_chain vs proxy_bs|vix_surface
- **Real marks always** for short-vol claims; proxy excludes short-premium pure kinds
- Walk-forward yearly: train meta on past years → allocate next year
- Meta label default: beat_spy (not mere ret>0)
- 1 sleeve per underlying before caps; spy_cash_blend benchmark
- ``--rescore-only``: re-run meta+allocator from existing sleeve_year_returns.json

VIRTUAL only.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.options.grid_zoo import filter_zoo_for_marks, write_grid_zoo
from paper_live.options.marks_policy import (
    BAN_RULE_NORM_VIOLATION,
    CHAIN_PRICING_ENGINE_AVAILABLE,
    MARKS_PROXY_BS,
    filter_sleeve_years_for_marks,
    honesty_disclaimer,
    is_proxy_marks,
    kind_from_sleeve_ymap,
    normalize_marks_mode,
    resolve_study_marks_context,
)
from paper_live.options.replay_options import run_options_strategy
from paper_live.options.risk import OptionsRiskConfig
from paper_live.options.strategies import OptionStrategySpec
from paper_live.portfolio.meta_label_selector import (
    LABEL_MODES,
    MetaLabelConfig,
    build_feature_row,
    fit_meta,
    make_meta_label,
    predict_proba,
    rank_sleeves_for_year,
)
from paper_live.portfolio.sleeve_portfolio import (
    PortfolioCaps,
    allocate_weights,
    portfolio_vs_spy_cash_blend,
    portfolio_year_return,
)
from paper_live.options.vol_surface import resolve_vix_level, VIX_TICKERS, VIX3M_TICKERS
from paper_live.options.vol_proxy import historical_vol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("opt_port_meta")

# Process-pool worker globals
_W_FEED = None
_W_CAPITAL0 = 100_000.0
_W_RISK: Optional[OptionsRiskConfig] = None
_W_CLAMPED: List[Tuple[str, date, date]] = []
_W_DATA_LABEL = "proxy_bs"


def _years_range(y0: int, y1: int) -> List[Tuple[str, str, str]]:
    out = []
    for y in range(y0, y1 + 1):
        if y == y1:
            out.append((str(y), f"{y}-01-02", "2099-12-31"))
        else:
            out.append((str(y), f"{y}-01-02", f"{y}-12-31"))
    return out


def _spec_from_dict(s: Mapping[str, Any]) -> OptionStrategySpec:
    return OptionStrategySpec(
        id=str(s["id"]),
        label=str(s.get("label") or s["id"]),
        kind=str(s["kind"]),
        underlying=str(s.get("underlying") or "SPY"),
        dte_days=int(s.get("dte_days") or 30),
        otm_pct=float(s.get("otm_pct") or 0.05),
        wing_otm_pct=float(s.get("wing_otm_pct") or 0.12),
        contracts=int(s.get("contracts") or 2),
        max_portfolio_dd=s.get("max_portfolio_dd"),
        max_single_day_drop=s.get("max_single_day_drop"),
        max_margin_fraction=s.get("max_margin_fraction"),
        hard_kill_enabled=s.get("hard_kill_enabled"),
        meta=dict(s.get("meta") or {}),
        notes=str(s.get("notes") or ""),
    )


def _specs_from_zoo(
    path: Path,
    max_n: Optional[int] = None,
    *,
    marks_mode: Optional[str] = None,
    apply_proxy_filter: bool = True,
) -> Tuple[List[OptionStrategySpec], float, OptionsRiskConfig, Dict[str, Any]]:
    z = json.loads(path.read_text(encoding="utf-8"))
    risk = OptionsRiskConfig.from_mapping(z.get("risk") or {})
    raw = list(z.get("strategies") or [])
    mode = normalize_marks_mode(
        marks_mode or z.get("marks_mode") or z.get("data_label") or "proxy_bs"
    )
    if apply_proxy_filter and is_proxy_marks(mode):
        before = len(raw)
        raw = filter_zoo_for_marks(raw, mode)
        logger.info(
            "Proxy marks filter: %d → %d strategies (short-vol pure dropped)",
            before,
            len(raw),
        )
    # Diversified sample: stride across full zoo so kinds/unds mix
    if max_n and len(raw) > max_n:
        if max_n <= 1:
            raw = raw[:1]
        else:
            idxs = np.linspace(0, len(raw) - 1, num=max_n, dtype=int)
            seen = set()
            picked = []
            for i in idxs:
                if int(i) not in seen:
                    seen.add(int(i))
                    picked.append(raw[int(i)])
            cash = [s for s in raw if str(s.get("id")) == "G_CASH_CTRL"]
            rest = [s for s in picked if str(s.get("id")) != "G_CASH_CTRL"]
            raw = (cash[:1] + rest)[:max_n]
    elif max_n:
        raw = raw[:max_n]
    specs = [_spec_from_dict(s) for s in raw]
    return specs, float(z.get("capital0") or 100_000.0), risk, z


def _bh(feed, ticker: str, start: date, end: date) -> Optional[float]:
    days = feed.days
    if not days:
        return None
    s = next((d for d in days if d >= start), None)
    e = next((d for d in reversed(days) if d <= end), None)
    if s is None or e is None or s > e:
        return None
    b0, b1 = feed.bar(ticker, s), feed.bar(ticker, e)
    if b0 and b1 and float(b0.close) > 0:
        return float(b1.close) / float(b0.close) - 1.0
    return None


def _macro_on_day(feed, day: date) -> Dict[str, float]:
    vix = resolve_vix_level(feed, day, aliases=VIX_TICKERS) or 20.0
    vix3m = resolve_vix_level(feed, day, aliases=VIX3M_TICKERS) or vix
    try:
        hist = feed.history("SPY", through=day, include_through=True)
        closes = (
            hist.set_index("date")["close"].astype(float)
            if hist is not None and not hist.empty
            else None
        )
        hv = float(historical_vol(closes, window=20)) if closes is not None else 0.15
        if not math.isfinite(hv):
            hv = 0.15
    except Exception:
        hv = 0.15
    return {"vix": float(vix), "vix3m": float(vix3m), "hv20": hv}


def _cell_from_result(r, sp: OptionStrategySpec) -> Dict[str, Any]:
    return {
        "total_return": float(r.total_return),
        "max_dd": float(r.max_dd),
        "n_opens": int(r.n_opens),
        "hard_kill": bool(r.hard_kill),
        "kind": sp.kind,
        "underlying": sp.underlying,
        "dte_days": sp.dte_days,
        "otm_pct": sp.otm_pct,
    }


def _eval_spec_years(
    feed,
    sp: OptionStrategySpec,
    clamped: Sequence[Tuple[str, date, date]],
    capital0: float,
    risk: OptionsRiskConfig,
    existing: Optional[Mapping[str, Any]] = None,
    *,
    data_label: str = "proxy_bs",
) -> Dict[str, Dict[str, Any]]:
    """Run each calendar year for one sleeve (uses optimized feed path)."""
    out: Dict[str, Dict[str, Any]] = dict(existing or {})
    for name, start_d, end_d in clamped:
        if name in out and not out[name].get("error"):
            continue
        try:
            r = run_options_strategy(
                feed,
                sp,
                start=start_d,
                end=end_d,
                capital0=capital0,
                risk=risk,
                data_label=data_label,
                compute_delta=False,
                store_curve=False,
            )
            out[name] = _cell_from_result(r, sp)
        except Exception as e:
            out[name] = {
                "total_return": 0.0,
                "max_dd": 0.0,
                "n_opens": 0,
                "hard_kill": False,
                "error": str(e),
                "kind": sp.kind,
                "underlying": sp.underlying,
                "dte_days": sp.dte_days,
                "otm_pct": sp.otm_pct,
            }
    return out


def _worker_init(
    unds: List[str],
    eodhd_from: str,
    cache_dir: str,
    capital0: float,
    risk_dict: Dict[str, Any],
    clamped_iso: List[Tuple[str, str, str]],
    data_label: str = "proxy_bs",
) -> None:
    global _W_FEED, _W_CAPITAL0, _W_RISK, _W_CLAMPED, _W_DATA_LABEL
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from paper_live.data.eodhd_client import build_eodhd_feed

    feed, _ = build_eodhd_feed(
        unds,
        start=eodhd_from,
        cache_dir=Path(cache_dir),
        min_history=60,
    )
    _W_FEED = feed
    _W_CAPITAL0 = capital0
    _W_RISK = OptionsRiskConfig.from_mapping(risk_dict)
    _W_CLAMPED = [
        (n, date.fromisoformat(s), date.fromisoformat(e)) for n, s, e in clamped_iso
    ]
    _W_DATA_LABEL = data_label


def _worker_run(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    sp = _spec_from_dict(payload["spec"])
    existing = payload.get("existing") or {}
    assert _W_FEED is not None and _W_RISK is not None
    out = _eval_spec_years(
        _W_FEED,
        sp,
        _W_CLAMPED,
        _W_CAPITAL0,
        _W_RISK,
        existing=existing,
        data_label=_W_DATA_LABEL,
    )
    return sp.id, out


def _run_walk_forward(
    *,
    feed,
    sleeve_years: Mapping[str, Dict[str, Dict[str, Any]]],
    clamped: Sequence[Tuple[str, date, date]],
    benches: Mapping[str, Mapping[str, Optional[float]]],
    cfg: MetaLabelConfig,
    caps: PortfolioCaps,
    top_k: int,
    marks_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
    year_names = [n for n, _, _ in clamped]
    wf_rows: List[Dict[str, Any]] = []
    port_rets: Dict[str, float] = {}
    equal_top_rets: Dict[str, float] = {}
    label_skip_spy_missing = 0

    for i, year in enumerate(year_names):
        if i < cfg.min_train_years:
            port_rets[year] = 0.0
            equal_top_rets[year] = 0.0
            wf_rows.append(
                {
                    "year": year,
                    "mode": "warmup",
                    "portfolio_return": 0.0,
                    "spy_bh": benches[year].get("SPY"),
                    "qqq_bh": benches[year].get("QQQ"),
                    "invested_weight": 0.0,
                    "spy_cash_blend": 0.0,
                    "port_vs_spy_cash_blend": 0.0,
                    "n_selected": 0,
                    "label_mode": cfg.label_mode,
                    "marks_mode": marks_mode,
                }
            )
            continue

        train_years = year_names[:i]
        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        year_label_skips = 0
        for ty_i, ty in enumerate(train_years[:-1]):
            ny = train_years[ty_i + 1]
            _, _, end_ty = next(c for c in clamped if c[0] == ty)
            macro = _macro_on_day(feed, end_ty)
            spy_ny = benches.get(ny, {}).get("SPY")
            for sid, ymap in sleeve_years.items():
                if sid == "G_CASH_CTRL":
                    continue
                cell_t = ymap.get(ty)
                cell_n = ymap.get(ny)
                if not cell_t or not cell_n or cell_t.get("error") or cell_n.get("error"):
                    continue
                y_lab = make_meta_label(
                    float(cell_n.get("total_return") or 0),
                    spy_ret=float(spy_ny) if spy_ny is not None else None,
                    cfg=cfg,
                )
                if y_lab is None:
                    year_label_skips += 1
                    label_skip_spy_missing += 1
                    continue
                prior1 = float(cell_t.get("total_return") or 0.0)
                hist_rets = []
                for hy in train_years[: ty_i + 1][-3:]:
                    if hy in ymap and not ymap[hy].get("error"):
                        hist_rets.append(float(ymap[hy]["total_return"]))
                prior3 = float(np.mean(hist_rets)) if hist_rets else prior1
                feat = {
                    **macro,
                    "prior_ret_1y": prior1,
                    "prior_ret_3y": prior3,
                    "prior_vol": abs(prior1) + 0.1,
                    "prior_max_dd": float(cell_t.get("max_dd") or 0.0),
                    "n_opens_1y": float(cell_t.get("n_opens") or 0),
                    "dte_days": float(cell_t.get("dte_days") or 30),
                    "otm_pct": float(cell_t.get("otm_pct") or 0.05),
                    "kind": cell_t.get("kind"),
                    "underlying": cell_t.get("underlying"),
                }
                X_list.append(build_feature_row(feat))
                y_list.append(float(y_lab))

        fit = None
        if X_list:
            X = np.vstack(X_list)
            y = np.asarray(y_list, dtype=float)
            fit = fit_meta(X, y, cfg=cfg)

        prev = year_names[i - 1]
        _, _, end_prev = next(c for c in clamped if c[0] == prev)
        macro = _macro_on_day(feed, end_prev)
        candidates: List[Dict[str, Any]] = []
        Xc: List[np.ndarray] = []
        for sid, ymap in sleeve_years.items():
            if sid == "G_CASH_CTRL":
                continue
            cell_p = ymap.get(prev)
            if not cell_p or cell_p.get("error"):
                continue
            hist_rets = []
            for hy in year_names[:i][-3:]:
                if hy in ymap and not ymap[hy].get("error"):
                    hist_rets.append(float(ymap[hy]["total_return"]))
            prior1 = float(cell_p.get("total_return") or 0.0)
            prior3 = float(np.mean(hist_rets)) if hist_rets else prior1
            feat = {
                **macro,
                "prior_ret_1y": prior1,
                "prior_ret_3y": prior3,
                "prior_vol": (
                    float(np.std(hist_rets)) if len(hist_rets) > 1 else abs(prior1) + 0.1
                ),
                "prior_max_dd": float(cell_p.get("max_dd") or 0.0),
                "n_opens_1y": float(cell_p.get("n_opens") or 0),
                "dte_days": float(cell_p.get("dte_days") or 30),
                "otm_pct": float(cell_p.get("otm_pct") or 0.05),
                "kind": cell_p.get("kind"),
                "underlying": cell_p.get("underlying"),
                "strategy_id": sid,
            }
            candidates.append(feat)
            Xc.append(build_feature_row(feat))

        if not candidates:
            port_rets[year] = 0.0
            equal_top_rets[year] = 0.0
            wf_rows.append(
                {
                    "year": year,
                    "mode": "no_candidates",
                    "portfolio_return": 0.0,
                    "invested_weight": 0.0,
                    "spy_cash_blend": None,
                    "label_mode": cfg.label_mode,
                    "marks_mode": marks_mode,
                }
            )
            continue

        if fit is not None:
            proba = predict_proba(fit, np.vstack(Xc))
        else:
            # fallback: prior ret vs prior SPY if available
            spy_prev = benches.get(prev, {}).get("SPY")
            proba = np.array(
                [
                    0.5
                    + 0.5
                    * np.tanh(
                        5
                        * (
                            float(c.get("prior_ret_1y") or 0)
                            - (float(spy_prev) if spy_prev is not None else 0.0)
                        )
                    )
                    for c in candidates
                ]
            )

        selected = rank_sleeves_for_year(
            candidates,
            proba,
            top_k=int(top_k),
            cfg=cfg,
            one_per_underlying=True,
        )
        weights = allocate_weights(selected, caps=caps)

        sleeve_r = {
            sid: float(sleeve_years[sid].get(year, {}).get("total_return") or 0.0)
            for sid in weights
        }
        pret = portfolio_year_return(weights, sleeve_r)
        port_rets[year] = pret

        inv_w, blend, vs_blend = portfolio_vs_spy_cash_blend(
            pret, weights, benches[year].get("SPY")
        )

        naive = sorted(candidates, key=lambda c: -float(c.get("prior_ret_1y") or 0))[:5]
        if naive:
            ew = 0.9 / len(naive)
            nr = sum(
                ew
                * float(
                    sleeve_years[str(c["strategy_id"])].get(year, {}).get("total_return")
                    or 0
                )
                for c in naive
            )
        else:
            nr = 0.0
        equal_top_rets[year] = nr

        wf_rows.append(
            {
                "year": year,
                "mode": "meta" if fit is not None else "fallback_prior",
                "portfolio_return": pret,
                "naive_top5_return": nr,
                "spy_bh": benches[year].get("SPY"),
                "qqq_bh": benches[year].get("QQQ"),
                "invested_weight": inv_w,
                "cash_weight": max(0.0, 1.0 - inv_w),
                "spy_cash_blend": blend,
                "port_vs_spy_cash_blend": vs_blend,
                "n_selected": len(weights),
                "weights": weights,
                "meta_train_rows": fit.train_rows if fit else 0,
                "meta_pos_rate": fit.train_pos_rate if fit else None,
                "label_mode": cfg.label_mode,
                "label_skip_spy_missing": year_label_skips,
                "marks_mode": marks_mode,
                "one_per_underlying": True,
                "selected": [
                    {
                        "id": s.get("strategy_id"),
                        "proba": s.get("meta_proba"),
                        "size": s.get("meta_size"),
                        "und": s.get("underlying"),
                        "kind": s.get("kind"),
                    }
                    for s in selected
                ],
            }
        )
        logger.info(
            "WF %s port=%+.2f%% spy=%s blend=%s w=%.2f selected=%d",
            year,
            100 * pret,
            benches[year].get("SPY"),
            blend,
            inv_w,
            len(weights),
        )

    return wf_rows, port_rets, equal_top_rets, label_skip_spy_missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Options portfolio + meta-label long-hist study")
    ap.add_argument("--zoo", default="paper_live/cloud/zoo_options_grid.json")
    ap.add_argument("--max-strategies", type=int, default=800)
    ap.add_argument("--build-grid", action="store_true", help="Rebuild grid zoo before run")
    ap.add_argument("--from-year", type=int, default=2010)
    ap.add_argument("--to-year", type=int, default=2025)
    ap.add_argument("--out", default="reports/options_portfolio_meta")
    ap.add_argument("--eodhd-from", default="2005-01-01")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--smoke", action="store_true", help="Faster: fewer years/strats")
    ap.add_argument("--resume", action="store_true", help="Load sleeve_year_cache if present")
    ap.add_argument(
        "--rescore-only",
        action="store_true",
        help="Skip sleeve replay; rescore meta+allocator from sleeve_year_returns.json",
    )
    ap.add_argument(
        "--marks-mode",
        default="proxy_bs",
        help="real_chain | proxy_bs | proxy_bs|vix_surface (default proxy_bs)",
    )
    ap.add_argument(
        "--label-mode",
        default="beat_spy",
        choices=list(LABEL_MODES),
        help="Meta training label (default beat_spy)",
    )
    ap.add_argument(
        "--no-proxy-filter",
        action="store_true",
        help="Do not drop short-vol kinds under proxy (violates research norm; debug only)",
    )
    args = ap.parse_args()

    # Fail closed: CLI real_chain cannot claim real marks until chain engine is wired
    marks_ctx = resolve_study_marks_context(
        args.marks_mode,
        chain_engine_available=CHAIN_PRICING_ENGINE_AVAILABLE,
        pricing_backend=MARKS_PROXY_BS,  # run_options_strategy is model BS only today
    )
    marks_mode = str(marks_ctx["effective_mode"])
    requested_marks_mode = str(marks_ctx["requested_mode"])
    option_marks_label = str(marks_ctx["option_marks_label"])
    pricing_backend = str(marks_ctx["pricing_backend"])
    short_vol_ok = bool(marks_ctx["short_vol_allowed"])
    if marks_ctx.get("forced_proxy_reason"):
        logger.warning("MARKS HONESTY: %s", marks_ctx["forced_proxy_reason"])

    if args.smoke:
        args.max_strategies = min(args.max_strategies, 80)
        args.from_year = max(args.from_year, 2015)

    zoo_path = Path(args.zoo)
    apply_filter = not args.no_proxy_filter
    # Under proxy pricing, filter on unless --no-proxy-filter
    # short_vol_evaluated: pure short-vol kinds may enter the meta universe
    if short_vol_ok:
        proxy_filter_applied = False
        short_vol_evaluated = True
    elif apply_filter:
        proxy_filter_applied = True
        short_vol_evaluated = False
    else:
        proxy_filter_applied = False
        short_vol_evaluated = True  # NORM VIOLATION under proxy pricing

    if args.build_grid or (not zoo_path.is_file() and not args.rescore_only):
        logger.info(
            "Building grid zoo max=%d marks_mode=%s (requested=%s)",
            max(args.max_strategies * 3, 2500),
            marks_mode,
            requested_marks_mode,
        )
        write_grid_zoo(
            zoo_path,
            max_strategies=max(args.max_strategies * 3, 2500),
            marks_mode=marks_mode,
            apply_proxy_short_vol_filter=proxy_filter_applied,
        )

    if args.rescore_only:
        # Zoo only for capital0/risk metadata — meta universe comes from full cache
        if zoo_path.is_file():
            z = json.loads(zoo_path.read_text(encoding="utf-8"))
            capital0 = float(z.get("capital0") or 100_000.0)
            risk = OptionsRiskConfig.from_mapping(z.get("risk") or {})
            zoo_meta = z
        else:
            capital0, risk, zoo_meta = 100_000.0, OptionsRiskConfig(), {}
        specs = []  # filled from cache after load
    else:
        specs, capital0, risk, zoo_meta = _specs_from_zoo(
            zoo_path,
            max_n=args.max_strategies,
            marks_mode=marks_mode,
            apply_proxy_filter=proxy_filter_applied,
        )

    unds = sorted(
        (
            {s.underlying.upper() for s in specs} | {"SPY", "QQQ", "IWM", "VIX", "VIX3M"}
            if specs
            else {"SPY", "QQQ", "IWM", "VIX", "VIX3M"}
        )
    )
    kinds = sorted({s.kind for s in specs}) if specs else []
    logger.info(
        "Strategies=%d underlyings=%d kinds=%s years=%d-%d marks_eff=%s requested=%s "
        "label=%s workers=%d rescore=%s proxy_filter=%s short_vol_ok=%s",
        len(specs),
        len(unds),
        kinds,
        args.from_year,
        args.to_year,
        marks_mode,
        requested_marks_mode,
        args.label_mode,
        args.workers,
        args.rescore_only,
        proxy_filter_applied,
        short_vol_ok,
    )
    if proxy_filter_applied:
        logger.info(
            "PROXY MARKS GATE: short-premium pure kinds excluded from meta evaluation "
            "(never claim OPRA edge from proxy_bs)"
        )
    elif short_vol_evaluated and not short_vol_ok:
        logger.warning(
            "NORM VIOLATION: --no-proxy-filter with proxy pricing — short-vol pure "
            "kinds included in meta (debug only; no OPRA claim)"
        )

    from paper_live.data.eodhd_client import build_eodhd_feed

    cache = Path(args.out) / "eodhd_cache"
    feed, sources = build_eodhd_feed(
        list(unds),
        start=args.eodhd_from,
        cache_dir=cache,
        min_history=60,
    )
    days = list(feed.days)
    logger.info(
        "Feed days %s→%s n=%d sources_eodhd=%d",
        days[0],
        days[-1],
        len(days),
        sum(1 for v in sources.values() if v == "eodhd_eod"),
    )

    year_windows = _years_range(args.from_year, args.to_year)
    clamped: List[Tuple[str, date, date]] = []
    for name, ws, we in year_windows:
        req_s = date.fromisoformat(ws)
        req_e = days[-1] if we.startswith("2099") else date.fromisoformat(we)
        s = next((d for d in days if d >= req_s), None)
        e = next((d for d in reversed(days) if d <= req_e), None)
        if s is None or e is None or s >= e:
            logger.warning("Skip year %s — no data", name)
            continue
        clamped.append((name, s, e))

    out_root = Path(args.out)
    latest = out_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    cache_file = latest / "sleeve_year_returns.json"

    sleeve_years: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    if (args.resume or args.rescore_only) and cache_file.is_file():
        raw_cache = json.loads(cache_file.read_text(encoding="utf-8"))
        sleeve_years = defaultdict(dict, {k: v for k, v in raw_cache.items()})
        logger.info("Loaded sleeve cache with %d strategies", len(sleeve_years))
    elif args.rescore_only:
        logger.error(" --rescore-only requires %s", cache_file)
        return 2

    # Replay data_label: real_chain only when short-vol claims are allowed (engine + backend)
    replay_label = (
        "real_chain"
        if short_vol_ok and not is_proxy_marks(pricing_backend)
        else MARKS_PROXY_BS
    )

    if not args.rescore_only:
        need: List[OptionStrategySpec] = []
        for sp in specs:
            ymap = sleeve_years.get(sp.id, {})
            missing = [n for n, _, _ in clamped if n not in ymap or ymap[n].get("error")]
            if missing:
                need.append(sp)
            elif sp.id not in sleeve_years:
                need.append(sp)

        logger.info("Sleeve jobs remaining: %d / %d", len(need), len(specs))
        t0 = time.time()
        done_specs = len(specs) - len(need)

        if need:
            clamped_iso = [(n, s.isoformat(), e.isoformat()) for n, s, e in clamped]
            risk_dict = risk.to_dict() if hasattr(risk, "to_dict") else {
                "max_portfolio_dd": risk.max_portfolio_dd,
                "max_single_day_drop": risk.max_single_day_drop,
                "max_margin_fraction": risk.max_margin_fraction,
                "hard_kill_enabled": risk.hard_kill_enabled,
                "max_contracts": getattr(risk, "max_contracts", 12),
            }

            if args.workers <= 1 or len(need) == 1:
                for si, sp in enumerate(need, 1):
                    sleeve_years[sp.id] = _eval_spec_years(
                        feed,
                        sp,
                        clamped,
                        capital0,
                        risk,
                        existing=sleeve_years.get(sp.id),
                        data_label=replay_label,
                    )
                    done_specs += 1
                    if si % 5 == 0 or si == len(need):
                        elapsed = time.time() - t0
                        rate = si / max(elapsed, 1e-6)
                        eta = (len(need) - si) / max(rate, 1e-6)
                        logger.info(
                            "Sleeve progress %d/%d (run %d) ETA %.1f min",
                            done_specs,
                            len(specs),
                            si,
                            eta / 60.0,
                        )
                        cache_file.write_text(
                            json.dumps(dict(sleeve_years), indent=2, default=str),
                            encoding="utf-8",
                        )
            else:
                payloads = []
                for sp in need:
                    payloads.append(
                        {
                            "spec": {
                                "id": sp.id,
                                "label": sp.label,
                                "kind": sp.kind,
                                "underlying": sp.underlying,
                                "dte_days": sp.dte_days,
                                "otm_pct": sp.otm_pct,
                                "wing_otm_pct": sp.wing_otm_pct,
                                "contracts": sp.contracts,
                                "max_portfolio_dd": sp.max_portfolio_dd,
                                "max_single_day_drop": sp.max_single_day_drop,
                                "max_margin_fraction": sp.max_margin_fraction,
                                "hard_kill_enabled": sp.hard_kill_enabled,
                                "meta": sp.meta,
                                "notes": sp.notes,
                            },
                            "existing": dict(sleeve_years.get(sp.id) or {}),
                        }
                    )
                n_workers = min(args.workers, len(payloads))
                logger.info("Starting ProcessPool workers=%d", n_workers)
                completed = 0
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(
                        list(unds),
                        args.eodhd_from,
                        str(cache.resolve()),
                        capital0,
                        risk_dict,
                        clamped_iso,
                        replay_label,
                    ),
                ) as ex:
                    futs = {ex.submit(_worker_run, p): p["spec"]["id"] for p in payloads}
                    for fut in as_completed(futs):
                        sid = futs[fut]
                        try:
                            rid, ymap = fut.result()
                            sleeve_years[rid] = ymap
                        except Exception as e:
                            logger.exception("Worker failed for %s: %s", sid, e)
                            sleeve_years[sid] = {
                                n: {
                                    "total_return": 0.0,
                                    "max_dd": 0.0,
                                    "n_opens": 0,
                                    "hard_kill": False,
                                    "error": str(e),
                                    "kind": "",
                                    "underlying": "",
                                    "dte_days": 30,
                                    "otm_pct": 0.05,
                                }
                                for n, _, _ in clamped
                            }
                        completed += 1
                        done_specs += 1
                        if completed % 10 == 0 or completed == len(payloads):
                            elapsed = time.time() - t0
                            rate = completed / max(elapsed, 1e-6)
                            eta = (len(payloads) - completed) / max(rate, 1e-6)
                            logger.info(
                                "Sleeve progress %d/%d (batch %d/%d) ETA %.1f min",
                                done_specs,
                                len(specs),
                                completed,
                                len(payloads),
                                eta / 60.0,
                            )
                            cache_file.write_text(
                                json.dumps(dict(sleeve_years), indent=2, default=str),
                                encoding="utf-8",
                            )

        cache_file.write_text(
            json.dumps(dict(sleeve_years), indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            "Phase A done in %.1f min — %d sleeves",
            (time.time() - t0) / 60.0,
            len(sleeve_years),
        )
    else:
        logger.info("Rescore-only: skipping Phase A sleeve replay")

    # Meta universe: full cache filtered by **kind** (rescore does not subsample via zoo max_n)
    if args.rescore_only:
        sleeve_for_meta = filter_sleeve_years_for_marks(
            sleeve_years,
            marks_mode,
            apply_filter=proxy_filter_applied,
            restrict_to_ids=None,  # full cache
            chain_engine_available=CHAIN_PRICING_ENGINE_AVAILABLE,
            pricing_backend=pricing_backend,
        )
        # Build specs/kinds/unds from filtered cache for reporting + unds feed already loaded
        specs = []
        unds_extra: set = set()
        for sid, ymap in sleeve_for_meta.items():
            kind = kind_from_sleeve_ymap(ymap, sid)
            und = "SPY"
            for cell in ymap.values():
                if isinstance(cell, dict) and cell.get("underlying"):
                    und = str(cell["underlying"])
                    break
            unds_extra.add(und.upper())
            specs.append(
                OptionStrategySpec(
                    id=sid, label=sid, kind=kind or "unknown", underlying=und
                )
            )
        kinds = sorted({s.kind for s in specs})
        if unds_extra:
            unds = sorted(set(unds) | unds_extra | {"SPY", "QQQ", "IWM", "VIX", "VIX3M"})
        logger.info(
            "Rescore meta universe: %d sleeves from full cache (proxy_filter=%s)",
            len(sleeve_for_meta),
            proxy_filter_applied,
        )
    else:
        # Phase A path: filter by kind always; optionally restrict to zoo sample IDs
        sample_ids = {sp.id for sp in specs} | {"G_CASH_CTRL"}
        sleeve_for_meta = filter_sleeve_years_for_marks(
            sleeve_years,
            marks_mode,
            apply_filter=proxy_filter_applied,
            restrict_to_ids=sample_ids if specs else None,
            chain_engine_available=CHAIN_PRICING_ENGINE_AVAILABLE,
            pricing_backend=pricing_backend,
        )
        # Defense: also drop any sleeve whose kind fails even if ID was in sample
        kinds = sorted(
            {
                kind_from_sleeve_ymap(ymap, sid)
                for sid, ymap in sleeve_for_meta.items()
            }
            - {""}
        )
        logger.info(
            "Meta universe sleeves: %d (marks_mode=%s proxy_filter=%s)",
            len(sleeve_for_meta),
            marks_mode,
            proxy_filter_applied,
        )

    # benchmarks
    benches: Dict[str, Dict[str, Optional[float]]] = {}
    for name, start_d, end_d in clamped:
        benches[name] = {
            "SPY": _bh(feed, "SPY", start_d, end_d),
            "QQQ": _bh(feed, "QQQ", start_d, end_d),
            "IWM": _bh(feed, "IWM", start_d, end_d),
        }

    cfg = MetaLabelConfig(min_train_years=3, take_threshold=0.55, label_mode=args.label_mode)
    caps = PortfolioCaps()

    wf_rows, port_rets, equal_top_rets, label_skip_spy_missing = _run_walk_forward(
        feed=feed,
        sleeve_years=sleeve_for_meta,
        clamped=clamped,
        benches=benches,
        cfg=cfg,
        caps=caps,
        top_k=int(args.top_k),
        marks_mode=marks_mode,
    )

    year_names = [n for n, _, _ in clamped]

    def _mean(d: Mapping[str, float], skip_warmup: bool = True) -> Optional[float]:
        vals = []
        for y, r in d.items():
            if skip_warmup and any(
                w.get("year") == y and w.get("mode") == "warmup" for w in wf_rows
            ):
                continue
            vals.append(r)
        return float(np.mean(vals)) if vals else None

    active_years = [
        w["year"] for w in wf_rows if w.get("mode") not in ("warmup", "no_candidates")
    ]
    port_list = [port_rets[y] for y in active_years if y in port_rets]
    spy_list = [benches[y]["SPY"] for y in active_years if benches[y].get("SPY") is not None]
    qqq_list = [benches[y]["QQQ"] for y in active_years if benches[y].get("QQQ") is not None]

    blend_list = []
    inv_w_list = []
    vs_blend_list = []
    for w in wf_rows:
        if w.get("mode") in ("warmup", "no_candidates"):
            continue
        if w.get("spy_cash_blend") is not None:
            blend_list.append(float(w["spy_cash_blend"]))
        if w.get("invested_weight") is not None:
            inv_w_list.append(float(w["invested_weight"]))
        if w.get("port_vs_spy_cash_blend") is not None:
            vs_blend_list.append(float(w["port_vs_spy_cash_blend"]))

    pos = [r for r in port_list if r > 0]
    max_upside_share = (
        float(max(pos) / sum(pos)) if pos and sum(pos) > 0 else None
    )

    ban_list = [
        "no single-name leverage",
        "no QQQ×2/NVDA×2 products",
        "never claim OPRA edge from proxy_bs",
    ]
    if proxy_filter_applied:
        ban_list.append("proxy: exclude IC/CCS/PCS/CSP from meta evaluation")
    elif short_vol_evaluated and not short_vol_ok:
        ban_list.append(BAN_RULE_NORM_VIOLATION)

    summary = {
        "generated_at": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "n_strategies": len(sleeve_for_meta),
        "kinds": kinds,
        "years": year_names,
        "active_years": active_years,
        "data_sources": sources,
        "marks_mode": marks_mode,
        "marks_mode_requested": requested_marks_mode,
        "pricing_backend": pricing_backend,
        "option_marks": option_marks_label,
        "chain_pricing_engine_available": CHAIN_PRICING_ENGINE_AVAILABLE,
        "forced_proxy": bool(marks_ctx.get("forced_proxy")),
        "forced_proxy_reason": marks_ctx.get("forced_proxy_reason"),
        "label_mode": args.label_mode,
        "label_skip_spy_missing": int(label_skip_spy_missing),
        "short_vol_allowed": short_vol_ok,
        "short_vol_evaluated": short_vol_evaluated,
        "proxy_filter_applied": proxy_filter_applied,
        "norm_violation": bool(short_vol_evaluated and not short_vol_ok),
        "one_per_underlying": True,
        "labels": {
            "underlyings": "eodhd_eod",
            "option_marks": option_marks_label,
            "marks_mode": marks_mode,
            "marks_mode_requested": requested_marks_mode,
            "pricing_backend": pricing_backend,
            "selection": "meta_label_gbt_walk_forward",
            "meta_label_mode": args.label_mode,
        },
        "ban": ban_list,
        "portfolio_mean_ret": float(np.mean(port_list)) if port_list else None,
        "spy_mean_ret": float(np.mean(spy_list)) if spy_list else None,
        "qqq_mean_ret": float(np.mean(qqq_list)) if qqq_list else None,
        "mean_invested_weight": float(np.mean(inv_w_list)) if inv_w_list else None,
        "spy_cash_blend_mean": float(np.mean(blend_list)) if blend_list else None,
        "portfolio_vs_spy_cash_blend": (
            float(np.mean(vs_blend_list)) if vs_blend_list else None
        ),
        "naive_top5_mean": _mean(equal_top_rets),
        "portfolio_vs_spy": (
            float(np.mean(port_list) - np.mean(spy_list))
            if port_list and spy_list
            else None
        ),
        "max_upside_year_share": max_upside_share,
        "walk_forward": wf_rows,
        "benchmarks": benches,
        "rescore_only": bool(args.rescore_only),
        "disclaimer": honesty_disclaimer(
            marks_mode,
            option_marks_label=option_marks_label,
            proxy_filter_applied=proxy_filter_applied,
            short_vol_evaluated=short_vol_evaluated,
            forced_proxy_reason=marks_ctx.get("forced_proxy_reason"),
        ),
    }

    (latest / "walk_forward.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    def pct(x):
        try:
            return f"{float(x):.2%}"
        except Exception:
            return "n/a"

    if proxy_filter_applied:
        short_vol_method = "**excluded** (proxy honesty filter applied)."
    elif short_vol_ok:
        short_vol_method = "**allowed** (real chain pricing engine active)."
    else:
        short_vol_method = (
            "**included (NORM VIOLATION / debug — `--no-proxy-filter` under proxy pricing).** "
            "No OPRA claim; short_vol_allowed=false."
        )

    lines = [
        "# Options portfolio + meta-label study (long history)",
        "",
        f"**Strategies:** {summary['n_strategies']} (grid / cache, no levered single-name)",
        f"**Kinds:** {', '.join(kinds) if kinds else 'from cache'}",
        f"**Years:** {year_names[0] if year_names else '?'}–{year_names[-1] if year_names else '?'} · active WF: {active_years}",
        f"**Data:** EODHD EOD underlyings · option_marks `{option_marks_label}` · "
        f"marks_mode_effective `{marks_mode}` · requested `{requested_marks_mode}` · "
        f"pricing_backend `{pricing_backend}`",
        f"**Meta label:** `{args.label_mode}` · one sleeve / underlying before caps · "
        f"label_skip_spy_missing={label_skip_spy_missing}",
        f"**Short-vol allowed (claims):** {short_vol_ok} · "
        f"**short_vol_evaluated:** {short_vol_evaluated} · "
        f"**proxy_filter_applied:** {proxy_filter_applied} · "
        f"**norm_violation:** {summary['norm_violation']}",
        "",
        "## Headline",
        "",
        "| Port mean | SPY mean | SPY·w+cash mean | vs blend | vs SPY full | QQQ mean | Naive top5 | Mean w |",
        "|-----------|----------|-----------------|----------|-------------|----------|------------|--------|",
        f"| {pct(summary['portfolio_mean_ret'])} | {pct(summary['spy_mean_ret'])} | "
        f"{pct(summary['spy_cash_blend_mean'])} | {pct(summary['portfolio_vs_spy_cash_blend'])} | "
        f"{pct(summary['portfolio_vs_spy'])} | {pct(summary['qqq_mean_ret'])} | "
        f"{pct(summary['naive_top5_mean'])} | {pct(summary['mean_invested_weight'])} |",
        "",
        "## Walk-forward annual",
        "",
        "| Year | Mode | Port | Naive5 | SPY | Blend | w | QQQ | N sel |",
        "|------|------|------|--------|-----|-------|---|-----|-------|",
    ]
    for w in wf_rows:
        lines.append(
            f"| {w.get('year')} | {w.get('mode')} | {pct(w.get('portfolio_return'))} | "
            f"{pct(w.get('naive_top5_return'))} | {pct(w.get('spy_bh'))} | "
            f"{pct(w.get('spy_cash_blend'))} | {pct(w.get('invested_weight'))} | "
            f"{pct(w.get('qqq_bh'))} | {w.get('n_selected', 0)} |"
        )
    lines += [
        "",
        "## Method",
        "",
        "1. Backtest **defined-risk / budgeted-debit** options sleeves (no ×2 single names).",
        f"2. Expanding walk-forward: meta predicts sleeves that **{args.label_mode}** next year "
        "(rows skipped if SPY year return missing for beat_spy/utility_excess).",
        "3. Rank by meta score → **one sleeve per underlying** → top-K → portfolio **caps**.",
        "4. Residual cash. Primary fair bench: **w·SPY + (1−w)·cash**; full SPY/QQQ secondary.",
        f"5. Short-vol pure kinds: {short_vol_method}",
        "",
        "---",
        summary["disclaimer"],
        "",
    ]
    (latest / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "DONE port_mean=%s spy_mean=%s blend=%s → %s",
        summary["portfolio_mean_ret"],
        summary["spy_mean_ret"],
        summary["spy_cash_blend_mean"],
        latest / "SUMMARY.md",
    )
    print(
        json.dumps(
            {
                "n_strategies": summary["n_strategies"],
                "kinds": kinds,
                "marks_mode": marks_mode,
                "marks_mode_requested": requested_marks_mode,
                "option_marks": option_marks_label,
                "label_mode": args.label_mode,
                "short_vol_allowed": short_vol_ok,
                "short_vol_evaluated": short_vol_evaluated,
                "proxy_filter_applied": proxy_filter_applied,
                "norm_violation": summary["norm_violation"],
                "portfolio_mean": summary["portfolio_mean_ret"],
                "spy_mean": summary["spy_mean_ret"],
                "spy_cash_blend_mean": summary["spy_cash_blend_mean"],
                "portfolio_vs_spy_cash_blend": summary["portfolio_vs_spy_cash_blend"],
                "qqq_mean": summary["qqq_mean_ret"],
                "vs_spy": summary["portfolio_vs_spy"],
                "mean_invested_weight": summary["mean_invested_weight"],
                "summary": str(latest / "SUMMARY.md"),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
