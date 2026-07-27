"""Optional diagnostic: today's Yahoo chain mid vs model BS mid.

Does **not** rewrite historical marks. Research calibration only.
On network failure: returns ``yahoo_chain_failed`` without inventing quotes.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from paper_live.options.bs import black_scholes_price
from paper_live.options.vol_surface import iv_from_surface


def _mid(q: Any) -> Optional[float]:
    if q is None:
        return None
    mid = getattr(q, "mid", None)
    if mid is not None and math.isfinite(float(mid)) and float(mid) > 0:
        return float(mid)
    bid = getattr(q, "bid", None)
    ask = getattr(q, "ask", None)
    try:
        if bid is not None and ask is not None and float(bid) > 0 and float(ask) > 0:
            return 0.5 * (float(bid) + float(ask))
    except (TypeError, ValueError):
        pass
    last = getattr(q, "last", None)
    if last is not None and math.isfinite(float(last)) and float(last) > 0:
        return float(last)
    return None


def diagnose_chain_vs_model(
    underlyings: Sequence[str] = ("SPY", "QQQ", "AAPL"),
    *,
    vix: Optional[float] = None,
    vix3m: Optional[float] = None,
    hv: float = 0.18,
    premium_mult: float = 1.15,
    max_quotes_per_side: int = 8,
) -> Dict[str, Any]:
    """
    Fetch Yahoo chain for each underlying; compare near ATM mid vs BS mid.

    Returns surface_error stats. Never fabricates chain on failure.
    """
    try:
        from paper_live.options.yahoo_chain import fetch_yahoo_option_chain as fetch_yahoo_chain, YahooChainError
    except Exception as e:  # pragma: no cover
        return {
            "ok": False,
            "label": "yahoo_chain_failed",
            "error": f"import_failed: {e}",
            "underlyings": {},
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
        }

    out: Dict[str, Any] = {
        "ok": True,
        "label": "chain_vs_model_diag",
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "underlyings": {},
        "aggregate": {},
        "notes": [
            "Diagnostic only — not used to rewrite historical proxy marks.",
            "Model IV: vix_surface when VIX given else proxy_hv.",
        ],
    }
    all_errs: List[float] = []
    any_ok = False

    for und in underlyings:
        try:
            snap = fetch_yahoo_chain(und)
        except Exception as e:
            out["underlyings"][und] = {
                "ok": False,
                "label": "yahoo_chain_failed",
                "error": str(e),
            }
            continue
        if not getattr(snap, "ok", True) or getattr(snap, "error", None):
            out["underlyings"][und] = {
                "ok": False,
                "label": "yahoo_chain_failed",
                "error": getattr(snap, "error", "unknown"),
            }
            continue

        spot = float(snap.spot) if snap.spot else None
        if spot is None or spot <= 0:
            out["underlyings"][und] = {
                "ok": False,
                "label": "yahoo_chain_failed",
                "error": "no_spot",
            }
            continue

        # nearest expiry
        exps = list(snap.expirations or [])
        if not exps:
            # derive from quotes
            for q in list(snap.calls or []) + list(snap.puts or []):
                if q.expiry and q.expiry not in exps:
                    exps.append(q.expiry)
        if not exps:
            out["underlyings"][und] = {
                "ok": False,
                "label": "yahoo_chain_failed",
                "error": "no_expirations",
            }
            continue

        exp0 = sorted(exps)[0]
        try:
            exp_d = date.fromisoformat(str(exp0)[:10])
        except ValueError:
            out["underlyings"][und] = {
                "ok": False,
                "label": "yahoo_chain_failed",
                "error": f"bad_expiry:{exp0}",
            }
            continue

        t_years = max((exp_d - date.today()).days, 0) / 365.0
        pairs: List[Dict[str, Any]] = []

        def consider(quotes: list, otype: str) -> None:
            ranked = sorted(
                quotes or [],
                key=lambda q: abs(float(q.strike or 0) - spot),
            )[:max_quotes_per_side]
            for q in ranked:
                mkt = _mid(q)
                if mkt is None:
                    continue
                k = float(q.strike)
                siv = iv_from_surface(
                    t_years=t_years,
                    spot=spot,
                    strike=k,
                    option_type=otype,
                    vix=vix,
                    vix3m=vix3m,
                    hv=hv,
                    premium_mult=premium_mult,
                )
                model = black_scholes_price(
                    spot, k, t_years, float(siv.iv), 0.0, option_type=otype
                )
                if model is None or not math.isfinite(model) or model <= 0:
                    continue
                err = (float(model) - mkt) / mkt
                pairs.append(
                    {
                        "type": otype,
                        "strike": k,
                        "expiry": str(exp0),
                        "market_mid": mkt,
                        "model_mid": float(model),
                        "rel_error": err,
                        "iv_source": siv.source,
                        "model_iv": siv.iv,
                        "yahoo_iv": getattr(q, "implied_volatility", None),
                    }
                )
                all_errs.append(err)

        consider(list(snap.calls or []), "call")
        consider(list(snap.puts or []), "put")
        if pairs:
            any_ok = True
            errs = [p["rel_error"] for p in pairs]
            out["underlyings"][und] = {
                "ok": True,
                "spot": spot,
                "expiry": str(exp0),
                "n_quotes": len(pairs),
                "mean_rel_error": float(sum(errs) / len(errs)),
                "median_rel_error": float(sorted(errs)[len(errs) // 2]),
                "bias_note": "positive mean_rel_error ⇒ model richer than market mid",
                "samples": pairs[:12],
            }
        else:
            out["underlyings"][und] = {
                "ok": False,
                "label": "yahoo_chain_failed",
                "error": "no_comparable_quotes",
            }

    if all_errs:
        out["aggregate"] = {
            "n": len(all_errs),
            "mean_rel_error": float(sum(all_errs) / len(all_errs)),
            "median_rel_error": float(sorted(all_errs)[len(all_errs) // 2]),
        }
    out["ok"] = any_ok
    if not any_ok:
        out["label"] = "yahoo_chain_failed"
    return out
