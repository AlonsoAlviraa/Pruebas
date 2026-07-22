"""Black–Scholes European option pricing (research proxy, no dividends by default)."""
from __future__ import annotations

import math
from typing import Literal

OptionType = Literal["call", "put"]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_price(
    spot: float,
    strike: float,
    t_years: float,
    vol: float,
    r: float = 0.02,
    *,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    """European BS price. ``t_years`` in years; ``vol`` annualized."""
    if spot <= 0 or strike <= 0:
        return 0.0
    if t_years <= 1e-8:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    vol = max(float(vol), 1e-6)
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    if option_type == "call":
        return spot * math.exp(-q * t_years) * _norm_cdf(d1) - strike * math.exp(
            -r * t_years
        ) * _norm_cdf(d2)
    return strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * math.exp(
        -q * t_years
    ) * _norm_cdf(-d1)


def bs_delta(
    spot: float,
    strike: float,
    t_years: float,
    vol: float,
    r: float = 0.02,
    *,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    if t_years <= 1e-8 or spot <= 0 or strike <= 0:
        if option_type == "call":
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    vol = max(float(vol), 1e-6)
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    if option_type == "call":
        return math.exp(-q * t_years) * _norm_cdf(d1)
    return -math.exp(-q * t_years) * _norm_cdf(-d1)
