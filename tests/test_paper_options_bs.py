"""Unit tests for options BS proxy (no network)."""
from __future__ import annotations

import math

from paper_live.options.bs import black_scholes_price, bs_delta
from paper_live.options.vol_proxy import historical_vol, iv_proxy_from_hv
import pandas as pd
import numpy as np


def test_bs_call_put_parity_rough():
    s, k, t, v, r = 100.0, 100.0, 0.25, 0.2, 0.01
    c = black_scholes_price(s, k, t, v, r, option_type="call")
    p = black_scholes_price(s, k, t, v, r, option_type="put")
    # C - P ≈ S - K e^{-rt}
    lhs = c - p
    rhs = s - k * math.exp(-r * t)
    assert abs(lhs - rhs) < 1e-4


def test_bs_expiry_intrinsic():
    assert black_scholes_price(110, 100, 0.0, 0.2, option_type="call") == 10.0
    assert black_scholes_price(90, 100, 0.0, 0.2, option_type="put") == 10.0


def test_delta_call_positive():
    d = bs_delta(100, 100, 0.3, 0.25, option_type="call")
    assert 0.4 < d < 0.7


def test_hv_and_iv_proxy():
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80)))
    hv = historical_vol(pd.Series(px), window=20)
    assert math.isfinite(hv) and hv > 0
    iv = iv_proxy_from_hv(hv, premium_mult=1.15)
    assert iv > hv
