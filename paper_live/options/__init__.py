"""Paper options research (proxy BS on free OHLCV). Virtual capital only."""

from paper_live.options.bs import black_scholes_price, bs_delta
from paper_live.options.strategies import OptionStrategySpec, list_builtin_specs
from paper_live.options.vol_proxy import historical_vol, iv_proxy_from_hv

__all__ = [
    "black_scholes_price",
    "bs_delta",
    "historical_vol",
    "iv_proxy_from_hv",
    "OptionStrategySpec",
    "list_builtin_specs",
]
