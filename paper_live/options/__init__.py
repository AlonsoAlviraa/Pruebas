"""Paper options research (proxy BS on free OHLCV). Virtual capital only."""

from paper_live.options.bs import black_scholes_price, bs_delta
from paper_live.options.management import (
    ManagementConfig,
    apply_bid_haircut,
    management_from_meta,
    should_stop_loss,
    should_take_profit,
)
from paper_live.options.metrics import cvar, metrics_from_curve
from paper_live.options.risk import OptionsRiskConfig, margin_at_risk_per_contract, size_contracts
from paper_live.options.strategies import OptionStrategySpec, list_builtin_specs
from paper_live.options.vol_proxy import historical_vol, iv_proxy_from_hv
from paper_live.options.vol_surface import SurfaceIV, iv_from_surface

__all__ = [
    "black_scholes_price",
    "bs_delta",
    "historical_vol",
    "iv_proxy_from_hv",
    "iv_from_surface",
    "SurfaceIV",
    "OptionStrategySpec",
    "list_builtin_specs",
    "OptionsRiskConfig",
    "margin_at_risk_per_contract",
    "size_contracts",
    "cvar",
    "metrics_from_curve",
    "ManagementConfig",
    "management_from_meta",
    "apply_bid_haircut",
    "should_take_profit",
    "should_stop_loss",
]
