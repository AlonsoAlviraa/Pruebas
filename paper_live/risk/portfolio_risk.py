"""Portfolio risk metrics and soft de-risk sizing (LIV-06)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RiskConfig:
    """Paper risk knobs (from strategy_freeze.risk_paper)."""

    max_portfolio_dd: float = 0.18
    dd_soft_scale: float = 0.5
    dd_soft_trigger_frac: float = 0.5
    kill_dd_from_start: float = 0.15
    kill_rolling_sharpe_20d: float = -1.0
    max_daily_new_entries: int = 5
    ticker_max_capital_pct: float = 0.12
    rolling_sharpe_window: int = 20
    max_leverage: float = 1.0
    # Don't trip sharpe kill until we have enough daily returns (avoids false hard-kills)
    min_returns_for_sharpe_kill: int = 40
    # Optional: disable sharpe kill entirely (None)
    enable_sharpe_kill: bool = True

    @classmethod
    def from_risk_paper(
        cls,
        risk_paper: Mapping[str, Any],
        *,
        max_leverage: float = 1.0,
    ) -> "RiskConfig":
        rp = dict(risk_paper or {})
        return cls(
            max_portfolio_dd=float(rp.get("max_portfolio_dd", 0.18)),
            dd_soft_scale=float(rp.get("dd_soft_scale", 0.5)),
            dd_soft_trigger_frac=float(rp.get("dd_soft_trigger_frac", 0.5)),
            kill_dd_from_start=float(rp.get("kill_dd_from_start", 0.15)),
            kill_rolling_sharpe_20d=float(rp.get("kill_rolling_sharpe_20d", -1.0)),
            max_daily_new_entries=int(rp.get("max_daily_new_entries", 5)),
            ticker_max_capital_pct=float(rp.get("ticker_max_capital_pct", 0.12)),
            rolling_sharpe_window=int(rp.get("rolling_sharpe_window", 20)),
            max_leverage=float(max_leverage),
            min_returns_for_sharpe_kill=int(rp.get("min_returns_for_sharpe_kill", 40)),
            enable_sharpe_kill=bool(rp.get("enable_sharpe_kill", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_portfolio_dd": self.max_portfolio_dd,
            "dd_soft_scale": self.dd_soft_scale,
            "dd_soft_trigger_frac": self.dd_soft_trigger_frac,
            "kill_dd_from_start": self.kill_dd_from_start,
            "kill_rolling_sharpe_20d": self.kill_rolling_sharpe_20d,
            "max_daily_new_entries": self.max_daily_new_entries,
            "ticker_max_capital_pct": self.ticker_max_capital_pct,
            "rolling_sharpe_window": self.rolling_sharpe_window,
            "max_leverage": self.max_leverage,
            "min_returns_for_sharpe_kill": self.min_returns_for_sharpe_kill,
            "enable_sharpe_kill": self.enable_sharpe_kill,
        }


@dataclass(frozen=True)
class RiskSnapshot:
    equity: float
    cash: float
    capital0: float
    peak_equity: float
    dd_from_peak: float  # negative when underwater, e.g. -0.12
    dd_from_start: float
    size_scale: float  # 1.0 normal; <1 soft de-risk
    rolling_sharpe_20d: Optional[float]
    gross_exposure: float
    leverage: float
    block_new_entries: bool
    hard_kill: bool
    reasons: tuple

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "capital0": self.capital0,
            "peak_equity": self.peak_equity,
            "dd_from_peak": self.dd_from_peak,
            "dd_from_start": self.dd_from_start,
            "size_scale": self.size_scale,
            "rolling_sharpe_20d": self.rolling_sharpe_20d,
            "gross_exposure": self.gross_exposure,
            "leverage": self.leverage,
            "block_new_entries": self.block_new_entries,
            "hard_kill": self.hard_kill,
            "reasons": list(self.reasons),
            "capital_label": "VIRTUAL",
        }


def rolling_sharpe(returns: Sequence[float], window: int = 20) -> Optional[float]:
    """Annualized Sharpe from daily simple returns (window)."""
    if window < 2 or len(returns) < window:
        return None
    arr = np.asarray(list(returns)[-window:], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < window:
        return None
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd <= 1e-12:
        return 0.0 if abs(mu) < 1e-12 else (10.0 if mu > 0 else -10.0)
    return float(mu / sd * np.sqrt(252.0))


class PortfolioRisk:
    """Track equity path and compute risk snapshot + soft size scale."""

    def __init__(self, config: RiskConfig, *, capital0: float):
        self.config = config
        self.capital0 = float(capital0)
        self.peak_equity = float(capital0)
        self.equity_history: List[float] = [float(capital0)]
        self.daily_returns: List[float] = []

    def update_equity(self, equity: float) -> None:
        eq = float(equity)
        if eq > self.peak_equity:
            self.peak_equity = eq
        prev = self.equity_history[-1] if self.equity_history else self.capital0
        if prev > 0:
            self.daily_returns.append(eq / prev - 1.0)
        else:
            self.daily_returns.append(0.0)
        self.equity_history.append(eq)

    def soft_size_scale(self, dd_from_peak: float) -> float:
        """At soft trigger (e.g. half of max DD), scale size by dd_soft_scale."""
        max_dd = abs(float(self.config.max_portfolio_dd))
        if max_dd <= 0:
            return 1.0
        # dd_from_peak is <= 0 when underwater
        depth = -float(dd_from_peak)
        trigger = max_dd * float(self.config.dd_soft_trigger_frac)
        if depth < trigger - 1e-12:
            return 1.0
        # linear toward dd_soft_scale between trigger and max_dd
        if depth >= max_dd:
            return float(self.config.dd_soft_scale)
        frac = (depth - trigger) / max(max_dd - trigger, 1e-9)
        # at trigger: frac=0 → scale=1.0 but we want soft start: use min step
        frac = max(frac, 0.01)
        scale = 1.0 - frac * (1.0 - float(self.config.dd_soft_scale))
        return float(max(self.config.dd_soft_scale, min(1.0, scale)))

    def snapshot(
        self,
        *,
        equity: float,
        cash: float,
        gross_exposure: float = 0.0,
    ) -> RiskSnapshot:
        eq = float(equity)
        peak = max(self.peak_equity, eq, self.capital0)
        dd_peak = eq / peak - 1.0 if peak > 0 else 0.0
        dd_start = eq / self.capital0 - 1.0 if self.capital0 > 0 else 0.0
        sh = rolling_sharpe(self.daily_returns, self.config.rolling_sharpe_window)
        size_scale = self.soft_size_scale(dd_peak)
        lev = (gross_exposure / eq) if eq > 0 else 0.0

        reasons: List[str] = []
        block = False
        hard = False

        if dd_peak <= -abs(self.config.max_portfolio_dd):
            block = True
            hard = True
            reasons.append(
                f"dd_from_peak={dd_peak:.2%} <= -{abs(self.config.max_portfolio_dd):.0%}"
            )
        if dd_start <= -abs(self.config.kill_dd_from_start):
            block = True
            hard = True
            reasons.append(
                f"dd_from_start={dd_start:.2%} <= -{abs(self.config.kill_dd_from_start):.0%}"
            )
        if (
            self.config.enable_sharpe_kill
            and sh is not None
            and len(self.daily_returns) >= int(self.config.min_returns_for_sharpe_kill)
            and sh < float(self.config.kill_rolling_sharpe_20d)
        ):
            block = True
            hard = True
            reasons.append(
                f"rolling_sharpe_20d={sh:.2f} < {self.config.kill_rolling_sharpe_20d}"
            )
        if lev > float(self.config.max_leverage) + 1e-9:
            block = True
            reasons.append(f"leverage={lev:.2f} > {self.config.max_leverage}")

        if size_scale < 1.0 and not hard:
            reasons.append(f"soft_derisk size_scale={size_scale:.2f}")

        return RiskSnapshot(
            equity=eq,
            cash=float(cash),
            capital0=self.capital0,
            peak_equity=peak,
            dd_from_peak=float(dd_peak),
            dd_from_start=float(dd_start),
            size_scale=float(size_scale),
            rolling_sharpe_20d=sh,
            gross_exposure=float(gross_exposure),
            leverage=float(lev),
            block_new_entries=block,
            hard_kill=hard,
            reasons=tuple(reasons),
        )
