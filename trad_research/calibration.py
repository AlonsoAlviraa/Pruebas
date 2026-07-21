"""MOD-04: threshold freeze on US OOS only (no foreign / IBEX data).

v1: discrete min_confidence grid on US years 2018–2021; freeze for later years
and all foreign transfer. No isotonic/Platt.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from trad_research.strategies import Strategy, get_strategy
from trad_research.strategy_runner import run_strategy_walk_forward

logger = logging.getLogger(__name__)

DEFAULT_CONF_GRID: tuple[float, ...] = (0.28, 0.32, 0.38, 0.42, 0.48)


@dataclass
class CalibrationResult:
    selected_min_confidence: float
    selection_window: tuple[int, int]
    freeze_from_year: int
    metric_name: str
    grid_scores: List[Dict[str, Any]]
    version: str = "us_oos_v1"


def _selection_score(result: Dict[str, Any]) -> float:
    """Pre-registered: max min(sharpe, 1.0) * 1_{mdd >= -0.40}."""
    rep = result["report"]
    if rep.max_drawdown < -0.40:
        return -999.0
    return min(float(rep.sharpe), 1.0)


def _with_confidence(strategy: Strategy, conf: float) -> Strategy:
    if is_dataclass(strategy) and any(f.name == "min_confidence" for f in fields(strategy)):
        return replace(strategy, min_confidence=float(conf))  # type: ignore[type-var]
    if hasattr(strategy, "min_confidence"):
        strategy.min_confidence = float(conf)  # type: ignore[attr-defined]
    return strategy


def calibrate_min_confidence_us(
    strategy_name: str = "aggressive_turbo",
    *,
    data_root: Union[str, Path] = "data",
    ticker_file: Union[str, Path] = "good_tickers_wf80.txt",
    universe_limit: int = 60,
    conf_grid: Sequence[float] = DEFAULT_CONF_GRID,
    select_first_year: int = 2018,
    select_last_year: int = 2021,
    preferred_index: Optional[Sequence[str]] = None,
) -> CalibrationResult:
    """Pick conf on US 2018–2021 only. Never pass foreign data roots."""
    root = Path(data_root)
    resolved = str(root.resolve()).lower()
    if "data_es" in resolved or "data_de" in resolved:
        raise ValueError("MOD-04 forbids calibration on foreign data roots")

    scores: List[Dict[str, Any]] = []
    best_conf = float(conf_grid[0])
    best_score = -1e18

    for conf in conf_grid:
        base = get_strategy(strategy_name)
        strat = _with_confidence(base, float(conf))
        logger.info("MOD-04 grid conf=%.2f on US %s–%s", conf, select_first_year, select_last_year)
        try:
            res = run_strategy_walk_forward(
                strat,
                data_root=root,
                ticker_file=Path(ticker_file),
                universe_limit=universe_limit,
                first_oos_year=select_first_year,
                last_oos_year=select_last_year,
                preferred_index=preferred_index,
            )
            sc = _selection_score(res)
            scores.append(
                {
                    "min_confidence": conf,
                    "score": sc,
                    "cagr": res["report"].cagr,
                    "sharpe": res["report"].sharpe,
                    "max_drawdown": res["report"].max_drawdown,
                }
            )
            if sc > best_score:
                best_score = sc
                best_conf = float(conf)
        except Exception as e:
            logger.exception("grid conf=%.2f failed: %s", conf, e)
            scores.append({"min_confidence": conf, "score": -999.0, "error": str(e)})

    return CalibrationResult(
        selected_min_confidence=best_conf,
        selection_window=(select_first_year, select_last_year),
        freeze_from_year=select_last_year + 1,
        metric_name="min(sharpe,1)*1_{mdd>=-0.40}",
        grid_scores=scores,
        version="us_oos_v1",
    )
