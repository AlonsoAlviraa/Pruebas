"""Performance metrics for equity curves and trade lists."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceReport:
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_return: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    years: float
    final_equity: float
    start_equity: float
    benchmark_total_return: Optional[float] = None
    benchmark_cagr: Optional[float] = None
    benchmark_sharpe: Optional[float] = None
    excess_cagr: Optional[float] = None
    positive_year_frac: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def equity_metrics(
    equity: pd.Series,
    start_equity: float,
    trades: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
    positive_year_frac: Optional[float] = None,
) -> PerformanceReport:
    equity = equity.dropna().astype(float)
    if equity.empty:
        return PerformanceReport(
            n_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_return=0.0,
            total_return=0.0,
            cagr=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            years=0.0,
            final_equity=start_equity,
            start_equity=start_equity,
            positive_year_frac=positive_year_frac,
        )

    rets = equity.pct_change().dropna()
    final = float(equity.iloc[-1])
    total_return = final / start_equity - 1.0
    # Prefer calendar span when index is datetimes; else trading-day count
    years = 0.0
    try:
        idx = pd.to_datetime(equity.index, utc=True)
        span_days = max((idx.max() - idx.min()).days, 1)
        years = span_days / 365.25
    except Exception:
        n_days = max(len(equity) - 1, 1)
        years = n_days / 252.0
    years = max(years, 1.0 / 365.25)
    cagr = (final / start_equity) ** (1 / years) - 1.0 if years > 0 and final > 0 else 0.0
    vol = float(rets.std()) if len(rets) else 0.0
    mean = float(rets.mean()) if len(rets) else 0.0
    sharpe = (mean / vol * np.sqrt(252)) if vol > 1e-12 else 0.0
    downside = rets[rets < 0]
    dvol = float(downside.std()) if len(downside) else 0.0
    sortino = (mean / dvol * np.sqrt(252)) if dvol > 1e-12 else 0.0
    mdd = _max_drawdown(equity)
    calmar = (cagr / abs(mdd)) if mdd < -1e-12 else 0.0

    n_trades = 0
    win_rate = 0.0
    profit_factor = 0.0
    avg_return = 0.0
    if trades is not None and not trades.empty and "net_profit" in trades.columns:
        n_trades = len(trades)
        wins = trades[trades["net_profit"] > 0]
        losses = trades[trades["net_profit"] <= 0]
        win_rate = len(wins) / n_trades if n_trades else 0.0
        gp = float(wins["net_profit"].sum()) if len(wins) else 0.0
        gl = float(-losses["net_profit"].sum()) if len(losses) else 0.0
        profit_factor = (gp / gl) if gl > 1e-9 else (999.0 if gp > 0 else 0.0)
        if "trade_return" in trades.columns:
            avg_return = float(trades["trade_return"].mean())
        else:
            avg_return = float((trades["net_profit"] / trades["capital_used"].clip(lower=1e-9)).mean())

    b_total = b_cagr = b_sharpe = excess = None
    if benchmark is not None and not benchmark.empty:
        b = benchmark.reindex(equity.index).ffill().dropna()
        if len(b) > 2:
            b_final = float(b.iloc[-1] / b.iloc[0])
            b_total = b_final - 1.0
            b_years = max(len(b) - 1, 1) / 252.0
            b_cagr = b_final ** (1 / b_years) - 1.0 if b_years > 0 else 0.0
            br = b.pct_change().dropna()
            bv = float(br.std()) if len(br) else 0.0
            bm = float(br.mean()) if len(br) else 0.0
            b_sharpe = (bm / bv * np.sqrt(252)) if bv > 1e-12 else 0.0
            excess = cagr - b_cagr

    return PerformanceReport(
        n_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_return=avg_return,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        years=years,
        final_equity=final,
        start_equity=start_equity,
        benchmark_total_return=b_total,
        benchmark_cagr=b_cagr,
        benchmark_sharpe=b_sharpe,
        excess_cagr=excess,
        positive_year_frac=positive_year_frac,
    )


def acceptance_gates(
    report: PerformanceReport,
    min_years: float = 8.0,
    *,
    sortino_min: float = 0.50,
) -> Dict[str, bool]:
    """
    Research gates (two-tier intent):
    - Stretch: Sharpe >= 0.80
    - Acceptable long-only after costs: Sharpe >= 0.55 AND CAGR >= 10%
      (or beat benchmark Sharpe by 0.15)
    - Sortino (MAR=0 ann) >= sortino_min for promotion-aware research (MET-01)
    """
    sharpe_stretch = report.sharpe >= 0.80
    sharpe_acceptable = report.sharpe >= 0.55
    beat_bench = (
        report.benchmark_sharpe is not None
        and report.sharpe >= report.benchmark_sharpe + 0.15
    )
    sortino_ok = report.sortino >= sortino_min
    return {
        "years_ok": report.years >= min_years * 0.9,
        "sharpe_ok": sharpe_stretch or sharpe_acceptable or beat_bench,
        "sortino_ok": sortino_ok,
        "cagr_ok": report.cagr >= 0.10,  # "muy aceptable" bar after costs
        "mdd_ok": report.max_drawdown >= -0.35,
        "trades_ok": report.n_trades >= 150,
        "consistency_ok": (report.positive_year_frac is not None and report.positive_year_frac >= 0.60)
        or report.positive_year_frac is None,
        "stretch_sharpe_0_80": sharpe_stretch,
    }
