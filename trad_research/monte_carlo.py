"""Monte Carlo robustness on trade PnLs / daily returns (MET-02 / VAL-MC-01).

Industry-style methods (StrategyQuant / BuildAlpha / bootstrap practice):
- shuffle: permute trade order (same total PnL, path/DD varies)
- bootstrap: sample trades with replacement (total PnL varies)
- skip: randomly drop a fraction of trades
- block_bootstrap: resample blocks of daily returns

MC does **not** replace residual-vs-style or geo FROZEN gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from trad_research.risk_metrics import (
    cagr_from_equity,
    max_drawdown,
    sharpe_from_returns,
    sortino_ratio,
)

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class MCResult:
    method: str
    n_sims: int
    seed: int
    n_trades: int
    diagnostic_only: bool
    # Percentiles
    cagr_p5: float
    cagr_p50: float
    cagr_p95: float
    sortino_p5: float
    sortino_p50: float
    sortino_p95: float
    sharpe_p5: float
    sharpe_p50: float
    sharpe_p95: float
    mdd_p5: float  # least bad (closest to 0)
    mdd_p50: float
    mdd_p95: float  # worst (most negative)
    prob_mdd_worse_than: Dict[str, float] = field(default_factory=dict)
    total_pnl_constant: Optional[bool] = None  # True for pure shuffle
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pnls(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def equity_from_trade_pnls(
    pnls: np.ndarray,
    *,
    start_equity: float = 100_000.0,
) -> np.ndarray:
    """Compound equity path: treat pnls as fractional trade returns if |p|<2 median, else $."""
    p = np.asarray(pnls, dtype=float).ravel()
    if p.size == 0:
        return np.array([start_equity], dtype=float)
    # Heuristic: if max abs < 2, treat as simple returns; else dollar PnL
    if np.nanmax(np.abs(p)) < 2.0:
        eq = np.empty(p.size + 1, dtype=float)
        eq[0] = start_equity
        for i, r in enumerate(p):
            eq[i + 1] = eq[i] * (1.0 + r)
        return eq
    eq = np.empty(p.size + 1, dtype=float)
    eq[0] = start_equity
    for i, x in enumerate(p):
        eq[i + 1] = eq[i] + x
        if eq[i + 1] <= 0:
            eq[i + 1] = 1e-6
    return eq


def _metrics_from_equity(eq: np.ndarray) -> Tuple[float, float, float, float]:
    if eq.size < 3:
        return 0.0, 0.0, 0.0, 0.0
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    years = max(len(eq) - 1, 1) / 252.0
    return (
        cagr_from_equity(eq, years=years),
        sharpe_from_returns(rets),
        sortino_ratio(rets, mar=0.0),
        max_drawdown(eq),
    )


def _summarize_sims(
    cagrs: np.ndarray,
    sharpes: np.ndarray,
    sortinos: np.ndarray,
    mdds: np.ndarray,
    *,
    method: str,
    n_sims: int,
    seed: int,
    n_trades: int,
    diagnostic_only: bool,
    total_pnl_constant: Optional[bool],
    note: str = "",
) -> MCResult:
    def pct(a: np.ndarray, q: float) -> float:
        if a.size == 0:
            return 0.0
        return float(np.quantile(a, q))

    # MDD: more negative = worse; p95 is worse tail if sorted ascending
    return MCResult(
        method=method,
        n_sims=n_sims,
        seed=seed,
        n_trades=n_trades,
        diagnostic_only=diagnostic_only,
        cagr_p5=pct(cagrs, 0.05),
        cagr_p50=pct(cagrs, 0.50),
        cagr_p95=pct(cagrs, 0.95),
        sortino_p5=pct(sortinos, 0.05),
        sortino_p50=pct(sortinos, 0.50),
        sortino_p95=pct(sortinos, 0.95),
        sharpe_p5=pct(sharpes, 0.05),
        sharpe_p50=pct(sharpes, 0.50),
        sharpe_p95=pct(sharpes, 0.95),
        mdd_p5=pct(mdds, 0.05),  # less severe if mdd negative? actually min is worst
        mdd_p50=pct(mdds, 0.50),
        mdd_p95=pct(mdds, 0.95),  # for negative mdd, p5 is worst; fix below
        prob_mdd_worse_than={
            "0.30": float(np.mean(mdds <= -0.30)) if mdds.size else 0.0,
            "0.50": float(np.mean(mdds <= -0.50)) if mdds.size else 0.0,
            "0.60": float(np.mean(mdds <= -0.60)) if mdds.size else 0.0,
        },
        total_pnl_constant=total_pnl_constant,
        note=note,
    )


def _fix_mdd_percentiles(res: MCResult, mdds: np.ndarray) -> MCResult:
    """MDD is negative: worst = min; p5 of mdd distribution = severe tail."""
    if mdds.size == 0:
        return res
    res.mdd_p5 = float(np.quantile(mdds, 0.05))  # most negative side
    res.mdd_p50 = float(np.quantile(mdds, 0.50))
    res.mdd_p95 = float(np.quantile(mdds, 0.95))  # least severe
    # For gates we use "worst p5" as the severe tail
    return res


def mc_shuffle_trades(
    trade_pnls: ArrayLike,
    *,
    n_sims: int = 2000,
    seed: int = 42,
    start_equity: float = 100_000.0,
    min_trades_full: int = 50,
) -> MCResult:
    p = _pnls(trade_pnls)
    n = p.size
    diagnostic = n < min_trades_full
    if n < 2:
        return MCResult(
            method="shuffle",
            n_sims=0,
            seed=seed,
            n_trades=n,
            diagnostic_only=True,
            cagr_p5=0.0,
            cagr_p50=0.0,
            cagr_p95=0.0,
            sortino_p5=0.0,
            sortino_p50=0.0,
            sortino_p95=0.0,
            sharpe_p5=0.0,
            sharpe_p50=0.0,
            sharpe_p95=0.0,
            mdd_p5=0.0,
            mdd_p50=0.0,
            mdd_p95=0.0,
            total_pnl_constant=True,
            note="too few trades",
        )
    rng = np.random.default_rng(seed)
    cagrs, sharpes, sortinos, mdds = [], [], [], []
    base_sum = float(p.sum())
    for _ in range(int(n_sims)):
        order = rng.permutation(n)
        sim = p[order]
        assert abs(float(sim.sum()) - base_sum) < 1e-6 * max(1.0, abs(base_sum))
        eq = equity_from_trade_pnls(sim, start_equity=start_equity)
        c, sh, so, m = _metrics_from_equity(eq)
        cagrs.append(c)
        sharpes.append(sh)
        sortinos.append(so)
        mdds.append(m)
    res = _summarize_sims(
        np.array(cagrs),
        np.array(sharpes),
        np.array(sortinos),
        np.array(mdds),
        method="shuffle",
        n_sims=n_sims,
        seed=seed,
        n_trades=n,
        diagnostic_only=diagnostic,
        total_pnl_constant=True,
        note="Trade order shuffle; total PnL constant.",
    )
    return _fix_mdd_percentiles(res, np.array(mdds))


def mc_bootstrap_trades(
    trade_pnls: ArrayLike,
    *,
    n_sims: int = 2000,
    seed: int = 42,
    start_equity: float = 100_000.0,
    min_trades_full: int = 50,
) -> MCResult:
    p = _pnls(trade_pnls)
    n = p.size
    diagnostic = n < min_trades_full
    if n < 2:
        return MCResult(
            method="bootstrap",
            n_sims=0,
            seed=seed,
            n_trades=n,
            diagnostic_only=True,
            cagr_p5=0.0,
            cagr_p50=0.0,
            cagr_p95=0.0,
            sortino_p5=0.0,
            sortino_p50=0.0,
            sortino_p95=0.0,
            sharpe_p5=0.0,
            sharpe_p50=0.0,
            sharpe_p95=0.0,
            mdd_p5=0.0,
            mdd_p50=0.0,
            mdd_p95=0.0,
            total_pnl_constant=False,
            note="too few trades",
        )
    rng = np.random.default_rng(seed)
    cagrs, sharpes, sortinos, mdds = [], [], [], []
    for _ in range(int(n_sims)):
        idx = rng.integers(0, n, size=n)
        sim = p[idx]
        eq = equity_from_trade_pnls(sim, start_equity=start_equity)
        c, sh, so, m = _metrics_from_equity(eq)
        cagrs.append(c)
        sharpes.append(sh)
        sortinos.append(so)
        mdds.append(m)
    res = _summarize_sims(
        np.array(cagrs),
        np.array(sharpes),
        np.array(sortinos),
        np.array(mdds),
        method="bootstrap",
        n_sims=n_sims,
        seed=seed,
        n_trades=n,
        diagnostic_only=diagnostic,
        total_pnl_constant=False,
        note="Bootstrap trades with replacement; total PnL varies.",
    )
    return _fix_mdd_percentiles(res, np.array(mdds))


def mc_skip_trades(
    trade_pnls: ArrayLike,
    *,
    skip_frac: float = 0.10,
    n_sims: int = 2000,
    seed: int = 42,
    start_equity: float = 100_000.0,
    min_trades_full: int = 50,
) -> MCResult:
    p = _pnls(trade_pnls)
    n = p.size
    diagnostic = n < min_trades_full
    if n < 3:
        return mc_shuffle_trades(p, n_sims=0, seed=seed, min_trades_full=min_trades_full)
    rng = np.random.default_rng(seed)
    keep = max(1, int(round(n * (1.0 - float(skip_frac)))))
    cagrs, sharpes, sortinos, mdds = [], [], [], []
    for _ in range(int(n_sims)):
        idx = rng.choice(n, size=keep, replace=False)
        sim = p[idx]
        eq = equity_from_trade_pnls(sim, start_equity=start_equity)
        c, sh, so, m = _metrics_from_equity(eq)
        cagrs.append(c)
        sharpes.append(sh)
        sortinos.append(so)
        mdds.append(m)
    res = _summarize_sims(
        np.array(cagrs),
        np.array(sharpes),
        np.array(sortinos),
        np.array(mdds),
        method="skip",
        n_sims=n_sims,
        seed=seed,
        n_trades=n,
        diagnostic_only=diagnostic,
        total_pnl_constant=False,
        note=f"Randomly keep {keep}/{n} trades (skip_frac={skip_frac}).",
    )
    return _fix_mdd_percentiles(res, np.array(mdds))


def mc_block_bootstrap_returns(
    daily_returns: ArrayLike,
    *,
    block: int = 5,
    n_sims: int = 2000,
    seed: int = 42,
    start_equity: float = 100_000.0,
) -> MCResult:
    r = _pnls(daily_returns)
    n = r.size
    if n < block + 2:
        return MCResult(
            method="block_bootstrap",
            n_sims=0,
            seed=seed,
            n_trades=n,
            diagnostic_only=True,
            cagr_p5=0.0,
            cagr_p50=0.0,
            cagr_p95=0.0,
            sortino_p5=0.0,
            sortino_p50=0.0,
            sortino_p95=0.0,
            sharpe_p5=0.0,
            sharpe_p50=0.0,
            sharpe_p95=0.0,
            mdd_p5=0.0,
            mdd_p50=0.0,
            mdd_p95=0.0,
            total_pnl_constant=False,
            note="too few daily returns",
        )
    rng = np.random.default_rng(seed)
    block = max(int(block), 1)
    cagrs, sharpes, sortinos, mdds = [], [], [], []
    for _ in range(int(n_sims)):
        out: List[float] = []
        while len(out) < n:
            start = int(rng.integers(0, max(n - block + 1, 1)))
            out.extend(r[start : start + block].tolist())
        sim = np.array(out[:n], dtype=float)
        eq = np.empty(n + 1, dtype=float)
        eq[0] = start_equity
        for i, x in enumerate(sim):
            eq[i + 1] = max(eq[i] * (1.0 + x), 1e-6)
        c, sh, so, m = _metrics_from_equity(eq)
        cagrs.append(c)
        sharpes.append(sh)
        sortinos.append(so)
        mdds.append(m)
    res = _summarize_sims(
        np.array(cagrs),
        np.array(sharpes),
        np.array(sortinos),
        np.array(mdds),
        method="block_bootstrap",
        n_sims=n_sims,
        seed=seed,
        n_trades=n,
        diagnostic_only=n < 100,
        total_pnl_constant=False,
        note=f"Block bootstrap of daily returns (block={block}).",
    )
    return _fix_mdd_percentiles(res, np.array(mdds))


def trade_pnls_from_equity_diff(equity: ArrayLike) -> np.ndarray:
    """Fallback: use daily simple returns as pseudo-trade returns for MC."""
    eq = np.asarray(equity, dtype=float).ravel()
    eq = eq[np.isfinite(eq)]
    if eq.size < 3:
        return np.array([], dtype=float)
    rets = np.diff(eq) / eq[:-1]
    return rets[np.isfinite(rets)]
