"""Research-only robustness metrics for Krypton.

No production trading decisions are made here. These helpers are used by the
research pipeline to reject fragile candidates and correct for multiple tests.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd


def concentration_metrics(trade_log: pd.DataFrame, equity_curve: pd.Series, initial_capital: float = 10_000.0) -> dict:
    pnl = trade_log.get("pnl", pd.Series(dtype=float)).astype(float) if not trade_log.empty else pd.Series(dtype=float)
    total_pnl = float(pnl.sum()) if len(pnl) else 0.0
    best_trade = float(pnl.max()) if len(pnl) else 0.0
    n_top = max(1, int(math.ceil(len(pnl) * 0.05))) if len(pnl) else 0
    top5 = float(pnl.nlargest(n_top).sum()) if n_top else 0.0

    if equity_curve.empty:
        daily_pnl = pd.Series(dtype=float)
        final_capital = initial_capital + total_pnl
    else:
        eq = equity_curve.astype(float).sort_index()
        daily_pnl = eq.diff().fillna(eq.iloc[0] - initial_capital)
        final_capital = float(eq.iloc[-1])

    def best_block(days: int) -> float:
        if daily_pnl.empty:
            return 0.0
        return float(daily_pnl.rolling(days, min_periods=1).sum().max())

    best5 = best_block(5)
    best20 = best_block(20)
    denom = abs(total_pnl) if abs(total_pnl) > 1e-12 else np.nan
    return {
        "return_without_best_trade": (final_capital - best_trade) / initial_capital - 1.0,
        "return_without_best_5d": (final_capital - best5) / initial_capital - 1.0,
        "return_without_best_20d": (final_capital - best20) / initial_capital - 1.0,
        "top_5pct_trade_pnl_share": float(top5 / denom) if np.isfinite(denom) else 0.0,
        "best_trade_pnl": best_trade,
        "best_5d_pnl": best5,
        "best_20d_pnl": best20,
    }


def correlation_report(data: dict, symbols: list[str], start, end, exposure: pd.DataFrame | None = None) -> dict:
    returns = {}
    for symbol in symbols:
        close = data[symbol]["df"].loc[start:end, "close"].astype(float)
        returns[symbol] = close.pct_change()
    ret_df = pd.DataFrame(returns).dropna(how="all")
    daily_corr = ret_df.corr()
    pos_corr = exposure.astype(float).corr() if exposure is not None and not exposure.empty else pd.DataFrame()
    return {"daily_return_correlation": daily_corr, "position_correlation": pos_corr}


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int) -> float:
    """Bailey/Lopez-de-Prado style DSR using an expected max Sharpe threshold."""
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 3 or r.std(ddof=1) <= 0:
        return 0.0
    sr = float(r.mean() / r.std(ddof=1) * math.sqrt(365.0))
    skew = float(r.skew()) if len(r) > 2 else 0.0
    kurt = float(r.kurt() + 3.0) if len(r) > 3 else 3.0
    n = len(r)
    trials = max(int(n_trials), 1)
    if trials <= 1:
        sr0 = 0.0
    else:
        nd = NormalDist()
        gamma = 0.5772156649015329
        z1 = nd.inv_cdf(1.0 - 1.0 / trials)
        z2 = nd.inv_cdf(1.0 - 1.0 / (trials * math.e))
        sr0 = ((1.0 - gamma) * z1 + gamma * z2) * math.sqrt(365.0 / max(n - 1, 1))
    variance = max((1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr) / max(n - 1, 1), 1e-12)
    z = (sr - sr0) / math.sqrt(variance)
    return float(NormalDist().cdf(z))


def white_reality_check(candidate_returns: dict[str, pd.Series], benchmark: pd.Series | None = None,
                        bootstrap_samples: int = 2000, block_size: int = 5, seed: int = 42) -> dict:
    """Circular-block White Reality Check on max mean return differential."""
    frame = pd.concat(candidate_returns, axis=1).dropna(how="all").fillna(0.0)
    if frame.empty:
        return {"p_value": 1.0, "observed": 0.0, "winner": None}
    if benchmark is None:
        diff = frame.copy()
    else:
        b = pd.Series(benchmark).reindex(frame.index).fillna(0.0)
        diff = frame.sub(b, axis=0)
    means = diff.mean()
    winner = str(means.idxmax())
    observed = float(means.max())
    centered = diff - means
    n = len(centered)
    rng = np.random.default_rng(seed)
    boot_max = np.empty(bootstrap_samples)
    block = max(1, min(int(block_size), n))
    arr = centered.to_numpy()
    for b in range(bootstrap_samples):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, n))
            idx.extend((start + j) % n for j in range(block))
        sample = arr[np.asarray(idx[:n])]
        boot_max[b] = np.nanmax(sample.mean(axis=0))
    p = float((1.0 + np.sum(boot_max >= observed)) / (bootstrap_samples + 1.0))
    return {"p_value": p, "observed": observed, "winner": winner}
