"""Research-only Cross-Asset Alpha v2 for Krypton.

Frozen hypotheses, evaluated independently before combination:
1) persistent time-series momentum (all 30/90/180d returns positive),
2) smooth volatility-regime scaling (20d vol vs its trailing 90d median),
3) smooth cross-asset dispersion scaling (30d return dispersion vs trailing 180d median),
4) persistent UP->UP regime (asset above SMA200 now and 20 bars ago).

All signals use completed daily closes and execute at the next daily open through
``simulate_alpha_v2``. Long-only Spot, no leverage, cash allowed.
"""
from __future__ import annotations

from math import sqrt
import numpy as np
import pandas as pd

import adaptive_portfolio as ap
import cross_asset_allocation as ca
from config import ENTRY_SLIPPAGE_PCT, EXIT_SLIPPAGE_PCT, FEE_RATE

INITIAL_CAPITAL = ap.INITIAL_CAPITAL
MOMENTUM_WINDOWS = (30, 90, 180)
VOL_WINDOW = 20
VOL_REGIME_WINDOW = 90
COV_WINDOW = 60
DISPERSION_LOOKBACK = 180
REGIME_PERSISTENCE_BARS = 20


def _history(data, symbols, ts):
    return ca._aligned_close(data, symbols).loc[:ts]


def _raw_state(data, symbols, ts):
    close = _history(data, symbols, ts)
    need = max(200 + REGIME_PERSISTENCE_BARS, max(MOMENTUM_WINDOWS),
               VOL_REGIME_WINDOW + VOL_WINDOW, DISPERSION_LOOKBACK + 30,
               COV_WINDOW) + 1
    if len(close) < need:
        return close, {}
    state = {}
    for s in symbols:
        series = close[s].dropna()
        if ts not in series.index or len(series.loc[:ts]) < need:
            continue
        hist = series.loc[:ts]
        sma = data[s]["sma200"].reindex(hist.index)
        if pd.isna(sma.loc[ts]):
            continue
        moms = {w: float(hist.loc[ts] / hist.shift(w).loc[ts] - 1.0)
                for w in MOMENTUM_WINDOWS}
        rv_series = hist.pct_change().rolling(VOL_WINDOW).std() * sqrt(365.0)
        rv = float(rv_series.loc[ts])
        rv_med = float(rv_series.tail(VOL_REGIME_WINDOW).median())
        if not all(np.isfinite(v) for v in [*moms.values(), rv, rv_med]) or rv <= 0 or rv_med <= 0:
            continue
        old_ts = hist.index[-1 - REGIME_PERSISTENCE_BARS]
        old_sma = sma.loc[old_ts] if old_ts in sma.index else np.nan
        state[s] = {
            "price": float(hist.loc[ts]), "sma": float(sma.loc[ts]),
            "old_price": float(hist.loc[old_ts]),
            "old_sma": float(old_sma) if pd.notna(old_sma) else np.nan,
            "moms": moms, "rv": rv, "rv_med": rv_med,
        }
    return close, state


def _dispersion_scale(close, ts):
    r30 = close.pct_change(30)
    dispersion = r30.std(axis=1, skipna=True)
    current = float(dispersion.loc[ts]) if ts in dispersion.index else np.nan
    median = float(dispersion.loc[:ts].tail(DISPERSION_LOOKBACK).median())
    if not np.isfinite(current) or not np.isfinite(median) or current <= 0 or median <= 0:
        return 0.0
    return float(min(1.0, median / current))


def target_weights_v2(data, symbols, ts, *, target_vol=0.15, top_n=2,
                      min_selected=2, use_tsmom=False, use_vol_regime=False,
                      use_dispersion=False, use_persistent_regime=False):
    """Causal Alpha-v2 target weights; feature switches are research ablations."""
    close, state = _raw_state(data, symbols, ts)
    if not state:
        return {s: 0.0 for s in symbols}

    scores, vols, vol_scales = {}, {}, {}
    for s, st in state.items():
        if st["price"] <= st["sma"]:
            continue
        moms = st["moms"]
        if use_tsmom and not all(moms[w] > 0 for w in MOMENTUM_WINDOWS):
            continue
        if use_persistent_regime:
            if not np.isfinite(st["old_sma"]) or st["old_price"] <= st["old_sma"]:
                continue
        score = float(np.mean([moms[w] for w in MOMENTUM_WINDOWS]))
        if score <= 0:
            continue
        scores[s] = score
        vols[s] = st["rv"]
        vol_scales[s] = min(1.0, st["rv_med"] / st["rv"]) if use_vol_regime else 1.0

    selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
    if len(selected) < min_selected:
        return {s: 0.0 for s in symbols}

    inv = np.array([1.0 / vols[s] for s in selected], dtype=float)
    base = inv / inv.sum()
    returns = close[selected].pct_change().dropna().tail(COV_WINDOW)
    if len(returns) < max(20, COV_WINDOW // 2):
        return {s: 0.0 for s in symbols}
    cov = returns.cov().to_numpy(dtype=float) * 365.0
    port_vol = sqrt(max(float(base @ cov @ base), 0.0))
    scale = min(1.0, target_vol / port_vol) if port_vol > 0 else 0.0

    if use_vol_regime:
        scale *= float(np.average([vol_scales[s] for s in selected], weights=base))
    if use_dispersion:
        scale *= _dispersion_scale(close, ts)

    out = {s: 0.0 for s in symbols}
    for s, w in zip(selected, base * scale):
        out[s] = float(w)
    return out


def simulate_alpha_v2(data, symbols, start, end, *, rebalance_days=45,
                      target_vol=0.15, top_n=2, min_selected=2, **feature_flags):
    """Fully funded Alpha-v2 sleeve with next-open execution and explicit costs."""
    start_ts, end_ts = ap.as_utc(start), ap.as_utc(end)
    calendar = sorted(set.intersection(*[
        set(data[s]["df"].loc[start_ts:end_ts].index) for s in symbols
    ]))
    if not calendar:
        empty = pd.Series(dtype=float)
        return {**ap.performance_metrics(empty), "equity_curve": empty,
                "rebalance_log": pd.DataFrame()}

    cash = INITIAL_CAPITAL
    qty = {s: 0.0 for s in symbols}
    pending = None; pending_signal_time = None; last_signal_i = None
    points, logs = [], []

    for i, ts in enumerate(calendar):
        if pending is not None:
            pre_eq = ca._mark(cash, qty, data, ts, "open")
            opens = {s: float(data[s]["df"].loc[ts, "open"]) for s in symbols}
            for s in symbols:
                target_value = pre_eq * float(pending.get(s, 0.0))
                current_value = qty[s] * opens[s]
                if current_value > target_value and qty[s] > 0:
                    sell_qty = min(qty[s], (current_value - target_value) / opens[s])
                    px = opens[s] * (1.0 - EXIT_SLIPPAGE_PCT)
                    gross = sell_qty * px
                    cash += gross * (1.0 - FEE_RATE); qty[s] -= sell_qty
            eq_after_sells = ca._mark(cash, qty, data, ts, "open")
            for s in symbols:
                target_value = eq_after_sells * float(pending.get(s, 0.0))
                current_value = qty[s] * opens[s]
                deficit = max(0.0, target_value - current_value)
                if deficit <= 0 or cash <= 0:
                    continue
                px = opens[s] * (1.0 + ENTRY_SLIPPAGE_PCT)
                spend = min(deficit, cash / (1.0 + FEE_RATE))
                buy_qty = spend / px
                cash -= buy_qty * px * (1.0 + FEE_RATE); qty[s] += buy_qty
            logs.append({"signal_time": pending_signal_time, "execution_time": ts,
                         "target_gross": sum(pending.values()),
                         "selected": sum(float(v) > 0 for v in pending.values()),
                         **{f"target_{s}": pending.get(s, 0.0) for s in symbols}})
            pending = None

        points.append((ts, ca._mark(cash, qty, data, ts, "close")))
        if last_signal_i is None or i - last_signal_i >= rebalance_days:
            pending = target_weights_v2(data, symbols, ts, target_vol=target_vol,
                                        top_n=top_n, min_selected=min_selected,
                                        **feature_flags)
            pending_signal_time = ts; last_signal_i = i

    ts = calendar[-1]
    for s in symbols:
        if qty[s] <= 0:
            continue
        px = float(data[s]["df"].loc[ts, "close"]) * (1.0 - EXIT_SLIPPAGE_PCT)
        cash += qty[s] * px * (1.0 - FEE_RATE); qty[s] = 0.0
    points[-1] = (ts, cash)
    equity = pd.Series([v for _, v in points], index=[t for t, _ in points], dtype=float)
    log = pd.DataFrame(logs)
    if not log.empty and not bool((pd.to_datetime(log["execution_time"], utc=True) >
                                   pd.to_datetime(log["signal_time"], utc=True)).all()):
        raise AssertionError("look-ahead detectado no Alpha v2")
    return {**ap.performance_metrics(equity), "equity_curve": equity, "rebalance_log": log}
