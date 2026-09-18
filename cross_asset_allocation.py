"""Research-only cross-asset trend/relative-strength allocator for Krypton.

Signals use completed daily closes. Rebalances execute at the next daily open.
The allocator is long-only Spot, never levered, and may hold cash.
"""
from __future__ import annotations

from math import sqrt
import numpy as np
import pandas as pd

import adaptive_portfolio as ap
from config import ENTRY_SLIPPAGE_PCT, EXIT_SLIPPAGE_PCT, FEE_RATE

INITIAL_CAPITAL = ap.INITIAL_CAPITAL


def _aligned_close(data, symbols):
    return pd.concat(
        {s: data[s]["df"]["close"].astype(float).sort_index() for s in symbols},
        axis=1,
    ).sort_index()


def target_weights(data, symbols, ts, *, target_vol=0.20, top_n=2,
                   momentum_windows=(30, 90, 180), vol_window=20,
                   cov_window=60):
    """Compute causal weights from information available at close ``ts``."""
    if target_vol <= 0 or top_n < 1:
        raise ValueError("target_vol e top_n precisam ser positivos")
    close = _aligned_close(data, symbols).loc[:ts]
    if len(close) < max(200, max(momentum_windows), cov_window) + 1:
        return {s: 0.0 for s in symbols}

    scores = {}
    vols = {}
    for s in symbols:
        series = close[s].dropna()
        if ts not in series.index:
            continue
        sma = data[s]["sma200"].reindex(series.index).loc[ts]
        if pd.isna(sma) or float(series.loc[ts]) <= float(sma):
            continue
        moms = []
        for w in momentum_windows:
            if len(series.loc[:ts]) <= w:
                moms = []
                break
            v = float(series.loc[ts] / series.shift(w).loc[ts] - 1.0)
            if not np.isfinite(v):
                moms = []
                break
            moms.append(v)
        if not moms:
            continue
        score = float(np.mean(moms))
        if score <= 0:
            continue
        rv = float(series.pct_change().rolling(vol_window).std().loc[ts] * sqrt(365.0))
        if not np.isfinite(rv) or rv <= 0:
            continue
        scores[s], vols[s] = score, rv

    selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
    if not selected:
        return {s: 0.0 for s in symbols}

    inv = np.array([1.0 / vols[s] for s in selected], dtype=float)
    base = inv / inv.sum()

    returns = close[selected].pct_change().dropna().tail(cov_window)
    if len(returns) < max(20, cov_window // 2):
        return {s: 0.0 for s in symbols}
    cov = returns.cov().to_numpy(dtype=float) * 365.0
    port_var = float(base @ cov @ base)
    port_vol = sqrt(max(port_var, 0.0))
    scale = min(1.0, target_vol / port_vol) if port_vol > 0 else 0.0

    out = {s: 0.0 for s in symbols}
    for s, w in zip(selected, base * scale):
        out[s] = float(w)
    return out


def _mark(cash, qty, data, ts, field="close"):
    equity = float(cash)
    for s, q in qty.items():
        if q == 0:
            continue
        df = data[s]["df"]
        if ts in df.index:
            px = float(df.loc[ts, field])
        else:
            hist = df.loc[:ts]
            if hist.empty:
                continue
            px = float(hist["close"].iloc[-1])
        equity += q * px
    return equity


def simulate_allocator(data, symbols, start, end, *, target_vol=0.20,
                       top_n=2, rebalance_days=7):
    """Simulate a fully funded allocator with explicit sell/buy costs."""
    start_ts, end_ts = ap.as_utc(start), ap.as_utc(end)
    calendar = sorted(set.intersection(*[
        set(data[s]["df"].loc[start_ts:end_ts].index) for s in symbols
    ]))
    if not calendar:
        empty = pd.Series(dtype=float)
        return {**ap.performance_metrics(empty), "equity_curve": empty,
                "rebalance_log": pd.DataFrame(), "allocations": pd.DataFrame()}

    cash = INITIAL_CAPITAL
    qty = {s: 0.0 for s in symbols}
    pending = None
    pending_signal_time = None
    last_signal_i = None
    equity_points, logs, alloc_rows = [], [], []

    for i, ts in enumerate(calendar):
        # Execute weights decided at the previous completed close.
        if pending is not None:
            pre_eq = _mark(cash, qty, data, ts, "open")
            opens = {s: float(data[s]["df"].loc[ts, "open"]) for s in symbols}

            # Sell excess first.
            for s in symbols:
                target_value = pre_eq * float(pending.get(s, 0.0))
                current_value = qty[s] * opens[s]
                if current_value > target_value and qty[s] > 0:
                    sell_qty = min(qty[s], (current_value - target_value) / opens[s])
                    px = opens[s] * (1.0 - EXIT_SLIPPAGE_PCT)
                    gross = sell_qty * px
                    fee = gross * FEE_RATE
                    cash += gross - fee
                    qty[s] -= sell_qty

            # Recompute equity after sells, then buy deficits with available cash.
            eq_after_sells = _mark(cash, qty, data, ts, "open")
            for s in symbols:
                target_value = eq_after_sells * float(pending.get(s, 0.0))
                current_value = qty[s] * opens[s]
                deficit = max(0.0, target_value - current_value)
                if deficit <= 0 or cash <= 0:
                    continue
                px = opens[s] * (1.0 + ENTRY_SLIPPAGE_PCT)
                spend = min(deficit, cash / (1.0 + FEE_RATE))
                buy_qty = spend / px
                debit = buy_qty * px * (1.0 + FEE_RATE)
                cash -= debit
                qty[s] += buy_qty

            post_eq = _mark(cash, qty, data, ts, "open")
            logs.append({
                "signal_time": pending_signal_time, "execution_time": ts,
                "target_vol": target_vol, "rebalance_days": rebalance_days,
                "pre_equity": pre_eq, "post_equity": post_eq,
                "target_gross": sum(pending.values()),
                **{f"target_{s}": pending.get(s, 0.0) for s in symbols},
            })
            pending = None

        close_eq = _mark(cash, qty, data, ts, "close")
        equity_points.append((ts, close_eq))
        alloc_rows.append({
            "time": ts, "equity": close_eq, "cash_weight": cash / close_eq if close_eq > 0 else 0.0,
            **{
                f"weight_{s}": qty[s] * float(data[s]["df"].loc[ts, "close"]) / close_eq
                if close_eq > 0 else 0.0
                for s in symbols
            },
        })

        if last_signal_i is None or i - last_signal_i >= rebalance_days:
            pending = target_weights(
                data, symbols, ts, target_vol=target_vol, top_n=top_n
            )
            pending_signal_time = ts
            last_signal_i = i

    # Conservative EOD liquidation so final capital includes exit costs.
    ts = calendar[-1]
    for s in symbols:
        if qty[s] <= 0:
            continue
        px = float(data[s]["df"].loc[ts, "close"]) * (1.0 - EXIT_SLIPPAGE_PCT)
        gross = qty[s] * px
        cash += gross * (1.0 - FEE_RATE)
        qty[s] = 0.0
    equity_points[-1] = (ts, cash)

    equity = pd.Series([v for _, v in equity_points],
                       index=[t for t, _ in equity_points], dtype=float)
    log = pd.DataFrame(logs)
    if not log.empty:
        sig = pd.to_datetime(log["signal_time"], utc=True)
        exe = pd.to_datetime(log["execution_time"], utc=True)
        if not bool((exe > sig).all()):
            raise AssertionError("look-ahead detectado no allocator")
    return {
        **ap.performance_metrics(equity),
        "equity_curve": equity,
        "rebalance_log": log,
        "allocations": pd.DataFrame(alloc_rows).set_index("time"),
    }


def simulate_buy_hold(data, symbols, start, end, weights=None):
    """Costed Spot buy-and-hold benchmark, entered at first open."""
    start_ts, end_ts = ap.as_utc(start), ap.as_utc(end)
    calendar = sorted(set.intersection(*[
        set(data[s]["df"].loc[start_ts:end_ts].index) for s in symbols
    ]))
    if not calendar:
        return pd.Series(dtype=float)
    if weights is None:
        weights = {s: 1.0 / len(symbols) for s in symbols}
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("benchmark weights precisam somar 1")
    first = calendar[0]
    qty, cash = {}, INITIAL_CAPITAL
    for s in symbols:
        px = float(data[s]["df"].loc[first, "open"]) * (1.0 + ENTRY_SLIPPAGE_PCT)
        budget = INITIAL_CAPITAL * weights[s]
        q = budget / (px * (1.0 + FEE_RATE))
        qty[s] = q
        cash -= q * px * (1.0 + FEE_RATE)
    points = [(ts, _mark(cash, qty, data, ts, "close")) for ts in calendar]
    last = calendar[-1]
    final_cash = cash
    for s, q in qty.items():
        px = float(data[s]["df"].loc[last, "close"]) * (1.0 - EXIT_SLIPPAGE_PCT)
        final_cash += q * px * (1.0 - FEE_RATE)
    points[-1] = (last, final_cash)
    return pd.Series([v for _, v in points], index=[t for t, _ in points], dtype=float)
