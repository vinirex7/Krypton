"""Research-only complementary Spot mean-reversion sleeve for Krypton.

Purpose: capture pullback rebounds only when the frozen cross-asset trend alpha is
inactive. Signals use completed daily closes and execute at the next daily open.
No shorting, no leverage, and no live-order integration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import adaptive_portfolio as ap
import cross_asset_hybrid_v2 as hv2
from config import ENTRY_SLIPPAGE_PCT, EXIT_SLIPPAGE_PCT, FEE_RATE

INITIAL_CAPITAL = ap.INITIAL_CAPITAL


def _rsi(series: pd.Series, window: int = 2) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).rolling(window).mean()
    dn = (-d.clip(upper=0)).rolling(window).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def trend_alpha_active(data, symbols, ts) -> bool:
    """Frozen Challenger-v1 breadth condition at a completed close."""
    w = hv2.breadth_target_weights(
        data, symbols, ts, target_vol=0.15, top_n=2, min_selected=2
    )
    return any(float(v) > 0 for v in w.values())


def entry_candidates(data, symbols, ts):
    """Causal pullback candidates only when the trend sleeve would be cash.

    Frozen hypothesis before results:
    - asset remains structurally healthy: close > own SMA200;
    - sharp but not crash-like pullback: 3d return <= -5%;
    - short-term oversold: RSI(2) <= 10;
    - price below 5d mean, which is the recovery target.
    Rank: most negative 3d return first.
    """
    if trend_alpha_active(data, symbols, ts):
        return []
    rows = []
    for s in symbols:
        df = data[s]["df"].sort_index()
        if ts not in df.index:
            continue
        close = df["close"].astype(float).loc[:ts]
        if len(close) < 205:
            continue
        sma200 = data[s]["sma200"].reindex(close.index).loc[ts]
        if pd.isna(sma200) or float(close.loc[ts]) <= float(sma200):
            continue
        r3 = float(close.loc[ts] / close.shift(3).loc[ts] - 1.0)
        rsi2 = float(_rsi(close, 2).loc[ts])
        sma5 = float(close.rolling(5).mean().loc[ts])
        if not np.isfinite(r3) or not np.isfinite(rsi2) or not np.isfinite(sma5):
            continue
        if r3 <= -0.05 and rsi2 <= 10.0 and float(close.loc[ts]) < sma5:
            rows.append((s, r3, sma5))
    rows.sort(key=lambda x: x[1])
    return rows


def simulate_mean_reversion(data, symbols, start, end, *, max_positions=1,
                            sleeve_target=0.20, max_hold_days=5,
                            stop_atr=2.0):
    """Fully funded sleeve with capped exposure and explicit costs.

    Entry at next open. Exit at next open after a completed close where close >=
    SMA5, or after max_hold_days bars. Intraday stop uses ATR known at the signal
    close and current low; stop gets conservative priority.
    """
    start_ts, end_ts = ap.as_utc(start), ap.as_utc(end)
    calendar = sorted(set.intersection(*[
        set(data[s]["df"].loc[start_ts:end_ts].index) for s in symbols
    ]))
    if not calendar:
        empty = pd.Series(dtype=float)
        return {**ap.performance_metrics(empty), "equity_curve": empty,
                "trade_log": pd.DataFrame()}

    cash = INITIAL_CAPITAL
    pos = {}
    pending_entries = []
    pending_exits = set()
    points, logs = [], []

    for i, ts in enumerate(calendar):
        opens = {s: float(data[s]["df"].loc[ts, "open"]) for s in symbols}

        for s in list(pending_exits):
            if s not in pos:
                continue
            p = pos.pop(s)
            px = opens[s] * (1.0 - EXIT_SLIPPAGE_PCT)
            gross = p["qty"] * px
            fee = gross * FEE_RATE
            cash += gross - fee
            logs.append({"symbol": s, "entry_time": p["entry_time"],
                         "exit_time": ts, "reason": p["exit_reason"],
                         "pnl": (gross - fee) - p["cost"]})
        pending_exits.clear()

        if pending_entries and len(pos) < max_positions:
            sleeve_eq = cash + sum(p["qty"] * opens[s] for s, p in pos.items())
            for s, signal_ts in pending_entries:
                if len(pos) >= max_positions or s in pos or cash <= 0:
                    break
                budget = min(cash / (1.0 + FEE_RATE), sleeve_eq * sleeve_target)
                if budget <= 0:
                    continue
                px = opens[s] * (1.0 + ENTRY_SLIPPAGE_PCT)
                qty = budget / px
                debit = qty * px * (1.0 + FEE_RATE)
                cash -= debit
                atr = float(data[s]["atr"].loc[signal_ts])
                pos[s] = {"qty": qty, "entry_time": ts, "entry_i": i,
                          "cost": debit, "stop": px - stop_atr * atr}
        pending_entries = []

        for s in list(pos):
            low = float(data[s]["df"].loc[ts, "low"])
            if low <= pos[s]["stop"]:
                p = pos.pop(s)
                px = p["stop"] * (1.0 - EXIT_SLIPPAGE_PCT)
                gross = p["qty"] * px
                fee = gross * FEE_RATE
                cash += gross - fee
                logs.append({"symbol": s, "entry_time": p["entry_time"],
                             "exit_time": ts, "reason": "SL",
                             "pnl": (gross - fee) - p["cost"]})

        eq = cash + sum(
            p["qty"] * float(data[s]["df"].loc[ts, "close"])
            for s, p in pos.items()
        )
        points.append((ts, eq))

        for s, p in pos.items():
            close = data[s]["df"]["close"].astype(float).loc[:ts]
            sma5 = float(close.rolling(5).mean().loc[ts])
            held = i - int(p["entry_i"]) + 1
            if float(close.loc[ts]) >= sma5:
                p["exit_reason"] = "SMA5"
                pending_exits.add(s)
            elif held >= max_hold_days:
                p["exit_reason"] = "TIME"
                pending_exits.add(s)

        if len(pos) < max_positions:
            cand = entry_candidates(data, symbols, ts)
            pending_entries = [(s, ts) for s, _, _ in cand[:max_positions-len(pos)]]

    ts = calendar[-1]
    for s, p in list(pos.items()):
        px = float(data[s]["df"].loc[ts, "close"]) * (1.0 - EXIT_SLIPPAGE_PCT)
        gross = p["qty"] * px
        fee = gross * FEE_RATE
        cash += gross - fee
        logs.append({"symbol": s, "entry_time": p["entry_time"],
                     "exit_time": ts, "reason": "EOD",
                     "pnl": (gross - fee) - p["cost"]})
    if points:
        points[-1] = (calendar[-1], cash)
    equity = pd.Series([v for _, v in points], index=[t for t, _ in points], dtype=float)
    return {**ap.performance_metrics(equity), "equity_curve": equity,
            "trade_log": pd.DataFrame(logs)}
