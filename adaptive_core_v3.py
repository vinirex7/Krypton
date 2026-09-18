"""Research-only BTC trend core v3 with ATR-scaled entry hysteresis.

The core enters only when the previous completed daily close is above the
SMA200 by a volatility-scaled margin and the SMA200 has a positive 20-day
slope. It exits when the structural trend condition itself is lost. Orders are
executed at the next daily open and include the same fees/slippage as Krypton.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import adaptive_portfolio as ap
from config import ENTRY_SLIPPAGE_PCT, EXIT_SLIPPAGE_PCT, FEE_RATE

INITIAL_CAPITAL = ap.INITIAL_CAPITAL


def hysteresis_target_series(df: pd.DataFrame, sma200: pd.Series, atr: pd.Series,
                             entry_atr_mult: float = 0.5,
                             slope_lookback: int = 20) -> pd.Series:
    if entry_atr_mult < 0:
        raise ValueError("entry_atr_mult deve ser >= 0")
    if slope_lookback < 1:
        raise ValueError("slope_lookback deve ser >= 1")
    close = df["close"].astype(float).sort_index()
    sma = sma200.astype(float).sort_index().reindex(close.index)
    av = atr.astype(float).sort_index().reindex(close.index)
    slope_ok = sma.gt(sma.shift(slope_lookback))
    base_trend = close.gt(sma) & slope_ok
    entry_ok = close.gt(sma + entry_atr_mult * av) & slope_ok
    state = False
    out = []
    for ts in close.index:
        if state:
            state = bool(base_trend.loc[ts]) if pd.notna(base_trend.loc[ts]) else False
        else:
            state = bool(entry_ok.loc[ts]) if pd.notna(entry_ok.loc[ts]) else False
        out.append(state)
    return pd.Series(out, index=close.index, dtype=bool)


def simulate_btc_trend_core(data, start, end, entry_atr_mult: float = 0.5,
                            slope_lookback: int = 20) -> dict:
    start_ts, end_ts = ap.as_utc(start), ap.as_utc(end)
    btc = data["BTCUSDT"]
    df = btc["df"].loc[start_ts:end_ts].copy()
    if df.empty:
        empty = pd.Series(dtype=float)
        return {**ap.performance_metrics(empty), "equity_curve": empty,
                "trade_log": pd.DataFrame(), "trades": 0}

    targets = hysteresis_target_series(
        btc["df"], btc["sma200"], btc["atr"],
        entry_atr_mult=entry_atr_mult, slope_lookback=slope_lookback,
    )
    cash = INITIAL_CAPITAL
    quantity = 0.0
    entry_price = np.nan
    entry_time = signal_time = None
    pending_target = None
    pending_signal_time = None
    trades = []
    equity_points = []

    for ts, row in df.iterrows():
        open_px = float(row["open"])
        if pending_target is True and quantity == 0:
            px = open_px * (1.0 + ENTRY_SLIPPAGE_PCT)
            quantity = cash / (px * (1.0 + FEE_RATE))
            cash -= quantity * px * (1.0 + FEE_RATE)
            entry_price, entry_time, signal_time = px, ts, pending_signal_time
        elif pending_target is False and quantity > 0:
            px = open_px * (1.0 - EXIT_SLIPPAGE_PCT)
            proceeds = quantity * px
            exit_fee = proceeds * FEE_RATE
            entry_fee = quantity * entry_price * FEE_RATE
            pnl = quantity * (px - entry_price) - entry_fee - exit_fee
            cash += proceeds - exit_fee
            trades.append({"signal_time": signal_time, "entry_time": entry_time,
                           "exit_time": ts, "entry_price": entry_price,
                           "exit_price": px, "pnl": pnl, "reason": "TrendOff"})
            quantity = 0.0
            entry_price = np.nan
            entry_time = signal_time = None

        equity_points.append((ts, cash + quantity * float(row["close"])))
        pending_target = bool(targets.loc[ts]) if ts in targets.index else False
        pending_signal_time = ts

    if quantity > 0:
        ts = df.index[-1]
        px = float(df["close"].iloc[-1]) * (1.0 - EXIT_SLIPPAGE_PCT)
        proceeds = quantity * px
        exit_fee = proceeds * FEE_RATE
        entry_fee = quantity * entry_price * FEE_RATE
        pnl = quantity * (px - entry_price) - entry_fee - exit_fee
        cash += proceeds - exit_fee
        trades.append({"signal_time": signal_time, "entry_time": entry_time,
                       "exit_time": ts, "entry_price": entry_price,
                       "exit_price": px, "pnl": pnl, "reason": "EOD"})
        equity_points[-1] = (ts, cash)

    equity = pd.Series([v for _, v in equity_points],
                       index=[t for t, _ in equity_points], dtype=float)
    log = pd.DataFrame(trades)
    if not log.empty and not bool((pd.to_datetime(log["entry_time"], utc=True) >
                                   pd.to_datetime(log["signal_time"], utc=True)).all()):
        raise AssertionError("look-ahead detectado no core v3")
    return {**ap.performance_metrics(equity), "equity_curve": equity,
            "trade_log": log, "trades": len(log)}
