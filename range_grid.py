"""Conservative research simulator for a small BTC Spot range-grid sleeve.

The simulator uses hourly Binance Spot candles and a regime computed strictly
from completed daily candles.  It intentionally permits at most one grid fill
per hourly candle, preventing the optimistic multi-fill assumption common in
OHLC grid backtests.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from adaptive_portfolio import INITIAL_CAPITAL, ROUND_TRIP_COST, as_utc, performance_metrics
from config import ENTRY_SLIPPAGE_PCT, EXIT_SLIPPAGE_PCT, FEE_RATE
from indicators import compute_atr

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
               "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def download_ohlcv(symbol: str, start, end, interval: str = "1h") -> pd.DataFrame:
    """Download exact Binance Global Spot candles without proxy fallback."""
    if interval not in INTERVAL_MS:
        raise ValueError(f"intervalo não suportado: {interval}")
    current = int(as_utc(start).timestamp() * 1000)
    end_ms = int(as_utc(end).timestamp() * 1000)
    rows = []
    while current < end_ms:
        params = {"symbol": symbol.upper(), "interval": interval, "startTime": current,
                  "endTime": end_ms, "limit": 1000}
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=20)
        response.raise_for_status()
        batch = response.json()
        if not batch or isinstance(batch, dict):
            break
        rows.extend(batch)
        next_ts = int(batch[-1][0]) + INTERVAL_MS[interval]
        if next_ts <= current:
            raise RuntimeError("paginação Binance não avançou")
        current = next_ts
        if len(batch) == 1000:
            time.sleep(0.08)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
    ])
    frame.index = pd.to_datetime(frame.pop("open_time"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = frame[col].astype(float)
    return frame[["open", "high", "low", "close", "volume"]].sort_index()


def _trend_efficiency(close: pd.Series, window: int = 20) -> pd.Series:
    displacement = close.diff(window).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return displacement / path.replace(0.0, np.nan)


def daily_range_state(daily: pd.DataFrame) -> pd.DataFrame:
    """Pre-declared range state using only daily information at each close."""
    close = daily["close"].astype(float)
    sma200 = close.rolling(200, min_periods=200).mean()
    atr_pct = compute_atr(daily["high"], daily["low"], close) / close
    momentum20 = close.pct_change(20)
    efficiency20 = _trend_efficiency(close, 20)
    state = (
        (close > sma200)
        & (momentum20.abs() < 0.10)
        & (efficiency20 < 0.35)
        & (atr_pct > ROUND_TRIP_COST)
    )
    return pd.DataFrame({
        "range_on": state.fillna(False),
        "atr_pct": atr_pct,
        "momentum20": momentum20,
        "efficiency20": efficiency20,
    }, index=daily.index)


def _known_daily_state(states: pd.DataFrame, ts: pd.Timestamp):
    """Return the last state from a daily candle completed before this hour."""
    eligible = states.loc[states.index < ts.normalize()]
    return None if eligible.empty else eligible.iloc[-1]


def simulate_range_grid(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    start,
    end,
    *,
    levels_each_side: int = 6,
    min_spacing: float | None = None,
) -> dict:
    """Simulate one independently funded BTC range-grid sleeve.

    A new grid starts 50/50 BTC and USDT.  The spacing is the greater of half
    the previous daily ATR percentage and three times all-in round-trip costs.
    A grid is liquidated when the range regime ends or the close escapes its
    finite boundaries.  It is never silently re-centred while carrying a bag.
    """
    if levels_each_side < 2:
        raise ValueError("levels_each_side precisa ser >= 2")
    spacing_floor = 3.0 * ROUND_TRIP_COST if min_spacing is None else min_spacing
    if spacing_floor <= ROUND_TRIP_COST:
        raise ValueError("spacing mínimo precisa exceder o custo de ida e volta")

    start_ts, end_ts = as_utc(start), as_utc(end)
    bars = hourly.loc[start_ts:end_ts].sort_index()
    if bars.empty:
        empty = pd.Series(dtype=float)
        return {**performance_metrics(empty), "equity_curve": empty,
                "trade_log": pd.DataFrame(), "fills": 0, "active_hours": 0}
    states = daily_range_state(daily)

    cash = INITIAL_CAPITAL
    btc_qty = 0.0
    active = False
    center = np.nan
    spacing = np.nan
    current_level = 0
    order_quote = 0.0
    trades = []
    equity_points = []
    active_hours = 0

    def liquidate(ts, price, reason):
        nonlocal cash, btc_qty, active
        if btc_qty > 0:
            px = price * (1.0 - EXIT_SLIPPAGE_PCT)
            gross = btc_qty * px
            fee = gross * FEE_RATE
            cash += gross - fee
            trades.append({"time": ts, "side": "SELL_ALL", "price": px,
                           "quantity": btc_qty, "fee": fee, "reason": reason})
            btc_qty = 0.0
        active = False

    for ts, row in bars.iterrows():
        known = _known_daily_state(states, ts)
        range_on = bool(known is not None and known["range_on"])
        op, hi, lo, close = map(float, (row["open"], row["high"], row["low"], row["close"]))

        if active and not range_on:
            liquidate(ts, op, "REGIME_OFF")

        if not active and range_on:
            atr_pct = float(known["atr_pct"])
            spacing = max(spacing_floor, min(0.03, 0.5 * atr_pct))
            center = op
            current_level = 0
            order_quote = INITIAL_CAPITAL / (2 * levels_each_side + 2)
            invest = cash * 0.5
            px = op * (1.0 + ENTRY_SLIPPAGE_PCT)
            qty = invest / (px * (1.0 + FEE_RATE))
            debit = qty * px * (1.0 + FEE_RATE)
            cash -= debit
            btc_qty += qty
            trades.append({"time": ts, "side": "BUY_INIT", "price": px,
                           "quantity": qty, "fee": qty * px * FEE_RATE, "reason": "RANGE_ON"})
            active = True

        if active:
            active_hours += 1
            lower_boundary = center * (1.0 - levels_each_side * spacing)
            upper_boundary = center * (1.0 + levels_each_side * spacing)
            if close < lower_boundary or close > upper_boundary:
                liquidate(ts, close, "RANGE_BREAK")
            elif close < op and current_level > -levels_each_side:
                level = current_level - 1
                price = center * (1.0 + level * spacing)
                if lo <= price:
                    fill_px = price * (1.0 + ENTRY_SLIPPAGE_PCT)
                    qty = min(order_quote, cash / (1.0 + FEE_RATE)) / fill_px
                    debit = qty * fill_px * (1.0 + FEE_RATE)
                    if qty > 0 and debit <= cash:
                        cash -= debit
                        btc_qty += qty
                        current_level = level
                        trades.append({"time": ts, "side": "BUY", "price": fill_px,
                                       "quantity": qty, "fee": qty * fill_px * FEE_RATE,
                                       "reason": "GRID"})
            elif close > op and current_level < levels_each_side:
                level = current_level + 1
                price = center * (1.0 + level * spacing)
                if hi >= price and btc_qty > 0:
                    fill_px = price * (1.0 - EXIT_SLIPPAGE_PCT)
                    qty = min(order_quote / fill_px, btc_qty)
                    gross = qty * fill_px
                    fee = gross * FEE_RATE
                    cash += gross - fee
                    btc_qty -= qty
                    current_level = level
                    trades.append({"time": ts, "side": "SELL", "price": fill_px,
                                   "quantity": qty, "fee": fee, "reason": "GRID"})

        equity_points.append((ts, cash + btc_qty * close))

    if btc_qty > 0:
        ts = bars.index[-1]
        liquidate(ts, float(bars["close"].iloc[-1]), "EOD")
        equity_points[-1] = (ts, cash)
    hourly_equity = pd.Series([v for _, v in equity_points], index=[t for t, _ in equity_points], dtype=float)
    daily_equity = hourly_equity.resample("1D").last().dropna()
    return {**performance_metrics(daily_equity), "equity_curve": daily_equity,
            "trade_log": pd.DataFrame(trades), "fills": len(trades), "active_hours": active_hours}
