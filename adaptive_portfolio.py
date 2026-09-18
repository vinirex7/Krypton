"""Research-only adaptive portfolio components for Krypton.

This module deliberately does not alter the live bot.  It provides auditable,
lagged-execution simulations for three pre-declared ideas:

* a BTC/SMA200 trend core;
* a partial take-profit runner for the existing tactical strategy;
* a cost-aware gate for new tactical entries.

All sleeves start with the same notional equity and can be combined with fixed
capital weights.  That makes the attribution explicit and avoids hidden
leverage or implicit daily rebalancing.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable

import numpy as np
import pandas as pd

import walk_forward as wf
from config import (
    CIRCUIT_BREAKER_PCT,
    ENTRY_SLIPPAGE_PCT,
    EXIT_SLIPPAGE_PCT,
    FEE_RATE,
    MAX_DRAWDOWN_PCT,
    MAX_SIMULTANEOUS_POS,
    RISK_PER_TRADE,
    STOP_LOSS_ATR_MULT,
)

INITIAL_CAPITAL = wf.INITIAL_CAPITAL
MIN_NOTIONAL = wf.MIN_NOTIONAL
ROUND_TRIP_COST = 2.0 * FEE_RATE + ENTRY_SLIPPAGE_PCT + EXIT_SLIPPAGE_PCT


def as_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def performance_metrics(equity: pd.Series) -> dict:
    """Return consistent daily portfolio metrics for a marked-to-market curve."""
    eq = equity.dropna().astype(float)
    if eq.empty:
        return {
            "return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "final_capital": INITIAL_CAPITAL,
        }
    # Every research sleeve is explicitly funded with INITIAL_CAPITAL.  Using
    # the first marked point would silently discard first-day fees/returns.
    initial = INITIAL_CAPITAL
    total_return = float(eq.iloc[-1] / initial - 1.0)
    elapsed_days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86_400.0, 1.0)
    years = elapsed_days / 365.25
    cagr = float((eq.iloc[-1] / initial) ** (1.0 / years) - 1.0) if initial > 0 else 0.0
    anchor_time = eq.index[0] - pd.Timedelta(nanoseconds=1)
    anchored = pd.concat([pd.Series([initial], index=[anchor_time]), eq])
    returns = anchored.pct_change().dropna()
    std = float(returns.std()) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / std * sqrt(365.0)) if std > 0 else 0.0
    drawdowns = (anchored - anchored.cummax()) / anchored.cummax()
    max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("inf") if cagr > 0 else 0.0
    return {
        "return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "final_capital": float(eq.iloc[-1]),
    }


def slice_and_rebase(equity: pd.Series, start, end) -> pd.Series:
    """Slice a curve and add a start anchor so period returns are comparable."""
    start_ts, end_ts = as_utc(start), as_utc(end)
    eq = equity.sort_index()
    before = eq.loc[eq.index < start_ts]
    period = eq.loc[start_ts:end_ts]
    if period.empty:
        return pd.Series(dtype=float)
    anchor = float(before.iloc[-1]) if not before.empty else float(period.iloc[0])
    rebased = period / anchor * INITIAL_CAPITAL
    if rebased.index[0] > start_ts:
        rebased.loc[start_ts] = INITIAL_CAPITAL
        rebased = rebased.sort_index()
    return rebased


def combine_sleeves(sleeves: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Combine independently funded sleeves without periodic rebalancing."""
    if set(sleeves) != set(weights):
        raise ValueError("sleeves e weights precisam ter as mesmas chaves")
    if any(w < 0 for w in weights.values()) or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("pesos precisam ser não-negativos e somar 1")
    index = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in sleeves.values()])))
    if index.empty:
        return pd.Series(dtype=float)
    combined = pd.Series(0.0, index=index)
    for name, curve in sleeves.items():
        aligned = curve.sort_index().reindex(index).ffill().bfill()
        if aligned.empty or aligned.iloc[0] <= 0:
            raise ValueError(f"curva inválida: {name}")
        combined = combined.add(aligned * weights[name], fill_value=0.0)
    return combined


def cost_gate(entry_price: float, atr: float, tp_mult: float = 3.0,
              cost_multiple: float = 3.0) -> bool:
    """Require the gross TP excursion to exceed conservative all-in costs."""
    if entry_price <= 0 or atr <= 0 or tp_mult <= 0 or cost_multiple <= 0:
        return False
    gross_edge = tp_mult * atr / entry_price
    return gross_edge > cost_multiple * ROUND_TRIP_COST


def persistent_state_permission(
    data,
    symbols,
    *,
    min_signal_age: int = 90,
    short_window: int = 30,
    long_window: int = 90,
):
    """Block stale long states only when recent cross-asset continuity collapses.

    The rule follows the previously identified causal hypothesis: a long signal
    that has persisted for at least 90 completed daily candles is not treated as
    a fresh entry when no configured asset has positive 30-day momentum while
    at least two thirds still retain positive 90-day momentum.  Every input is
    known at the signal close; execution remains at the next open.
    """
    if min_signal_age < 2 or short_window < 1 or long_window <= short_window:
        raise ValueError("janelas inválidas para continuity gate")
    ages = {}
    short_momentum = {}
    long_momentum = {}
    for symbol in symbols:
        signal = data[symbol]["signals"].sort_index().astype(int)
        groups = signal.ne(signal.shift()).cumsum()
        age = signal.groupby(groups).cumcount().add(1).where(signal.eq(1), 0)
        ages[symbol] = age
        close = data[symbol]["df"]["close"].astype(float).sort_index()
        short_momentum[symbol] = close.pct_change(short_window)
        long_momentum[symbol] = close.pct_change(long_window)

    def allowed(symbol: str, ts: pd.Timestamp) -> bool:
        age = ages.get(symbol)
        if age is None or ts not in age.index or int(age.loc[ts]) < min_signal_age:
            return True
        short_values = [series.loc[ts] for series in short_momentum.values()
                        if ts in series.index and pd.notna(series.loc[ts])]
        long_values = [series.loc[ts] for series in long_momentum.values()
                       if ts in series.index and pd.notna(series.loc[ts])]
        if len(short_values) != len(symbols) or len(long_values) != len(symbols):
            return True
        short_breadth = sum(float(value) > 0 for value in short_values) / len(symbols)
        long_breadth = sum(float(value) > 0 for value in long_values) / len(symbols)
        stale_deterioration = short_breadth == 0.0 and long_breadth >= (2.0 / 3.0)
        return not stale_deterioration

    return allowed


def simulate_btc_core(data, start, end) -> dict:
    """100%-of-sleeve BTC trend core, switched at the next daily open.

    In the final portfolio this sleeve receives a fixed fractional allocation
    (40% in the pre-declared experiment), so the total portfolio never uses
    leverage.
    """
    start_ts, end_ts = as_utc(start), as_utc(end)
    btc = data["BTCUSDT"]
    df = btc["df"].loc[start_ts:end_ts]
    if df.empty:
        return {"equity_curve": pd.Series(dtype=float), "trade_log": pd.DataFrame(), "trades": 0}

    cash = INITIAL_CAPITAL
    quantity = 0.0
    entry_price = np.nan
    pending_target: bool | None = None
    trades = []
    equity_points = []

    for ts, row in df.iterrows():
        open_px = float(row["open"])
        if pending_target is True and quantity == 0:
            px = open_px * (1.0 + ENTRY_SLIPPAGE_PCT)
            quantity = cash / (px * (1.0 + FEE_RATE))
            debit = quantity * px * (1.0 + FEE_RATE)
            cash -= debit
            entry_price = px
        elif pending_target is False and quantity > 0:
            px = open_px * (1.0 - EXIT_SLIPPAGE_PCT)
            proceeds = quantity * px
            exit_fee = proceeds * FEE_RATE
            pnl = quantity * (px - entry_price) - quantity * entry_price * FEE_RATE - exit_fee
            cash += proceeds - exit_fee
            trades.append({"entry_price": entry_price, "exit_price": px, "exit_time": ts, "pnl": pnl})
            quantity = 0.0
            entry_price = np.nan

        equity_points.append((ts, cash + quantity * float(row["close"])))
        sma = btc["sma200"].loc[ts] if ts in btc["sma200"].index else np.nan
        pending_target = bool(pd.notna(sma) and float(row["close"]) > float(sma))

    if quantity > 0:
        ts = df.index[-1]
        px = float(df["close"].iloc[-1]) * (1.0 - EXIT_SLIPPAGE_PCT)
        proceeds = quantity * px
        exit_fee = proceeds * FEE_RATE
        pnl = quantity * (px - entry_price) - quantity * entry_price * FEE_RATE - exit_fee
        cash += proceeds - exit_fee
        trades.append({"entry_price": entry_price, "exit_price": px, "exit_time": ts, "pnl": pnl})
        equity_points[-1] = (ts, cash)

    equity = pd.Series([v for _, v in equity_points], index=[t for t, _ in equity_points], dtype=float)
    return {**performance_metrics(equity), "equity_curve": equity,
            "trade_log": pd.DataFrame(trades), "trades": len(trades)}


@dataclass
class RunnerPosition:
    symbol: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp
    equity_at_entry: float
    tp_taken: bool = False


def _mark(cash, positions, data, ts, field="close") -> float:
    equity = cash
    for symbol, pos in positions.items():
        df = data[symbol]["df"]
        if ts in df.index:
            px = float(df.loc[ts, field])
        else:
            eligible = df.loc[:ts]
            px = pos.entry_price if eligible.empty else float(eligible["close"].iloc[-1])
        equity += pos.quantity * px
    return equity


def _sell_leg(cash, pos: RunnerPosition, qty: float, price: float, ts, reason: str):
    qty = min(qty, pos.quantity)
    gross = qty * price
    exit_fee = gross * FEE_RATE
    entry_fee_share = qty * pos.entry_price * FEE_RATE
    pnl = qty * (price - pos.entry_price) - entry_fee_share - exit_fee
    cash += gross - exit_fee
    trade = {
        "symbol": pos.symbol,
        "entry_time": pos.entry_time,
        "exit_time": ts,
        "entry_price": pos.entry_price,
        "exit_price": price,
        "quantity": qty,
        "pnl": pnl,
        "portfolio_return": pnl / pos.equity_at_entry if pos.equity_at_entry > 0 else 0.0,
        "reason": reason,
    }
    pos.quantity -= qty
    return cash, trade


def simulate_tactical(
    data,
    symbols,
    start,
    end,
    *,
    runner: bool = False,
    cost_aware: bool = False,
    entry_permission: Callable[[str, pd.Timestamp], bool] | None = None,
    risk_per_trade: float = RISK_PER_TRADE,
    tp_mult: float = 3.0,
) -> dict:
    """Simulate the frozen tactical engine with optional partial-TP runner.

    The runner sells half at the existing TP and leaves the other half protected
    by the original stop until the normal signal exit.  If SL and TP are both
    touched before the partial fill, the full position exits at SL.
    """
    start_ts, end_ts = as_utc(start), as_utc(end)
    weights = wf.normalized_weights(symbols)
    calendar = sorted(set().union(*[set(data[s]["df"].loc[start_ts:end_ts].index) for s in symbols]))
    if len(calendar) < 20:
        empty = pd.Series(dtype=float)
        return {**performance_metrics(empty), "equity_curve": empty, "trade_log": pd.DataFrame(), "trades": 0}

    cash = INITIAL_CAPITAL
    positions: dict[str, RunnerPosition] = {}
    pending: dict[str, float] = {}
    trades = []
    equity_points = []
    peak = INITIAL_CAPITAL
    daily_start = INITIAL_CAPITAL
    daily_date = None
    halted = False

    for ts in calendar:
        pre_eq = _mark(cash, positions, data, ts, "open")
        if daily_date != ts.date():
            daily_date, daily_start = ts.date(), pre_eq

        # Normal signal exits known at the previous close.
        for symbol in list(positions):
            df, sig = data[symbol]["df"], data[symbol]["signals"]
            if ts not in df.index:
                continue
            loc = df.index.get_loc(ts)
            if isinstance(loc, slice) or loc == 0:
                continue
            prev = df.index[loc - 1]
            if prev >= start_ts and int(sig.loc[prev]) != 1:
                pos = positions.pop(symbol)
                px = float(df.loc[ts, "open"]) * (1.0 - EXIT_SLIPPAGE_PCT)
                cash, trade = _sell_leg(cash, pos, pos.quantity, px, ts, "SigRunner" if pos.tp_taken else "Sig")
                trades.append(trade)

        # Execute entries scheduled at the prior close.
        for symbol in list(pending):
            if halted or symbol in positions or len(positions) >= MAX_SIMULTANEOUS_POS:
                pending.pop(symbol, None)
                continue
            df = data[symbol]["df"]
            if ts not in df.index:
                continue
            eq = _mark(cash, positions, data, ts, "open")
            daily_loss = (daily_start - eq) / daily_start if daily_start > 0 else 0.0
            if daily_loss >= CIRCUIT_BREAKER_PCT:
                pending.clear()
                break
            atr = pending.pop(symbol)
            entry = float(df.loc[ts, "open"]) * (1.0 + ENTRY_SLIPPAGE_PCT)
            sl_distance = atr * STOP_LOSS_ATR_MULT
            raw_qty = eq * risk_per_trade / sl_distance if sl_distance > 0 else 0.0
            cap = min(eq * weights[symbol], cash / (1.0 + FEE_RATE))
            qty = min(raw_qty, cap / entry if entry > 0 else 0.0)
            notional = qty * entry
            if qty <= 0 or notional < MIN_NOTIONAL or notional * (1.0 + FEE_RATE) > cash:
                continue
            cash -= notional * (1.0 + FEE_RATE)
            positions[symbol] = RunnerPosition(
                symbol, entry, qty, entry - sl_distance, entry + atr * tp_mult, ts, eq
            )

        # Conservative intraday ordering.
        for symbol in list(positions):
            df = data[symbol]["df"]
            if ts not in df.index:
                continue
            row, pos = df.loc[ts], positions[symbol]
            op = float(row["open"])
            sl_hit = op <= pos.stop_loss or float(row["low"]) <= pos.stop_loss
            tp_hit = (not pos.tp_taken) and (op >= pos.take_profit or float(row["high"]) >= pos.take_profit)
            if sl_hit:
                px = (op if op <= pos.stop_loss else pos.stop_loss) * (1.0 - EXIT_SLIPPAGE_PCT)
                cash, trade = _sell_leg(cash, pos, pos.quantity, px, ts, "SL_GAP" if op <= pos.stop_loss else "SL")
                trades.append(trade)
                positions.pop(symbol)
            elif tp_hit:
                if runner:
                    half = pos.quantity * 0.5
                    cash, trade = _sell_leg(cash, pos, half, pos.take_profit, ts, "TP_PARTIAL")
                    trades.append(trade)
                    pos.tp_taken = True
                    # Do not let a profitable runner fall all the way back to
                    # its original stop.  Include exit slippage and both fees
                    # so the remaining leg's stop is approximately net-flat.
                    net_break_even = (
                        pos.entry_price * (1.0 + FEE_RATE)
                        / ((1.0 - EXIT_SLIPPAGE_PCT) * (1.0 - FEE_RATE))
                    )
                    pos.stop_loss = max(pos.stop_loss, net_break_even)
                else:
                    cash, trade = _sell_leg(cash, pos, pos.quantity, pos.take_profit, ts, "TP")
                    trades.append(trade)
                    positions.pop(symbol)

        eq = _mark(cash, positions, data, ts)
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0.0
        daily_loss = (daily_start - eq) / daily_start if daily_start > 0 else 0.0
        if dd >= MAX_DRAWDOWN_PCT:
            halted = True
            pending.clear()
        if daily_loss >= CIRCUIT_BREAKER_PCT:
            pending.clear()

        risk_on = (not halted) and wf._risk_on(data, ts)
        if risk_on and daily_loss < CIRCUIT_BREAKER_PCT:
            for symbol in symbols:
                if symbol in positions or symbol in pending:
                    continue
                if len(positions) + len(pending) >= MAX_SIMULTANEOUS_POS:
                    break
                df, sig, atr_s = data[symbol]["df"], data[symbol]["signals"], data[symbol]["atr"]
                if ts not in df.index:
                    continue
                loc = df.index.get_loc(ts)
                if isinstance(loc, slice) or loc + 1 >= len(df) or df.index[loc + 1] > end_ts:
                    continue
                atr = float(atr_s.loc[ts]) if pd.notna(atr_s.loc[ts]) else np.nan
                allowed = entry_permission(symbol, ts) if entry_permission is not None else True
                cost_ok = (not cost_aware) or cost_gate(float(df.loc[ts, "close"]), atr, tp_mult)
                if int(sig.loc[ts]) == 1 and np.isfinite(atr) and atr > 0 and allowed and cost_ok:
                    pending[symbol] = atr

        equity_points.append((ts, eq))

    for symbol in list(positions):
        df = data[symbol]["df"].loc[:end_ts]
        if df.empty:
            continue
        ts = df.index[-1]
        pos = positions.pop(symbol)
        px = float(df["close"].iloc[-1]) * (1.0 - EXIT_SLIPPAGE_PCT)
        cash, trade = _sell_leg(cash, pos, pos.quantity, px, ts, "EOD")
        trades.append(trade)
    if equity_points:
        equity_points[-1] = (equity_points[-1][0], cash)
    equity = pd.Series([v for _, v in equity_points], index=[t for t, _ in equity_points], dtype=float)
    return {**performance_metrics(equity), "equity_curve": equity,
            "trade_log": pd.DataFrame(trades), "trades": len(trades), "halted": halted}
