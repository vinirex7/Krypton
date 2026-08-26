"""Deep validation for Krypton no_eth_regime candidate.

Frozen architecture:
- SOLUSDT, BTCUSDT, BNBUSDT
- 1% risk/trade
- LONG-only spot
- BTC close > SMA regime filter
- TP selected only from data strictly before each holdout

This script adds:
1. Exact trade log export (entry/exit timestamps, symbol, prices, qty, fees, PnL, return, reason).
2. Portfolio block bootstrap using realized OOS daily equity returns.
3. Multiple frozen pseudo-holdouts. Each holdout chooses TP only from prior data.
4. SMA sensitivity 180/200/220 as a robustness diagnostic, never as an OOS optimizer.
5. Data-integrity audit and a per-symbol signal funnel with explicit block reasons.
6. Cost stress, TP stability, attribution and a chronologically stitched OOS curve.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import walk_forward as wf
from config import (
    CIRCUIT_BREAKER_PCT, FEE_RATE, MAX_DRAWDOWN_PCT, MAX_SIMULTANEOUS_POS,
    RISK_PER_TRADE, STOP_LOSS_ATR_MULT, EXIT_SLIPPAGE_PCT,
)

SYMBOLS = ["SOLUSDT", "BTCUSDT", "BNBUSDT"]
RISK = 0.01
ENTRY_SLIPPAGE_PCT = wf.ENTRY_SLIPPAGE_PCT
MIN_NOTIONAL = wf.MIN_NOTIONAL
SMA_WINDOWS = [180, 200, 220]
DEFAULT_HOLDOUTS = ["2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01"]


@dataclass
class Position:
    symbol: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp
    entry_fee: float
    equity_at_entry: float


def _set_sma(data, window: int):
    btc = data["BTCUSDT"]
    btc["regime_sma"] = btc["df"]["close"].rolling(window, min_periods=window).mean()


def _risk_on(data, ts) -> bool:
    btc = data["BTCUSDT"]
    eligible = btc["df"].loc[:ts]
    if eligible.empty:
        return False
    t = eligible.index[-1]
    sma = btc["regime_sma"].loc[t]
    return pd.notna(sma) and float(btc["df"].loc[t, "close"]) > float(sma)


def _mtm(cash, positions, data, ts, price_field="close"):
    equity = cash
    for symbol, pos in positions.items():
        df = data[symbol]["df"]
        eligible = df.loc[:ts]
        if ts in df.index:
            px = float(df.loc[ts, price_field])
        else:
            px = pos.entry_price if eligible.empty else float(eligible["close"].iloc[-1])
        equity += pos.quantity * px
    return equity


def _close_trade(cash, pos, exit_price, exit_time, reason):
    proceeds = pos.quantity * exit_price
    exit_fee = proceeds * FEE_RATE
    cash += proceeds - exit_fee
    gross_pnl = pos.quantity * (exit_price - pos.entry_price)
    pnl = gross_pnl - pos.entry_fee - exit_fee
    invested = pos.quantity * pos.entry_price + pos.entry_fee
    trade_return = pnl / invested if invested > 0 else 0.0
    portfolio_return = pnl / pos.equity_at_entry if pos.equity_at_entry > 0 else 0.0
    trade = {
        "symbol": pos.symbol,
        "entry_time": pos.entry_time,
        "exit_time": exit_time,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "quantity": pos.quantity,
        "entry_fee": pos.entry_fee,
        "exit_fee": exit_fee,
        "gross_pnl": gross_pnl,
        "pnl": pnl,
        "trade_return": trade_return,
        "portfolio_return": portfolio_return,
        "reason": reason,
        "holding_days": max((exit_time - pos.entry_time).days, 0),
    }
    return cash, trade


def simulate(
    data,
    start,
    end,
    tp_mult,
    sma_window=200,
    capture_trades=True,
    capture_diagnostics=False,
    extra_slippage_bps=0.0,
):
    _set_sma(data, sma_window)
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    if extra_slippage_bps < 0:
        raise ValueError("extra_slippage_bps não pode ser negativo.")
    extra_slippage = extra_slippage_bps / 10_000.0
    entry_slippage = ENTRY_SLIPPAGE_PCT + extra_slippage
    exit_slippage = EXIT_SLIPPAGE_PCT + extra_slippage

    weights = wf.normalized_weights(SYMBOLS)
    calendar = sorted(set().union(*[set(data[s]["df"].loc[start:end].index) for s in SYMBOLS]))
    cash = wf.INITIAL_CAPITAL
    positions = {}
    pending = {}
    pending_diag = {}
    trades = []
    diagnostic_rows = []
    equity_points = []
    peak = cash
    day0 = None
    day_start_eq = cash
    halted = False

    for ts in calendar:
        pre_eq = _mtm(cash, positions, data, ts, price_field="open")
        if day0 != ts.date():
            day0 = ts.date()
            day_start_eq = pre_eq

        # OPEN 1/2: signal exits known since the previous close.
        for symbol in list(positions):
            df = data[symbol]["df"]
            sig = data[symbol]["signals"]
            if ts not in df.index:
                continue
            loc = df.index.get_loc(ts)
            if isinstance(loc, slice) or loc == 0:
                continue
            prev_ts = df.index[loc-1]
            if prev_ts >= start and int(sig.loc[prev_ts]) != 1:
                pos = positions[symbol]
                px = float(df.loc[ts, "open"]) * (1.0 - exit_slippage)
                cash, trade = _close_trade(cash, pos, px, ts, "Sig")
                trades.append(trade)
                positions.pop(symbol)

        # OPEN 2/2: previous close -> current open entries.
        for symbol in list(pending):
            if halted or symbol in positions or len(positions) >= MAX_SIMULTANEOUS_POS:
                pending.pop(symbol, None)
                diag_idx = pending_diag.pop(symbol, None)
                if diag_idx is not None:
                    reason = ("rejected_halt" if halted else
                              "rejected_position_exists" if symbol in positions else
                              "rejected_capacity")
                    diagnostic_rows[diag_idx]["entry_status"] = reason
                continue
            df = data[symbol]["df"]
            if ts not in df.index:
                continue
            eq = _mtm(cash, positions, data, ts, price_field="open")
            daily_loss = (day_start_eq - eq) / day_start_eq if day_start_eq > 0 else 0
            if daily_loss >= CIRCUIT_BREAKER_PCT:
                for pending_symbol, diag_idx in list(pending_diag.items()):
                    diagnostic_rows[diag_idx]["entry_status"] = "rejected_daily_breaker"
                    pending_diag.pop(pending_symbol, None)
                pending.clear()
                break
            atr_v = pending.pop(symbol)
            diag_idx = pending_diag.pop(symbol, None)
            entry = float(df.loc[ts, "open"]) * (1 + entry_slippage)
            sl_dist = atr_v * STOP_LOSS_ATR_MULT
            tp_dist = atr_v * tp_mult
            raw_qty = (eq * RISK) / sl_dist if sl_dist > 0 else 0
            allocation_cap = eq * weights[symbol]
            max_notional = min(allocation_cap, cash / (1 + FEE_RATE))
            qty = min(raw_qty, max_notional / entry if entry > 0 else 0)
            notional = qty * entry
            if qty <= 0 or notional < MIN_NOTIONAL:
                if diag_idx is not None:
                    diagnostic_rows[diag_idx]["entry_status"] = "rejected_min_notional"
                continue
            entry_fee = notional * FEE_RATE
            if notional + entry_fee > cash:
                if diag_idx is not None:
                    diagnostic_rows[diag_idx]["entry_status"] = "rejected_cash"
                continue
            cash -= notional + entry_fee
            positions[symbol] = Position(symbol, entry, qty, entry-sl_dist, entry+tp_dist, ts, entry_fee, eq)
            if diag_idx is not None:
                diagnostic_rows[diag_idx]["entry_status"] = "executed"
                diagnostic_rows[diag_idx]["entry_time"] = ts
                diagnostic_rows[diag_idx]["entry_price"] = entry
                diagnostic_rows[diag_idx]["entry_notional"] = notional

        # SL/TP against intraday low/high; SL priority.
        for symbol in list(positions):
            df = data[symbol]["df"]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            pos = positions[symbol]
            open_px = float(row["open"])
            if open_px <= pos.stop_loss:
                reason, px = "SL_GAP", open_px * (1.0 - exit_slippage)
            elif open_px >= pos.take_profit:
                reason, px = "TP_GAP", pos.take_profit * (1.0 - extra_slippage)
            elif float(row["low"]) <= pos.stop_loss:
                reason, px = "SL", pos.stop_loss * (1.0 - exit_slippage)
            elif float(row["high"]) >= pos.take_profit:
                reason, px = "TP", pos.take_profit * (1.0 - extra_slippage)
            else:
                continue
            if reason:
                cash, trade = _close_trade(cash, pos, px, ts, reason)
                trades.append(trade)
                positions.pop(symbol)

        eq = _mtm(cash, positions, data, ts)
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        daily_loss = (day_start_eq - eq) / day_start_eq if day_start_eq > 0 else 0
        if dd >= MAX_DRAWDOWN_PCT:
            halted = True
            for pending_symbol, diag_idx in list(pending_diag.items()):
                diagnostic_rows[diag_idx]["entry_status"] = "rejected_halt"
                pending_diag.pop(pending_symbol, None)
            pending.clear()
        if daily_loss >= CIRCUIT_BREAKER_PCT:
            for pending_symbol, diag_idx in list(pending_diag.items()):
                diagnostic_rows[diag_idx]["entry_status"] = "rejected_daily_breaker"
                pending_diag.pop(pending_symbol, None)
            pending.clear()

        # Current close schedules the next open. Record one auditable decision per symbol/day.
        regime_on = _risk_on(data, ts)
        btc = data["BTCUSDT"]
        btc_close = float(btc["df"].loc[ts, "close"]) if ts in btc["df"].index else np.nan
        btc_sma = float(btc["regime_sma"].loc[ts]) if ts in btc["regime_sma"].index and pd.notna(btc["regime_sma"].loc[ts]) else np.nan
        for symbol in SYMBOLS:
            df = data[symbol]["df"]
            sig_series = data[symbol]["signals"]
            atr_series = data[symbol]["atr"]
            has_bar = ts in df.index
            signal = int(sig_series.loc[ts]) if has_bar and pd.notna(sig_series.loc[ts]) else np.nan
            av = float(atr_series.loc[ts]) if has_bar and pd.notna(atr_series.loc[ts]) else np.nan
            row = {
                "timestamp": ts,
                "symbol": symbol,
                "close": float(df.loc[ts, "close"]) if has_bar else np.nan,
                "signal": signal,
                "atr": av,
                "indicators_valid": bool(pd.notna(signal) and np.isfinite(av) and av > 0),
                "btc_close": btc_close,
                "btc_sma": btc_sma,
                "regime_on": bool(regime_on),
                "position_open": symbol in positions,
                "pending_before_decision": symbol in pending,
                "portfolio_slots_used": len(positions) + len(pending),
                "daily_loss": daily_loss,
                "halted": halted,
                "decision": "",
                "scheduled_for": pd.NaT,
                "entry_status": "",
                "entry_time": pd.NaT,
                "entry_price": np.nan,
                "entry_notional": np.nan,
            }
            if not has_bar:
                row["decision"] = "missing_bar"
            elif symbol in positions:
                row["decision"] = "position_open"
            elif symbol in pending:
                row["decision"] = "pending_entry"
            elif halted:
                row["decision"] = "blocked_halt"
            elif daily_loss >= CIRCUIT_BREAKER_PCT:
                row["decision"] = "blocked_daily_breaker"
            elif not regime_on:
                row["decision"] = "blocked_regime"
            elif len(positions) + len(pending) >= MAX_SIMULTANEOUS_POS:
                row["decision"] = "blocked_capacity"
            else:
                loc = df.index.get_loc(ts)
                if isinstance(loc, slice) or loc + 1 >= len(df):
                    row["decision"] = "no_next_bar"
                elif df.index[loc+1] > end:
                    row["decision"] = "next_bar_after_holdout"
                elif signal != 1:
                    row["decision"] = "no_signal"
                elif not np.isfinite(av) or av <= 0:
                    row["decision"] = "invalid_atr"
                else:
                    row["decision"] = "scheduled"
                    row["scheduled_for"] = df.index[loc+1]
                    pending[symbol] = av
            if capture_diagnostics:
                diagnostic_rows.append(row)
                if row["decision"] == "scheduled":
                    pending_diag[symbol] = len(diagnostic_rows) - 1
        equity_points.append((ts, eq))

    for symbol, diag_idx in pending_diag.items():
        diagnostic_rows[diag_idx]["entry_status"] = "expired_at_holdout_end"

    # EOD liquidation.
    for symbol in list(positions):
        df = data[symbol]["df"].loc[:end]
        if df.empty:
            continue
        ts = df.index[-1]
        pos = positions[symbol]
        px = float(df["close"].iloc[-1]) * (1.0 - exit_slippage)
        cash, trade = _close_trade(cash, pos, px, ts, "EOD")
        trades.append(trade)
        positions.pop(symbol)

    if equity_points:
        equity_points[-1] = (equity_points[-1][0], cash)
    eqs = pd.Series([x[1] for x in equity_points], index=[x[0] for x in equity_points], dtype=float)
    rets = eqs.pct_change().dropna()
    max_dd = float(((eqs - eqs.cummax()) / eqs.cummax()).min()) if not eqs.empty else 0.0
    sharpe = float(rets.mean()/rets.std()*np.sqrt(365)) if len(rets)>1 and rets.std()>0 else 0.0
    pnls = [t["pnl"] for t in trades]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    pf = sum(wins)/abs(sum(losses)) if losses and abs(sum(losses))>0 else (float("inf") if wins else 0.0)
    return {
        "return": cash/wf.INITIAL_CAPITAL - 1,
        "final_capital": cash,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": len(trades),
        "win_rate": len(wins)/len(trades) if trades else 0.0,
        "profit_factor": pf,
        "halted": halted,
        "trade_log": pd.DataFrame(trades) if capture_trades else pd.DataFrame(),
        "equity_curve": eqs,
        "diagnostics": pd.DataFrame(diagnostic_rows) if capture_diagnostics else pd.DataFrame(),
    }


def choose_tp(data, development_start, holdout_start, candidates, sma_window=200):
    dev_end = pd.Timestamp(holdout_start) - pd.Timedelta(days=1)
    scored = []
    for tp in candidates:
        m = simulate(data, development_start, dev_end, tp, sma_window, capture_trades=False)
        scored.append((m["return"], m["sharpe"], -tp, tp))
    return max(scored, key=lambda x:(x[0],x[1],x[2]))[3]


def block_bootstrap_monte_carlo(daily_returns, runs=10000, block_size=10, seed=42):
    """Circular block bootstrap of realized portfolio returns, preserving short volatility clusters."""
    r = pd.Series(daily_returns, dtype=float).dropna().to_numpy()
    if len(r) == 0:
        return {}
    n = len(r)
    rng = np.random.default_rng(seed)
    finals = np.empty(runs)
    maxdds = np.empty(runs)
    for i in range(runs):
        sample_parts = []
        while sum(len(x) for x in sample_parts) < n:
            start = int(rng.integers(0, n))
            idx = (np.arange(start, start + block_size) % n).astype(int)
            sample_parts.append(r[idx])
        sample = np.concatenate(sample_parts)[:n]
        curve = np.cumprod(1.0 + sample)
        full = np.r_[1.0, curve]
        peak = np.maximum.accumulate(full)
        dd = full/peak - 1.0
        finals[i] = curve[-1] - 1.0
        maxdds[i] = dd.min()
    return {
        "runs": runs,
        "observations": n,
        "block_size": block_size,
        "median_return": float(np.median(finals)),
        "p05_return": float(np.quantile(finals, .05)),
        "p95_return": float(np.quantile(finals, .95)),
        "loss_probability": float(np.mean(finals < 0)),
        "median_max_dd": float(np.median(maxdds)),
        "p95_adverse_dd": float(np.quantile(maxdds, .05)),
    }


def audit_market_data(data, symbols, start, end):
    """Summarize coverage and integrity issues without silently repairing market data."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    expected = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
    rows = []
    for symbol in symbols:
        item = data[symbol]
        df = item["df"].sort_index()
        period = df.loc[start:end]
        observed_days = pd.DatetimeIndex(period.index.normalize().unique())
        missing = expected.difference(observed_days)
        ohlcv = [c for c in ["open", "high", "low", "close", "volume"] if c in period]
        ohlc = [c for c in ["open", "high", "low", "close"] if c in period]
        rows.append({
            "symbol": symbol,
            "source": item.get("source", "unknown"),
            "dataset_first": df.index.min() if not df.empty else pd.NaT,
            "dataset_last": df.index.max() if not df.empty else pd.NaT,
            "requested_start": start,
            "requested_end": end,
            "period_first": period.index.min() if not period.empty else pd.NaT,
            "period_last": period.index.max() if not period.empty else pd.NaT,
            "expected_days": len(expected),
            "observed_rows": len(period),
            "missing_days": len(missing),
            "first_missing_day": missing.min() if len(missing) else pd.NaT,
            "last_missing_day": missing.max() if len(missing) else pd.NaT,
            "duplicate_timestamps": int(period.index.duplicated().sum()),
            "nan_ohlcv_rows": int(period[ohlcv].isna().any(axis=1).sum()) if ohlcv else len(period),
            "nonpositive_ohlc_rows": int((period[ohlc] <= 0).any(axis=1).sum()) if ohlc else len(period),
            "inconsistent_ohlc_rows": int((
                (period["high"] < period[["open", "close", "low"]].max(axis=1)) |
                (period["low"] > period[["open", "close", "high"]].min(axis=1))
            ).sum()) if len(ohlc) == 4 else len(period),
            "valid_atr_rows": int(item["atr"].reindex(period.index).notna().sum()),
            "raw_signal_rows": int((item["signals"].reindex(period.index) == 1).sum()),
        })
    return pd.DataFrame(rows)


def build_signal_funnel(diagnostics):
    """Aggregate the daily diagnostic trail while retaining every explicit block reason."""
    if diagnostics.empty:
        return pd.DataFrame()
    group_cols = ["holdout_start", "symbol"]
    base = diagnostics.groupby(group_cols, dropna=False).agg(
        candles=("timestamp", "size"),
        indicators_valid=("indicators_valid", "sum"),
        raw_signals=("signal", lambda x: int((x == 1).sum())),
        risk_on_days=("regime_on", "sum"),
        scheduled=("decision", lambda x: int((x == "scheduled").sum())),
        executed=("entry_status", lambda x: int((x == "executed").sum())),
    )
    reasons = diagnostics.pivot_table(
        index=group_cols,
        columns="decision",
        values="timestamp",
        aggfunc="size",
        fill_value=0,
    )
    reasons.columns = [f"decision_{c}" for c in reasons.columns]
    signal_rows = diagnostics[diagnostics["signal"] == 1]
    signal_reasons = signal_rows.pivot_table(
        index=group_cols,
        columns="decision",
        values="timestamp",
        aggfunc="size",
        fill_value=0,
    )
    if not signal_reasons.empty:
        signal_reasons.columns = [f"raw_signal_{c}" for c in signal_reasons.columns]
        base = base.join(signal_reasons, how="left")
    statuses = diagnostics[diagnostics["entry_status"].astype(str) != ""].pivot_table(
        index=group_cols,
        columns="entry_status",
        values="timestamp",
        aggfunc="size",
        fill_value=0,
    )
    if not statuses.empty:
        statuses.columns = [f"entry_{c}" for c in statuses.columns]
        base = base.join(statuses, how="left")
    return base.join(reasons, how="left").fillna(0).reset_index()


def _profit_factor(pnls):
    pnls = pd.Series(pnls, dtype=float)
    gains = float(pnls[pnls > 0].sum())
    losses = float(pnls[pnls <= 0].sum())
    if losses < 0:
        return gains / abs(losses)
    return float("inf") if gains > 0 else 0.0


def build_attribution(trades):
    """Return additive PnL attribution; it is not presented as a standalone equity curve."""
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["exit_year"] = pd.to_datetime(frame["exit_time"], utc=True).dt.year
    output = []
    for group_type, columns in [
        ("symbol", ["symbol"]),
        ("exit_year", ["exit_year"]),
        ("holdout_symbol", ["holdout_start", "symbol"]),
        ("exit_reason", ["reason"]),
    ]:
        for keys, group in frame.groupby(columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            labels = {column: value for column, value in zip(columns, keys)}
            wins = group.loc[group["pnl"] > 0, "pnl"]
            losses = group.loc[group["pnl"] <= 0, "pnl"]
            output.append({
                "group_type": group_type,
                "group_key": " | ".join(f"{k}={v}" for k, v in labels.items()),
                "trades": len(group),
                "wins": int((group["pnl"] > 0).sum()),
                "win_rate": float((group["pnl"] > 0).mean()),
                "total_pnl": float(group["pnl"].sum()),
                "pnl_points_on_10k": float(group["pnl"].sum() / wf.INITIAL_CAPITAL),
                "profit_factor": _profit_factor(group["pnl"]),
                "average_win": float(wins.mean()) if not wins.empty else 0.0,
                "average_loss": float(losses.mean()) if not losses.empty else 0.0,
                "average_holding_days": float(group["holding_days"].mean()),
            })
    return pd.DataFrame(output)


def stitch_oos_curves(curves):
    """Chain holdout daily returns chronologically and retain zero-return gaps."""
    if not curves:
        return pd.DataFrame(), {}
    pieces = []
    active_dates = set()
    for holdout_start, curve in curves:
        if curve.empty:
            continue
        returns = curve.pct_change().dropna()
        pieces.append(returns)
        active_dates.update(returns.index)
    if not pieces:
        return pd.DataFrame(), {}
    combined = pd.concat(pieces).sort_index()
    if combined.index.duplicated().any():
        raise ValueError("Holdouts sobrepostos: curva OOS contínua teria datas duplicadas.")
    full_index = pd.date_range(combined.index.min(), combined.index.max(), freq="D", tz="UTC")
    daily_returns = combined.reindex(full_index, fill_value=0.0).astype(float)
    equity = wf.INITIAL_CAPITAL * (1.0 + daily_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    std = daily_returns.std()
    metrics = {
        "start": full_index.min(),
        "end": full_index.max(),
        "days": len(full_index),
        "active_oos_days": len(active_dates),
        "return": float(equity.iloc[-1] / wf.INITIAL_CAPITAL - 1.0),
        "sharpe": float(daily_returns.mean() / std * np.sqrt(365)) if len(daily_returns) > 1 and std > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
    }
    out = pd.DataFrame({
        "timestamp": full_index,
        "daily_return": daily_returns.to_numpy(),
        "equity": equity.to_numpy(),
        "drawdown": drawdown.to_numpy(),
        "active_oos_day": [ts in active_dates for ts in full_index],
    })
    return out, metrics


def main():
    p = argparse.ArgumentParser(description="Krypton deep validation")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    p.add_argument("--holdouts", nargs="+", default=DEFAULT_HOLDOUTS)
    p.add_argument("--holdout-days", type=int, default=180)
    p.add_argument("--candidate-tp", nargs="+", type=float, default=wf.DEFAULT_CANDIDATE_TP)
    p.add_argument("--mc-runs", type=int, default=10000)
    p.add_argument("--diagnostics-year", type=int, default=2026,
                   help="Ano exportado também como diagnostics_YEAR.csv")
    p.add_argument("--stress-bps", nargs="+", type=float, default=[0, 5, 10, 20, 50],
                   help="Slippage adicional por lado para o stress de custos")
    p.add_argument("--tp-grid", nargs="+", type=float, default=[2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                   help="Grade diagnóstica; não é usada para escolher o TP OOS")
    args = p.parse_args()

    if any(x < 0 for x in args.stress_bps):
        p.error("--stress-bps aceita apenas valores não negativos.")
    if any(x <= 0 for x in args.tp_grid):
        p.error("--tp-grid aceita apenas valores positivos.")

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    data = wf._prepare_data(SYMBOLS, start, end)
    data_audit = audit_market_data(data, SYMBOLS, start, end)
    data_audit.to_csv("deep_validation_data_audit.csv", index=False)

    holdout_rows = []
    all_trades = []
    all_daily_returns = []
    all_diagnostics = []
    oos_curves = []
    sensitivity_rows = []
    cost_stress_rows = []
    tp_stability_rows = []

    for htxt in args.holdouts:
        hstart = pd.Timestamp(htxt, tz="UTC")
        if hstart <= pd.Timestamp(start) or hstart >= pd.Timestamp(end):
            continue
        hend = min(hstart + pd.Timedelta(days=args.holdout_days-1), pd.Timestamp(end))
        tp = choose_tp(data, start, hstart, args.candidate_tp, 200)
        m = simulate(
            data, hstart, hend, tp, 200,
            capture_trades=True, capture_diagnostics=True,
        )
        holdout_rows.append({"holdout_start":hstart.date(),"holdout_end":hend.date(),"tp":tp,
            "return":m["return"],"sharpe":m["sharpe"],"max_drawdown":m["max_drawdown"],
            "win_rate":m["win_rate"],"profit_factor":m["profit_factor"],"trades":m["trades"],"halted":m["halted"]})
        if not m["trade_log"].empty:
            t = m["trade_log"].copy(); t["holdout_start"] = hstart.date(); all_trades.append(t)
        if not m["equity_curve"].empty:
            all_daily_returns.append(m["equity_curve"].pct_change().dropna())
            oos_curves.append((hstart, m["equity_curve"]))
        if not m["diagnostics"].empty:
            d = m["diagnostics"].copy()
            d["holdout_start"] = hstart.date()
            d["holdout_end"] = hend.date()
            d["selected_tp"] = tp
            all_diagnostics.append(d)
        print(f"Holdout {hstart.date()}->{hend.date()} | TP={tp:.1f} | ret={m['return']:+.2%} | "
              f"Sharpe={m['sharpe']:.2f} | DD={m['max_drawdown']:.2%} | WR={m['win_rate']:.1%} | trades={m['trades']}")

        # Diagnostic sensitivity: same frozen TP, only SMA changes.
        for sma in SMA_WINDOWS:
            sm = simulate(data, hstart, hend, tp, sma, capture_trades=False)
            sensitivity_rows.append({"holdout_start":hstart.date(),"sma":sma,"tp":tp,
                "return":sm["return"],"sharpe":sm["sharpe"],"max_drawdown":sm["max_drawdown"],"trades":sm["trades"]})

        # Keep the pre-holdout-selected TP fixed while execution costs are stressed.
        for extra_bps in args.stress_bps:
            cm = simulate(
                data, hstart, hend, tp, 200, capture_trades=False,
                extra_slippage_bps=extra_bps,
            )
            cost_stress_rows.append({
                "holdout_start": hstart.date(), "holdout_end": hend.date(),
                "selected_tp": tp, "extra_slippage_bps_per_side": extra_bps,
                "return": cm["return"], "sharpe": cm["sharpe"],
                "max_drawdown": cm["max_drawdown"], "profit_factor": cm["profit_factor"],
                "trades": cm["trades"], "halted": cm["halted"],
            })

        # Diagnostic surface only: never feed these OOS results back into TP selection.
        for test_tp in args.tp_grid:
            tm = simulate(data, hstart, hend, test_tp, 200, capture_trades=False)
            tp_stability_rows.append({
                "holdout_start": hstart.date(), "holdout_end": hend.date(),
                "selected_tp": tp, "diagnostic_tp": test_tp,
                "return": tm["return"], "sharpe": tm["sharpe"],
                "max_drawdown": tm["max_drawdown"], "profit_factor": tm["profit_factor"],
                "trades": tm["trades"],
            })

    holdouts = pd.DataFrame(holdout_rows)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    diagnostics = pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame()
    sensitivity = pd.DataFrame(sensitivity_rows)
    cost_stress = pd.DataFrame(cost_stress_rows)
    tp_stability = pd.DataFrame(tp_stability_rows)
    funnel = build_signal_funnel(diagnostics)
    attribution = build_attribution(trades)
    continuous_oos, continuous_metrics = stitch_oos_curves(oos_curves)
    holdouts.to_csv("deep_validation_holdouts.csv", index=False)
    trades.to_csv("deep_validation_trades.csv", index=False)
    diagnostics.to_csv("deep_validation_signal_diagnostics.csv", index=False)
    if not diagnostics.empty:
        target_year = diagnostics[pd.to_datetime(diagnostics["timestamp"], utc=True).dt.year == args.diagnostics_year]
    else:
        target_year = diagnostics
    target_year.to_csv(f"diagnostics_{args.diagnostics_year}.csv", index=False)
    funnel.to_csv("deep_validation_signal_funnel.csv", index=False)
    attribution.to_csv("deep_validation_attribution.csv", index=False)
    sensitivity.to_csv("deep_validation_sma_sensitivity.csv", index=False)
    cost_stress.to_csv("deep_validation_cost_stress.csv", index=False)
    tp_stability.to_csv("deep_validation_tp_stability.csv", index=False)
    continuous_oos.to_csv("deep_validation_continuous_oos.csv", index=False)
    pd.DataFrame([continuous_metrics]).to_csv("deep_validation_continuous_oos_summary.csv", index=False)

    daily_returns = pd.concat(all_daily_returns, ignore_index=True) if all_daily_returns else pd.Series(dtype=float)
    mc = block_bootstrap_monte_carlo(daily_returns, args.mc_runs)
    pd.DataFrame([mc]).to_csv("deep_validation_monte_carlo.csv", index=False)

    print("\nDEEP VALIDATION SUMMARY")
    if holdouts.empty:
        print("Nenhum holdout válido.")
        return
    print(holdouts.to_string(index=False))
    compounded = float(np.prod(1+holdouts["return"].to_numpy())-1)
    print(f"\nHoldouts positivos: {(holdouts['return']>0).sum()}/{len(holdouts)}")
    print(f"Produto dos holdouts separados (não é retorno contínuo): {compounded:+.2%}")
    print(f"Retorno mediano/holdout: {holdouts['return'].median():+.2%}")
    print(f"Sharpe médio: {holdouts['sharpe'].mean():.3f}")
    print(f"Pior DD: {holdouts['max_drawdown'].min():.2%}")
    print(f"Trades reais agregados: {len(trades)}")

    if mc:
        print("\nPORTFOLIO BLOCK BOOTSTRAP MONTE CARLO")
        print(f"Runs: {mc['runs']}")
        print(f"Retorno mediano: {mc['median_return']:+.2%}")
        print(f"P05/P95 retorno: {mc['p05_return']:+.2%} / {mc['p95_return']:+.2%}")
        print(f"Probabilidade de retorno negativo: {mc['loss_probability']:.1%}")
        print(f"DD mediano: {mc['median_max_dd']:.2%}")
        print(f"DD P95 adverso: {mc['p95_adverse_dd']:.2%}")

    if not sensitivity.empty:
        print("\nSMA SENSITIVITY (median return by SMA)")
        print(sensitivity.groupby("sma")["return"].agg(["count","mean","median","min"]).to_string())

    print("\nDATA INTEGRITY AUDIT")
    print(data_audit[["symbol", "period_first", "period_last", "expected_days", "observed_rows",
                      "missing_days", "duplicate_timestamps", "nan_ohlcv_rows",
                      "nonpositive_ohlc_rows", "inconsistent_ohlc_rows",
                      "raw_signal_rows"]].to_string(index=False))

    if not funnel.empty:
        print("\nSIGNAL FUNNEL")
        funnel_core = [
            "holdout_start", "symbol", "candles", "indicators_valid", "raw_signals",
            "risk_on_days", "scheduled", "executed",
        ]
        funnel_core += sorted(c for c in funnel.columns if c.startswith("raw_signal_") and c != "raw_signals")
        print(funnel[funnel_core].to_string(index=False))

        zero_trade_starts = set(holdouts.loc[holdouts["trades"] == 0, "holdout_start"])
        if zero_trade_starts:
            print("\nZERO-TRADE HOLDOUT DIAGNOSIS")
            zero_funnel = funnel[funnel["holdout_start"].isin(zero_trade_starts)]
            print(zero_funnel[funnel_core].to_string(index=False))

    if continuous_metrics:
        print("\nCHRONOLOGICALLY STITCHED OOS CURVE")
        print(f"Retorno contínuo (gaps sem exposição): {continuous_metrics['return']:+.2%}")
        print(f"Sharpe contínuo: {continuous_metrics['sharpe']:.3f}")
        print(f"Max DD contínuo: {continuous_metrics['max_drawdown']:.2%}")

    if not cost_stress.empty:
        print("\nCOST STRESS (aggregate across separate holdouts)")
        cost_summary = cost_stress.groupby("extra_slippage_bps_per_side").agg(
            holdouts=("return", "size"),
            positive_holdouts=("return", lambda x: int((x > 0).sum())),
            median_return=("return", "median"),
            worst_return=("return", "min"),
            median_profit_factor=("profit_factor", "median"),
        )
        print(cost_summary.to_string())

    if not tp_stability.empty:
        print("\nTP STABILITY (diagnostic only; no OOS selection)")
        tp_summary = tp_stability.groupby("diagnostic_tp").agg(
            holdouts=("return", "size"),
            positive_holdouts=("return", lambda x: int((x > 0).sum())),
            median_return=("return", "median"),
            worst_return=("return", "min"),
        )
        print(tp_summary.to_string())

    print("\nArquivos principais: deep_validation_holdouts.csv, deep_validation_trades.csv, "
          "deep_validation_data_audit.csv, deep_validation_signal_diagnostics.csv, "
          f"diagnostics_{args.diagnostics_year}.csv, deep_validation_signal_funnel.csv, "
          "deep_validation_attribution.csv, deep_validation_cost_stress.csv, "
          "deep_validation_tp_stability.csv, deep_validation_continuous_oos.csv, "
          "deep_validation_continuous_oos_summary.csv, deep_validation_sma_sensitivity.csv, "
          "deep_validation_monte_carlo.csv")

if __name__ == "__main__":
    main()
