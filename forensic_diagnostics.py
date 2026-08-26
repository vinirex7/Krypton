"""Forensic diagnostics for Krypton research only.

Questions answered without changing the live strategy:
1) Why is bull-market capture low?
2) What specifically went wrong in 2025 H1 sideways/choppy conditions?

The simulator mirrors research_validation.simulate but records execution-neutral
telemetry: actual capital exposure, binding sizing constraint, exit decomposition,
post-exit asset returns, SMA200 crossings, signal availability and blocked entries.
No parameter search is performed and no live configuration is changed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import regime_diagnostics as rd
import research_validation as rv
import walk_forward as wf
from config import (
    CIRCUIT_BREAKER_PCT,
    ENTRY_SLIPPAGE_PCT,
    EXIT_SLIPPAGE_PCT,
    FEE_RATE,
    MAX_SIMULTANEOUS_POS,
    RISK_PER_TRADE,
    STOP_LOSS_ATR_MULT,
    TRADING_PAIRS,
)


@dataclass
class FPosition:
    symbol: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp
    equity_at_entry: float
    entry_notional: float
    risk_target: float
    actual_risk: float
    binding_cap: str


def _as_utc(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _mark(cash, positions, data, ts, field="close"):
    equity = cash
    gross = 0.0
    for symbol, pos in positions.items():
        df = data[symbol]["df"]
        if ts in df.index:
            px = float(df.loc[ts, field])
        else:
            eligible = df.loc[:ts]
            px = pos.entry_price if eligible.empty else float(eligible["close"].iloc[-1])
        value = pos.quantity * px
        equity += value
        gross += value
    return equity, gross


def _exit(cash, pos: FPosition, price: float):
    gross = pos.quantity * price
    exit_fee = gross * FEE_RATE
    entry_fee = pos.quantity * pos.entry_price * FEE_RATE
    pnl = pos.quantity * (price - pos.entry_price) - entry_fee - exit_fee
    return cash + gross - exit_fee, pnl


def _holding_days(entry_time, exit_time) -> int:
    return max(int((pd.Timestamp(exit_time) - pd.Timestamp(entry_time)).days), 0)


def _post_exit_return(data, symbol, exit_time, exit_price, days):
    df = data[symbol]["df"]
    eligible = df.loc[pd.Timestamp(exit_time):]
    if eligible.empty:
        return np.nan
    loc = df.index.get_indexer([eligible.index[0]])[0]
    target = loc + int(days)
    if target >= len(df):
        return np.nan
    return float(df["close"].iloc[target] / exit_price - 1.0)


def _btc_structure(data, start, end):
    btc = data["BTCUSDT"]
    df = btc["df"].loc[_as_utc(start):_as_utc(end)].copy()
    sma = btc["sma200"].reindex(df.index)
    above = (df["close"] > sma).fillna(False)
    crossings = int((above.astype(int).diff().abs() == 1).sum())
    rets = df["close"].pct_change().dropna()
    realized_vol = float(rets.std() * np.sqrt(365)) if len(rets) > 1 else 0.0
    max_dd = float(((df["close"] / df["close"].cummax()) - 1.0).min()) if not df.empty else 0.0
    if len(df) > 1:
        net = abs(float(df["close"].iloc[-1] - df["close"].iloc[0]))
        path = float(df["close"].diff().abs().sum())
        efficiency = net / path if path > 0 else 0.0
    else:
        efficiency = 0.0
    return {
        "btc_sma200_crossings": crossings,
        "btc_pct_days_above_sma200": float(above.mean()) if len(above) else 0.0,
        "btc_realized_vol": realized_vol,
        "btc_max_drawdown": max_dd,
        "btc_trend_efficiency": efficiency,
    }


def simulate_forensic(data, symbols, start, end, regime_mode="btc", weights=None):
    start_ts, end_ts = _as_utc(start), _as_utc(end)
    weights = weights or wf.normalized_weights(symbols)
    calendar = sorted(set().union(*[set(data[s]["df"].loc[start_ts:end_ts].index) for s in symbols]))
    if len(calendar) < 20:
        return {"daily": pd.DataFrame(), "trades": pd.DataFrame(), "summary": {}}

    cash = rv.INITIAL_CAPITAL
    positions: dict[str, FPosition] = {}
    pending: dict[str, dict] = {}
    trades = []
    daily_rows = []
    peak = rv.INITIAL_CAPITAL
    daily_start = rv.INITIAL_CAPITAL
    daily_date = None

    counters = {
        "long_signal_days": 0,
        "long_signal_asset_events": 0,
        "long_signal_blocked_regime": 0,
        "long_signal_blocked_capacity": 0,
        "entries_scheduled": 0,
        "entries_executed": 0,
        "entry_skipped_min_notional": 0,
        "entry_skipped_cash": 0,
        "circuit_breaker_blocks": 0,
    }

    def close_trade(symbol, pos, ts, px, reason):
        nonlocal cash
        cash, pnl = _exit(cash, pos, px)
        trades.append({
            "symbol": symbol,
            "entry_time": pos.entry_time,
            "exit_time": ts,
            "reason": reason,
            "pnl": pnl,
            "portfolio_return": pnl / pos.equity_at_entry if pos.equity_at_entry else 0.0,
            "entry_price": pos.entry_price,
            "exit_price": px,
            "quantity": pos.quantity,
            "entry_notional": pos.entry_notional,
            "entry_weight": pos.entry_notional / pos.equity_at_entry if pos.equity_at_entry else 0.0,
            "risk_target": pos.risk_target,
            "actual_risk": pos.actual_risk,
            "risk_utilization": pos.actual_risk / pos.risk_target if pos.risk_target > 0 else np.nan,
            "binding_cap": pos.binding_cap,
            "holding_days": _holding_days(pos.entry_time, ts),
            "post_exit_return_5d": _post_exit_return(data, symbol, ts, px, 5),
            "post_exit_return_10d": _post_exit_return(data, symbol, ts, px, 10),
            "post_exit_return_20d": _post_exit_return(data, symbol, ts, px, 20),
        })

    for ts in calendar:
        pre_eq, _ = _mark(cash, positions, data, ts, "open")
        if daily_date != ts.date():
            daily_date, daily_start = ts.date(), pre_eq

        # Signal exits at open.
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
                close_trade(symbol, pos, ts, px, "Sig")

        # Execute scheduled entries.
        for symbol in list(pending):
            if len(positions) >= MAX_SIMULTANEOUS_POS or symbol in positions:
                pending.pop(symbol, None)
                counters["long_signal_blocked_capacity"] += 1
                continue
            df = data[symbol]["df"]
            if ts not in df.index:
                continue
            current_eq, _ = _mark(cash, positions, data, ts, "open")
            daily_loss = (daily_start - current_eq) / daily_start if daily_start > 0 else 0.0
            if daily_loss >= CIRCUIT_BREAKER_PCT:
                pending.clear()
                counters["circuit_breaker_blocks"] += 1
                break

            event = pending.pop(symbol)
            atr_value = event["atr"]
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue
            entry = float(df.loc[ts, "open"]) * (1.0 + ENTRY_SLIPPAGE_PCT)
            sl_dist = atr_value * STOP_LOSS_ATR_MULT
            tp_dist = atr_value * rv.FIXED_TP
            risk_target = current_eq * RISK_PER_TRADE
            risk_qty = risk_target / sl_dist
            allocation_notional = current_eq * weights[symbol]
            allocation_qty = allocation_notional / entry if entry > 0 else 0.0
            cash_notional = cash / (1.0 + FEE_RATE)
            cash_qty = cash_notional / entry if entry > 0 else 0.0
            candidates = {"risk": risk_qty, "allocation": allocation_qty, "cash": cash_qty}
            binding_cap = min(candidates, key=candidates.get)
            qty = min(candidates.values())
            notional = qty * entry
            if qty <= 0 or notional < rv.MIN_NOTIONAL:
                counters["entry_skipped_min_notional"] += 1
                continue
            debit = notional * (1.0 + FEE_RATE)
            if debit > cash:
                counters["entry_skipped_cash"] += 1
                continue
            cash -= debit
            actual_risk = qty * sl_dist
            positions[symbol] = FPosition(
                symbol=symbol,
                entry_price=entry,
                quantity=qty,
                stop_loss=entry - sl_dist,
                take_profit=entry + tp_dist,
                entry_time=ts,
                equity_at_entry=current_eq,
                entry_notional=notional,
                risk_target=risk_target,
                actual_risk=actual_risk,
                binding_cap=binding_cap,
            )
            counters["entries_executed"] += 1

        # Intraday barriers.
        for symbol in list(positions):
            df = data[symbol]["df"]
            if ts not in df.index:
                continue
            row, pos = df.loc[ts], positions[symbol]
            op = float(row["open"])
            if op <= pos.stop_loss:
                reason, px = "SL_GAP", op * (1.0 - EXIT_SLIPPAGE_PCT)
            elif op >= pos.take_profit:
                reason, px = "TP_GAP", pos.take_profit
            elif float(row["low"]) <= pos.stop_loss:
                reason, px = "SL", pos.stop_loss * (1.0 - EXIT_SLIPPAGE_PCT)
            elif float(row["high"]) >= pos.take_profit:
                reason, px = "TP", pos.take_profit
            else:
                continue
            positions.pop(symbol)
            close_trade(symbol, pos, ts, px, reason)

        current_eq, gross = _mark(cash, positions, data, ts, "close")
        peak = max(peak, current_eq)
        daily_loss = (daily_start - current_eq) / daily_start if daily_start > 0 else 0.0
        if daily_loss >= CIRCUIT_BREAKER_PCT:
            pending.clear()

        any_long_signal = False
        signal_count = 0
        regime_allowed_assets = 0
        if daily_loss < CIRCUIT_BREAKER_PCT:
            for symbol in symbols:
                df, sig, atr = data[symbol]["df"], data[symbol]["signals"], data[symbol]["atr"]
                if ts not in df.index:
                    continue
                loc = df.index.get_loc(ts)
                if isinstance(loc, slice) or loc + 1 >= len(df) or df.index[loc + 1] > end_ts:
                    continue
                allowed = rv.regime_allows(data, symbols, symbol, ts, regime_mode)
                regime_allowed_assets += int(allowed)
                signal_is_long = int(sig.loc[ts]) == 1
                if signal_is_long:
                    any_long_signal = True
                    signal_count += 1
                    counters["long_signal_asset_events"] += 1
                    if not allowed:
                        counters["long_signal_blocked_regime"] += 1
                        continue
                if not signal_is_long or not allowed:
                    continue
                if symbol in positions or symbol in pending:
                    continue
                if len(positions) + len(pending) >= MAX_SIMULTANEOUS_POS:
                    counters["long_signal_blocked_capacity"] += 1
                    continue
                av = float(atr.loc[ts]) if pd.notna(atr.loc[ts]) else np.nan
                if np.isfinite(av) and av > 0:
                    pending[symbol] = {"atr": av, "signal_time": ts}
                    counters["entries_scheduled"] += 1
        if any_long_signal:
            counters["long_signal_days"] += 1

        daily_rows.append({
            "time": ts,
            "equity": current_eq,
            "cash": cash,
            "gross_position_value": gross,
            "gross_exposure_ratio": gross / current_eq if current_eq > 0 else 0.0,
            "cash_ratio": cash / current_eq if current_eq > 0 else 0.0,
            "positions_count": len(positions),
            "pending_count": len(pending),
            "long_signal_count": signal_count,
            "regime_allowed_assets": regime_allowed_assets,
            **{f"pos_{s}": int(s in positions) for s in symbols},
        })

    # End liquidation.
    for symbol in list(positions):
        df = data[symbol]["df"].loc[:end_ts]
        if df.empty:
            continue
        ts = df.index[-1]
        pos = positions.pop(symbol)
        px = float(df["close"].iloc[-1]) * (1.0 - EXIT_SLIPPAGE_PCT)
        close_trade(symbol, pos, ts, px, "EOD")

    daily = pd.DataFrame(daily_rows).set_index("time") if daily_rows else pd.DataFrame()
    trade_df = pd.DataFrame(trades)
    final_capital = cash
    if not daily.empty:
        daily.iloc[-1, daily.columns.get_loc("equity")] = final_capital

    pnl_by_symbol = trade_df.groupby("symbol")["pnl"].sum().to_dict() if not trade_df.empty else {}
    reason_counts = trade_df["reason"].value_counts().to_dict() if not trade_df.empty else {}
    reason_pnl = trade_df.groupby("reason")["pnl"].sum().to_dict() if not trade_df.empty else {}
    binding_counts = trade_df["binding_cap"].value_counts().to_dict() if not trade_df.empty else {}

    losing_streak = 0
    max_losing_streak = 0
    if not trade_df.empty:
        for pnl in trade_df.sort_values("exit_time")["pnl"]:
            if pnl <= 0:
                losing_streak += 1
                max_losing_streak = max(max_losing_streak, losing_streak)
            else:
                losing_streak = 0

    summary = {
        "start": start_ts.date().isoformat(),
        "end": end_ts.date().isoformat(),
        "regime_mode": regime_mode,
        "strategy_return": final_capital / rv.INITIAL_CAPITAL - 1.0,
        "trades": int(len(trade_df)),
        "winning_trades": int((trade_df["pnl"] > 0).sum()) if not trade_df.empty else 0,
        "losing_trades": int((trade_df["pnl"] <= 0).sum()) if not trade_df.empty else 0,
        "max_losing_streak": int(max_losing_streak),
        "mean_gross_exposure": float(daily["gross_exposure_ratio"].mean()) if not daily.empty else 0.0,
        "median_gross_exposure": float(daily["gross_exposure_ratio"].median()) if not daily.empty else 0.0,
        "pct_days_zero_exposure": float((daily["gross_exposure_ratio"] <= 1e-12).mean()) if not daily.empty else 1.0,
        "mean_cash_ratio": float(daily["cash_ratio"].mean()) if not daily.empty else 1.0,
        "mean_entry_weight": float(trade_df["entry_weight"].mean()) if not trade_df.empty else 0.0,
        "mean_risk_utilization": float(trade_df["risk_utilization"].mean()) if not trade_df.empty else 0.0,
        "median_holding_days": float(trade_df["holding_days"].median()) if not trade_df.empty else 0.0,
        "short_holds_le_5d": int((trade_df["holding_days"] <= 5).sum()) if not trade_df.empty else 0,
        "median_post_exit_return_5d": float(trade_df["post_exit_return_5d"].median()) if not trade_df.empty else np.nan,
        "median_post_exit_return_10d": float(trade_df["post_exit_return_10d"].median()) if not trade_df.empty else np.nan,
        "median_post_exit_return_20d": float(trade_df["post_exit_return_20d"].median()) if not trade_df.empty else np.nan,
        "pnl_by_symbol": pnl_by_symbol,
        "exit_reason_counts": reason_counts,
        "exit_reason_pnl": reason_pnl,
        "binding_cap_counts": binding_counts,
        **counters,
        **_btc_structure(data, start_ts, end_ts),
    }
    return {"daily": daily, "trades": trade_df, "summary": summary}


def _asset_period_returns(data, symbols, start, end):
    out = {}
    for symbol in symbols:
        df = data[symbol]["df"].loc[_as_utc(start):_as_utc(end)]
        out[symbol] = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0) if len(df) > 1 else np.nan
    return out


def _flatten_summary(summary: dict):
    row = {}
    for k, v in summary.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                row[f"{k}.{kk}"] = vv
        else:
            row[k] = v
    return row


def main():
    p = argparse.ArgumentParser(description="Krypton forensic bull/sideways diagnostics")
    p.add_argument("--start", default="2020-08-01")
    p.add_argument("--end", default="2026-06-30")
    p.add_argument("--window-days", type=int, default=180)
    p.add_argument("--focus-start", default="2025-01-07")
    p.add_argument("--focus-end", default="2025-07-05")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    focus_start = datetime.strptime(args.focus_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    focus_end = datetime.strptime(args.focus_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    symbols = list(TRADING_PAIRS)
    data = wf._prepare_data(symbols, start, end)

    rows = []
    all_trades = []
    all_daily = []
    detailed = {}

    for period, ps, pe in rd._periods(start, end, args.window_days):
        btc_return = rd._btc_period_return(data, ps, pe)
        if not np.isfinite(btc_return):
            continue
        market_regime = rd.classify_period(btc_return)
        result = simulate_forensic(data, symbols, ps, pe, regime_mode="btc")
        summary = result["summary"]
        summary.update({
            "period": period,
            "market_regime": market_regime,
            "btc_return": btc_return,
            "bull_capture_ratio": summary["strategy_return"] / btc_return if market_regime == "bull" and btc_return > 0 else np.nan,
            "asset_returns": _asset_period_returns(data, symbols, ps, pe),
        })
        rows.append(_flatten_summary(summary))
        if not result["trades"].empty:
            t = result["trades"].copy()
            t.insert(0, "period", period)
            t.insert(1, "market_regime", market_regime)
            all_trades.append(t)
        if not result["daily"].empty:
            d = result["daily"].copy()
            d.insert(0, "period", period)
            d.insert(1, "market_regime", market_regime)
            all_daily.append(d.reset_index())

    focus = simulate_forensic(data, symbols, focus_start, focus_end, regime_mode="btc")
    detailed["focus_2025_h1"] = focus["summary"]

    period_df = pd.DataFrame(rows)
    trade_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()

    bull = period_df[period_df["market_regime"] == "bull"] if not period_df.empty else pd.DataFrame()
    forensic_report = {
        "focus_2025_h1": focus["summary"],
        "bull_aggregate": {
            "periods": int(len(bull)),
            "mean_strategy_return": float(bull["strategy_return"].mean()) if len(bull) else np.nan,
            "mean_btc_return": float(bull["btc_return"].mean()) if len(bull) else np.nan,
            "mean_bull_capture": float(bull["bull_capture_ratio"].mean()) if len(bull) else np.nan,
            "mean_gross_exposure": float(bull["mean_gross_exposure"].mean()) if len(bull) else np.nan,
            "mean_entry_weight": float(bull["mean_entry_weight"].mean()) if len(bull) else np.nan,
            "mean_risk_utilization": float(bull["mean_risk_utilization"].mean()) if len(bull) else np.nan,
            "median_post_exit_return_20d": float(bull["median_post_exit_return_20d"].median()) if len(bull) else np.nan,
        },
    }

    period_df.to_csv("forensic_period_summary.csv", index=False)
    trade_df.to_csv("forensic_trades.csv", index=False)
    daily_df.to_csv("forensic_daily.csv", index=False)
    focus["trades"].to_csv("forensic_2025h1_trades.csv", index=False)
    focus["daily"].to_csv("forensic_2025h1_daily.csv")
    with open("forensic_report.json", "w", encoding="utf-8") as fh:
        json.dump(forensic_report, fh, indent=2, ensure_ascii=False, default=lambda x: None if pd.isna(x) else x)

    print("\nFORENSIC PERIOD SUMMARY")
    keep = [c for c in [
        "period", "start", "end", "market_regime", "btc_return", "strategy_return",
        "bull_capture_ratio", "mean_gross_exposure", "mean_cash_ratio", "mean_entry_weight",
        "mean_risk_utilization", "trades", "max_losing_streak", "btc_sma200_crossings",
        "btc_pct_days_above_sma200", "btc_trend_efficiency", "median_post_exit_return_20d",
    ] if c in period_df.columns]
    print(period_df[keep].to_string(index=False) if not period_df.empty else "Sem dados")
    print("\nFORENSIC REPORT")
    print(json.dumps(forensic_report, indent=2, ensure_ascii=False, default=lambda x: None if pd.isna(x) else x))
    print("\nArquivos: forensic_period_summary.csv | forensic_trades.csv | forensic_daily.csv | forensic_report.json")


if __name__ == "__main__":
    main()
