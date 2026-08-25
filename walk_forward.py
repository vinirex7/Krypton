"""Walk-forward validation for Krypton Spot strategy.

Uses Binance Spot daily OHLCV, keeps the strategy fixed except TP optimization,
and selects TP on each training window before evaluating the following test window.
"""
import argparse
import contextlib
import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import backtest
from config import FEE_RATE, RISK_PER_TRADE, STOP_LOSS_ATR_MULT, CIRCUIT_BREAKER_PCT, MAX_DRAWDOWN_PCT, MAX_SIMULTANEOUS_POS
from indicators import compute_atr, compute_signals

CANDIDATE_TP = [3.0, 4.0, 4.5, 5.0]
TRAIN_DAYS = 365
TEST_DAYS = 180
STEP_DAYS = 180


def simulate(df, signals, atr, start, end, tp_mult):
    d = df.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")].copy()
    s = signals.loc[d.index]
    a = atr.loc[d.index]
    if len(d) < 20:
        return {"return": 0.0, "trades": 0, "win_rate": 0.0, "pf": 0.0, "max_dd": 0.0}
    capital = 10000.0
    peak = capital
    daily_start = capital
    daily_date = None
    positions = []
    trades = []
    pending = None

    def reset_day(day):
        nonlocal daily_date, daily_start
        if daily_date != day:
            daily_date, daily_start = day, capital

    for i in range(len(d)):
        row = d.iloc[i]
        reset_day(d.index[i].date())
        if pending is not None and not positions and i > 0:
            av = pending
            entry = float(row.open) * 1.0005
            sl_dist = av * STOP_LOSS_ATR_MULT
            tp_dist = av * tp_mult
            qty = min((capital * RISK_PER_TRADE) / sl_dist, capital / (entry * (1 + FEE_RATE))) if sl_dist > 0 else 0
            if qty > 0 and qty * entry >= 10:
                capital -= qty * entry * FEE_RATE
                positions.append({"entry": entry, "qty": qty, "sl": entry-sl_dist, "tp": entry+tp_dist})
            pending = None

        if positions:
            p = positions[0]
            hit_sl = float(row.low) <= p["sl"]
            hit_tp = float(row.high) >= p["tp"]
            if hit_sl or hit_tp:
                px = p["sl"] if hit_sl else p["tp"]
                pnl = p["qty"] * (px-p["entry"]) - p["qty"] * px * FEE_RATE
                capital += pnl
                trades.append(pnl)
                positions.clear()
            elif i > 0 and int(s.iloc[i-1]) != 1:
                px = float(row.open)
                pnl = p["qty"] * (px-p["entry"]) - p["qty"] * px * FEE_RATE
                capital += pnl
                trades.append(pnl)
                positions.clear()

        peak = max(peak, capital)
        if (peak-capital)/peak >= MAX_DRAWDOWN_PCT:
            pending = None
            continue
        if (daily_start-capital)/daily_start >= CIRCUIT_BREAKER_PCT:
            pending = None
            continue
        if not positions and pending is None and int(s.iloc[i]) == 1 and pd.notna(a.iloc[i]) and a.iloc[i] > 0 and i+1 < len(d):
            pending = float(a.iloc[i])

    if positions:
        p = positions[0]
        px = float(d.close.iloc[-1])
        pnl = p["qty"] * (px-p["entry"]) - p["qty"] * px * FEE_RATE
        capital += pnl
        trades.append(pnl)

    eq_return = capital / 10000.0 - 1
    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x <= 0]
    pf = sum(wins)/abs(sum(losses)) if losses else float("inf")
    return {"return": eq_return, "trades": len(trades), "win_rate": len(wins)/len(trades) if trades else 0, "pf": pf, "max_dd": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--symbols", nargs="+", default=["SOLUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT"])
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    rows = []
    for symbol in args.symbols:
        print(f"\n### {symbol}")
        warm = start - timedelta(days=400)
        df, source = backtest.get_ohlcv(symbol, warm.strftime("%Y-%m-%d"), args.end)
        df = df.sort_index()
        signals = compute_signals(df)
        atr = compute_atr(df.high, df.low, df.close)
        fold_start = start
        while fold_start + timedelta(days=TRAIN_DAYS+TEST_DAYS) <= end:
            train_start = fold_start
            train_end = fold_start + timedelta(days=TRAIN_DAYS-1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=TEST_DAYS-1)
            scores = []
            for tp in CANDIDATE_TP:
                with contextlib.redirect_stdout(io.StringIO()):
                    m = simulate(df, signals, atr, train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d'), tp)
                scores.append((m["return"], tp, m))
            _, best_tp, train_m = max(scores, key=lambda x: x[0])
            with contextlib.redirect_stdout(io.StringIO()):
                test_m = simulate(df, signals, atr, test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), best_tp)
            rows.append({"symbol":symbol,"train_start":train_start.date(),"train_end":train_end.date(),"test_start":test_start.date(),"test_end":test_end.date(),"selected_tp":best_tp,"train_return":train_m["return"],"test_return":test_m["return"],"test_trades":test_m["trades"],"test_win_rate":test_m["win_rate"],"test_profit_factor":test_m["pf"],"source":source})
            fold_start += timedelta(days=STEP_DAYS)
    out = pd.DataFrame(rows)
    out.to_csv("walk_forward_results.csv", index=False)
    print("\nWALK-FORWARD RESULTS")
    print(out.to_string(index=False))
    if not out.empty:
        print("\nSUMMARY")
        print(out.groupby("symbol")["test_return"].agg(["count","mean","sum"]))

if __name__ == "__main__":
    main()
