"""Deep validation for Krypton no_eth_regime candidate.

Frozen architecture:
- SOLUSDT, BTCUSDT, BNBUSDT
- 1% risk/trade
- LONG-only spot
- BTC close > SMA regime filter
- TP selected only from data strictly before each holdout

This script adds:
1. Exact trade log export (entry/exit timestamps, symbol, prices, qty, fees, PnL, return, reason).
2. Exact bootstrap Monte Carlo using realized OOS trade returns, not synthetic returns.
3. Multiple frozen pseudo-holdouts. Each holdout chooses TP only from prior data.
4. SMA sensitivity 180/200/220 as a robustness diagnostic, never as an OOS optimizer.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import walk_forward as wf
from config import (
    CIRCUIT_BREAKER_PCT, FEE_RATE, MAX_DRAWDOWN_PCT, MAX_SIMULTANEOUS_POS,
    RISK_PER_TRADE, STOP_LOSS_ATR_MULT,
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


def _mtm(cash, positions, data, ts):
    equity = cash
    for symbol, pos in positions.items():
        df = data[symbol]["df"]
        eligible = df.loc[:ts]
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
        "reason": reason,
        "holding_days": max((exit_time - pos.entry_time).days, 0),
    }
    return cash, trade


def simulate(data, start, end, tp_mult, sma_window=200, capture_trades=True):
    _set_sma(data, sma_window)
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")

    weights = wf.normalized_weights(SYMBOLS)
    calendar = sorted(set().union(*[set(data[s]["df"].loc[start:end].index) for s in SYMBOLS]))
    cash = wf.INITIAL_CAPITAL
    positions = {}
    pending = {}
    trades = []
    equity_points = []
    peak = cash
    day0 = None
    day_start_eq = cash
    halted = False

    for ts in calendar:
        pre_eq = _mtm(cash, positions, data, ts)
        if day0 != ts.date():
            day0 = ts.date()
            day_start_eq = pre_eq

        # Previous close -> current open entries.
        for symbol in list(pending):
            if halted or symbol in positions or len(positions) >= MAX_SIMULTANEOUS_POS:
                pending.pop(symbol, None)
                continue
            df = data[symbol]["df"]
            if ts not in df.index:
                continue
            eq = _mtm(cash, positions, data, ts)
            daily_loss = (day_start_eq - eq) / day_start_eq if day_start_eq > 0 else 0
            if daily_loss >= CIRCUIT_BREAKER_PCT:
                pending.clear()
                break
            atr_v = pending.pop(symbol)
            entry = float(df.loc[ts, "open"]) * (1 + ENTRY_SLIPPAGE_PCT)
            sl_dist = atr_v * STOP_LOSS_ATR_MULT
            tp_dist = atr_v * tp_mult
            raw_qty = (eq * RISK) / sl_dist if sl_dist > 0 else 0
            allocation_cap = eq * weights[symbol]
            max_notional = min(allocation_cap, cash / (1 + FEE_RATE))
            qty = min(raw_qty, max_notional / entry if entry > 0 else 0)
            notional = qty * entry
            if qty <= 0 or notional < MIN_NOTIONAL:
                continue
            entry_fee = notional * FEE_RATE
            if notional + entry_fee > cash:
                continue
            cash -= notional + entry_fee
            positions[symbol] = Position(symbol, entry, qty, entry-sl_dist, entry+tp_dist, ts, entry_fee)

        # SL/TP against intraday low/high; SL priority.
        for symbol in list(positions):
            df = data[symbol]["df"]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            pos = positions[symbol]
            hit_sl = float(row["low"]) <= pos.stop_loss
            hit_tp = float(row["high"]) >= pos.take_profit
            if hit_sl or hit_tp:
                reason = "SL" if hit_sl else "TP"
                px = pos.stop_loss if hit_sl else pos.take_profit
                cash, trade = _close_trade(cash, pos, px, ts, reason)
                trades.append(trade)
                positions.pop(symbol)

        # Previous close signal exit -> current open.
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
                px = float(df.loc[ts, "open"])
                cash, trade = _close_trade(cash, pos, px, ts, "Sig")
                trades.append(trade)
                positions.pop(symbol)

        eq = _mtm(cash, positions, data, ts)
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        daily_loss = (day_start_eq - eq) / day_start_eq if day_start_eq > 0 else 0
        if dd >= MAX_DRAWDOWN_PCT:
            halted = True
            pending.clear()
        if daily_loss >= CIRCUIT_BREAKER_PCT:
            pending.clear()

        # New entries only in BTC risk-on regime.
        if not halted and daily_loss < CIRCUIT_BREAKER_PCT and _risk_on(data, ts):
            for symbol in SYMBOLS:
                if symbol in positions or symbol in pending:
                    continue
                if len(positions) + len(pending) >= MAX_SIMULTANEOUS_POS:
                    break
                df = data[symbol]["df"]
                if ts not in df.index:
                    continue
                loc = df.index.get_loc(ts)
                if isinstance(loc, slice) or loc + 1 >= len(df):
                    continue
                if df.index[loc+1] > end:
                    continue
                sig = data[symbol]["signals"]
                atr = data[symbol]["atr"]
                av = atr.loc[ts]
                if int(sig.loc[ts]) == 1 and pd.notna(av) and float(av) > 0:
                    pending[symbol] = float(av)
        equity_points.append((ts, eq))

    # EOD liquidation.
    for symbol in list(positions):
        df = data[symbol]["df"].loc[:end]
        if df.empty:
            continue
        ts = df.index[-1]
        pos = positions[symbol]
        px = float(df["close"].iloc[-1])
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
    }


def choose_tp(data, development_start, holdout_start, candidates, sma_window=200):
    dev_end = pd.Timestamp(holdout_start) - pd.Timedelta(days=1)
    scored = []
    for tp in candidates:
        m = simulate(data, development_start, dev_end, tp, sma_window, capture_trades=False)
        scored.append((m["return"], m["sharpe"], -tp, tp))
    return max(scored, key=lambda x:(x[0],x[1],x[2]))[3]


def bootstrap_monte_carlo(trades, runs=10000, seed=42):
    """Bootstrap realized OOS trade returns with replacement and compute terminal return/DD."""
    if trades.empty:
        return {}
    r = trades["trade_return"].astype(float).to_numpy()
    n = len(r)
    rng = np.random.default_rng(seed)
    finals = np.empty(runs)
    maxdds = np.empty(runs)
    for i in range(runs):
        sample = rng.choice(r, size=n, replace=True)
        curve = np.cumprod(1.0 + sample)
        full = np.r_[1.0, curve]
        peak = np.maximum.accumulate(full)
        dd = full/peak - 1.0
        finals[i] = curve[-1] - 1.0
        maxdds[i] = dd.min()
    return {
        "runs": runs,
        "median_return": float(np.median(finals)),
        "p05_return": float(np.quantile(finals, .05)),
        "p95_return": float(np.quantile(finals, .95)),
        "loss_probability": float(np.mean(finals < 0)),
        "median_max_dd": float(np.median(maxdds)),
        "p95_adverse_dd": float(np.quantile(maxdds, .05)),
    }


def main():
    p = argparse.ArgumentParser(description="Krypton deep validation")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    p.add_argument("--holdouts", nargs="+", default=DEFAULT_HOLDOUTS)
    p.add_argument("--holdout-days", type=int, default=180)
    p.add_argument("--candidate-tp", nargs="+", type=float, default=wf.DEFAULT_CANDIDATE_TP)
    p.add_argument("--mc-runs", type=int, default=10000)
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    data = wf._prepare_data(SYMBOLS, start, end)

    holdout_rows = []
    all_trades = []
    sensitivity_rows = []

    for htxt in args.holdouts:
        hstart = pd.Timestamp(htxt, tz="UTC")
        if hstart <= pd.Timestamp(start) or hstart >= pd.Timestamp(end):
            continue
        hend = min(hstart + pd.Timedelta(days=args.holdout_days-1), pd.Timestamp(end))
        tp = choose_tp(data, start, hstart, args.candidate_tp, 200)
        m = simulate(data, hstart, hend, tp, 200, capture_trades=True)
        holdout_rows.append({"holdout_start":hstart.date(),"holdout_end":hend.date(),"tp":tp,
            "return":m["return"],"sharpe":m["sharpe"],"max_drawdown":m["max_drawdown"],
            "win_rate":m["win_rate"],"profit_factor":m["profit_factor"],"trades":m["trades"],"halted":m["halted"]})
        if not m["trade_log"].empty:
            t = m["trade_log"].copy(); t["holdout_start"] = hstart.date(); all_trades.append(t)
        print(f"Holdout {hstart.date()}->{hend.date()} | TP={tp:.1f} | ret={m['return']:+.2%} | "
              f"Sharpe={m['sharpe']:.2f} | DD={m['max_drawdown']:.2%} | WR={m['win_rate']:.1%} | trades={m['trades']}")

        # Diagnostic sensitivity: same frozen TP, only SMA changes.
        for sma in SMA_WINDOWS:
            sm = simulate(data, hstart, hend, tp, sma, capture_trades=False)
            sensitivity_rows.append({"holdout_start":hstart.date(),"sma":sma,"tp":tp,
                "return":sm["return"],"sharpe":sm["sharpe"],"max_drawdown":sm["max_drawdown"],"trades":sm["trades"]})

    holdouts = pd.DataFrame(holdout_rows)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    sensitivity = pd.DataFrame(sensitivity_rows)
    holdouts.to_csv("deep_validation_holdouts.csv", index=False)
    trades.to_csv("deep_validation_trades.csv", index=False)
    sensitivity.to_csv("deep_validation_sma_sensitivity.csv", index=False)

    mc = bootstrap_monte_carlo(trades, args.mc_runs)
    pd.DataFrame([mc]).to_csv("deep_validation_monte_carlo.csv", index=False)

    print("\nDEEP VALIDATION SUMMARY")
    if holdouts.empty:
        print("Nenhum holdout válido.")
        return
    print(holdouts.to_string(index=False))
    compounded = float(np.prod(1+holdouts["return"].to_numpy())-1)
    print(f"\nHoldouts positivos: {(holdouts['return']>0).sum()}/{len(holdouts)}")
    print(f"Retorno composto dos holdouts: {compounded:+.2%}")
    print(f"Retorno mediano/holdout: {holdouts['return'].median():+.2%}")
    print(f"Sharpe médio: {holdouts['sharpe'].mean():.3f}")
    print(f"Pior DD: {holdouts['max_drawdown'].min():.2%}")
    print(f"Trades reais agregados: {len(trades)}")

    if mc:
        print("\nEXACT TRADE BOOTSTRAP MONTE CARLO")
        print(f"Runs: {mc['runs']}")
        print(f"Retorno mediano: {mc['median_return']:+.2%}")
        print(f"P05/P95 retorno: {mc['p05_return']:+.2%} / {mc['p95_return']:+.2%}")
        print(f"Probabilidade de retorno negativo: {mc['loss_probability']:.1%}")
        print(f"DD mediano: {mc['median_max_dd']:.2%}")
        print(f"DD P95 adverso: {mc['p95_adverse_dd']:.2%}")

    if not sensitivity.empty:
        print("\nSMA SENSITIVITY (median return by SMA)")
        print(sensitivity.groupby("sma")["return"].agg(["count","mean","median","min"]).to_string())

    print("\nArquivos: deep_validation_holdouts.csv, deep_validation_trades.csv, "
          "deep_validation_sma_sensitivity.csv, deep_validation_monte_carlo.csv")

if __name__ == "__main__":
    main()
