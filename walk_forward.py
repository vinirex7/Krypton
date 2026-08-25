"""Walk-forward validation for Krypton Spot strategy.

This module performs a realistic rolling walk-forward on Binance Spot daily OHLCV.
It keeps the strategy logic aligned with ``backtest.py`` and only optimizes the
Take-Profit ATR multiplier inside each training window. The selected parameter is
then frozen and evaluated on the immediately following out-of-sample test window.

Key properties
--------------
- Spot LONG-only.
- Signal generated on candle close and executed on next candle open.
- SL/TP evaluated against candle low/high, with SL priority if both are touched.
- Entry and exit Binance fees included.
- 5 bps entry slippage included, matching backtest.py.
- 1% risk per trade from TOTAL portfolio equity.
- Capital/notional cap prevents implicit leverage.
- Portfolio-level MAX_SIMULTANEOUS_POS, circuit breaker and max-drawdown halt.
- Training objective is portfolio return, not per-asset return.
- Out-of-sample folds never influence parameter selection.

The live bot uses U while historical validation uses USDT pairs.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import backtest
from config import (
    CIRCUIT_BREAKER_PCT,
    FEE_RATE,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MAX_DRAWDOWN_PCT,
    MAX_SIMULTANEOUS_POS,
    RISK_PER_TRADE,
    RSI_HIGH,
    RSI_LOW,
    RSI_PERIOD,
    STOP_LOSS_ATR_MULT,
    SUPERTREND_MULTIPLIER,
    SUPERTREND_PERIOD,
)
from indicators import compute_atr, compute_signals

INITIAL_CAPITAL = 10_000.0
ENTRY_SLIPPAGE_PCT = 0.0005
CANDIDATE_TP = [3.0, 4.0, 4.5, 5.0]
TRAIN_DAYS = 365
TEST_DAYS = 180
STEP_DAYS = 180
MIN_NOTIONAL = 10.0

# Mesmos pesos usados no live. Os símbolos históricos são USDT.
PORTFOLIO_WEIGHTS = {
    "SOLUSDT": 0.25,
    "BTCUSDT": 0.40,
    "ETHUSDT": 0.20,
    "BNBUSDT": 0.15,
}


@dataclass
class Position:
    symbol: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp


def _prepare_data(symbols: list[str], start: datetime, end: datetime):
    """Download market data once and precompute indicators over warm-up history."""
    data = {}
    warmup_start = start - timedelta(days=400)

    for symbol in symbols:
        print(f"Baixando {symbol}...")
        df, source = backtest.get_ohlcv(
            symbol,
            warmup_start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        if df.empty:
            raise RuntimeError(f"Sem dados suficientes para {symbol}.")

        df = df.sort_index()
        signals = compute_signals(
            df,
            st_period=SUPERTREND_PERIOD,
            st_mult=SUPERTREND_MULTIPLIER,
            rsi_period=RSI_PERIOD,
            rsi_low=RSI_LOW,
            rsi_high=RSI_HIGH,
            macd_fast=MACD_FAST,
            macd_slow=MACD_SLOW,
            macd_sig=MACD_SIGNAL,
        )
        atr = compute_atr(df["high"], df["low"], df["close"])
        data[symbol] = {
            "df": df,
            "signals": signals,
            "atr": atr,
            "source": source,
        }

    return data


def _portfolio_mark_to_market(cash: float, positions: dict[str, Position], data, ts) -> float:
    equity = cash
    for symbol, pos in positions.items():
        df = data[symbol]["df"]
        if ts in df.index:
            px = float(df.loc[ts, "close"])
        else:
            eligible = df.loc[:ts]
            if eligible.empty:
                px = pos.entry_price
            else:
                px = float(eligible["close"].iloc[-1])
        equity += pos.quantity * px
    return equity


def _exit_position(cash: float, pos: Position, exit_price: float) -> tuple[float, float]:
    """Spot close: return sale proceeds to cash and deduct exit fee."""
    gross_proceeds = pos.quantity * exit_price
    exit_fee = gross_proceeds * FEE_RATE
    new_cash = cash + gross_proceeds - exit_fee
    trade_pnl = (
        pos.quantity * (exit_price - pos.entry_price)
        - pos.quantity * pos.entry_price * FEE_RATE
        - exit_fee
    )
    return new_cash, trade_pnl


def simulate_portfolio(data, symbols, start, end, tp_mult):
    """Run one portfolio simulation with a fixed TP multiplier."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    # Master calendar = union of available daily candles across all selected assets.
    calendar = sorted(
        set().union(*[
            set(data[s]["df"].loc[start_ts:end_ts].index)
            for s in symbols
        ])
    )
    if len(calendar) < 20:
        return {
            "return": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "final_capital": INITIAL_CAPITAL,
        }

    cash = INITIAL_CAPITAL
    positions: dict[str, Position] = {}
    pending_entries: dict[str, float] = {}
    trades = []
    equity_points = []
    peak_equity = INITIAL_CAPITAL
    daily_start_equity = INITIAL_CAPITAL
    daily_date = None
    halted = False

    for ts in calendar:
        # Start-of-day mark-to-market for daily circuit breaker baseline.
        pre_equity = _portfolio_mark_to_market(cash, positions, data, ts)
        if daily_date != ts.date():
            daily_date = ts.date()
            daily_start_equity = pre_equity

        # 1) Execute entries scheduled by the previous candle's close.
        for symbol in list(pending_entries):
            if halted or len(positions) >= MAX_SIMULTANEOUS_POS:
                pending_entries.pop(symbol, None)
                continue
            if symbol in positions:
                pending_entries.pop(symbol, None)
                continue

            df = data[symbol]["df"]
            if ts not in df.index:
                continue

            current_equity = _portfolio_mark_to_market(cash, positions, data, ts)
            daily_loss = (
                (daily_start_equity - current_equity) / daily_start_equity
                if daily_start_equity > 0 else 0.0
            )
            if daily_loss >= CIRCUIT_BREAKER_PCT:
                pending_entries.clear()
                break

            atr_value = pending_entries.pop(symbol)
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue

            entry_price = float(df.loc[ts, "open"]) * (1.0 + ENTRY_SLIPPAGE_PCT)
            sl_distance = atr_value * STOP_LOSS_ATR_MULT
            tp_distance = atr_value * tp_mult

            risk_amount = current_equity * RISK_PER_TRADE
            raw_qty = risk_amount / sl_distance

            # Respect the configured portfolio allocation and the actually available cash.
            weight = PORTFOLIO_WEIGHTS.get(symbol, 1.0 / max(len(symbols), 1))
            allocation_cap = current_equity * weight
            max_notional = min(allocation_cap, cash / (1.0 + FEE_RATE))
            max_qty = max_notional / entry_price if entry_price > 0 else 0.0
            qty = min(raw_qty, max_qty)
            notional = qty * entry_price

            if qty <= 0 or notional < MIN_NOTIONAL:
                continue

            entry_fee = notional * FEE_RATE
            total_debit = notional + entry_fee
            if total_debit > cash:
                continue

            cash -= total_debit
            positions[symbol] = Position(
                symbol=symbol,
                entry_price=entry_price,
                quantity=qty,
                stop_loss=entry_price - sl_distance,
                take_profit=entry_price + tp_distance,
                entry_time=ts,
            )

        # 2) Intraday SL/TP. SL wins if both are touched on the same candle.
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
                exit_price = pos.stop_loss if hit_sl else pos.take_profit
                cash, pnl = _exit_position(cash, pos, exit_price)
                trades.append({
                    "symbol": symbol,
                    "entry_time": pos.entry_time,
                    "exit_time": ts,
                    "pnl": pnl,
                    "reason": reason,
                })
                positions.pop(symbol, None)

        # 3) Signal exits generated at yesterday's close execute at today's open.
        for symbol in list(positions):
            df = data[symbol]["df"]
            sig = data[symbol]["signals"]
            if ts not in df.index:
                continue
            loc = df.index.get_loc(ts)
            if isinstance(loc, slice) or loc == 0:
                continue
            prev_ts = df.index[loc - 1]
            if prev_ts < start_ts:
                continue
            if int(sig.loc[prev_ts]) != 1:
                pos = positions[symbol]
                exit_price = float(df.loc[ts, "open"])
                cash, pnl = _exit_position(cash, pos, exit_price)
                trades.append({
                    "symbol": symbol,
                    "entry_time": pos.entry_time,
                    "exit_time": ts,
                    "pnl": pnl,
                    "reason": "Sig",
                })
                positions.pop(symbol, None)

        current_equity = _portfolio_mark_to_market(cash, positions, data, ts)
        peak_equity = max(peak_equity, current_equity)
        drawdown = (
            (peak_equity - current_equity) / peak_equity
            if peak_equity > 0 else 0.0
        )
        daily_loss = (
            (daily_start_equity - current_equity) / daily_start_equity
            if daily_start_equity > 0 else 0.0
        )

        if drawdown >= MAX_DRAWDOWN_PCT:
            halted = True
            pending_entries.clear()
        if daily_loss >= CIRCUIT_BREAKER_PCT:
            pending_entries.clear()

        # 4) Today's close signal schedules tomorrow's open entry.
        if not halted and daily_loss < CIRCUIT_BREAKER_PCT:
            for symbol in symbols:
                if symbol in positions or symbol in pending_entries:
                    continue
                if len(positions) + len(pending_entries) >= MAX_SIMULTANEOUS_POS:
                    break

                df = data[symbol]["df"]
                sig = data[symbol]["signals"]
                atr = data[symbol]["atr"]
                if ts not in df.index:
                    continue
                loc = df.index.get_loc(ts)
                if isinstance(loc, slice) or loc + 1 >= len(df):
                    continue
                next_ts = df.index[loc + 1]
                if next_ts > end_ts:
                    continue
                av = float(atr.loc[ts]) if pd.notna(atr.loc[ts]) else np.nan
                if int(sig.loc[ts]) == 1 and np.isfinite(av) and av > 0:
                    pending_entries[symbol] = av

        equity_points.append((ts, current_equity))

    # Liquidate remaining positions at the last available close in the test period.
    for symbol in list(positions):
        df = data[symbol]["df"].loc[:end_ts]
        if df.empty:
            continue
        exit_ts = df.index[-1]
        exit_price = float(df["close"].iloc[-1])
        pos = positions[symbol]
        cash, pnl = _exit_position(cash, pos, exit_price)
        trades.append({
            "symbol": symbol,
            "entry_time": pos.entry_time,
            "exit_time": exit_ts,
            "pnl": pnl,
            "reason": "EOD",
        })
        positions.pop(symbol, None)

    final_equity = cash
    if equity_points:
        equity_points[-1] = (equity_points[-1][0], final_equity)
    eq = pd.Series(
        [v for _, v in equity_points],
        index=[t for t, _ in equity_points],
        dtype=float,
    )

    total_return = final_equity / INITIAL_CAPITAL - 1.0
    if not eq.empty:
        max_dd = ((eq - eq.cummax()) / eq.cummax()).min()
        daily_returns = eq.pct_change().dropna()
        sharpe = (
            daily_returns.mean() / daily_returns.std() * np.sqrt(365)
            if len(daily_returns) > 1 and daily_returns.std() > 0
            else 0.0
        )
    else:
        max_dd = 0.0
        sharpe = 0.0

    pnl_values = [t["pnl"] for t in trades]
    wins = [x for x in pnl_values if x > 0]
    losses = [x for x in pnl_values if x <= 0]
    profit_factor = (
        sum(wins) / abs(sum(losses))
        if losses and abs(sum(losses)) > 0
        else float("inf") if wins else 0.0
    )

    return {
        "return": total_return,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "final_capital": final_equity,
    }


def _choose_tp(data, symbols, train_start, train_end):
    """Select TP only from training data; ties prefer the less aggressive TP."""
    candidates = []
    for tp in CANDIDATE_TP:
        metrics = simulate_portfolio(data, symbols, train_start, train_end, tp)
        candidates.append((metrics["return"], metrics["sharpe"], -tp, tp, metrics))
    _, _, _, best_tp, best_metrics = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    return best_tp, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Krypton rolling walk-forward validation")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument(
        "--end",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SOLUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT"],
    )
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    parser.add_argument("--test-days", type=int, default=TEST_DAYS)
    parser.add_argument("--step-days", type=int, default=STEP_DAYS)
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    symbols = [s.upper() for s in args.symbols]

    unknown = [s for s in symbols if s not in PORTFOLIO_WEIGHTS]
    if unknown:
        raise ValueError(f"Símbolos sem peso configurado: {unknown}")

    data = _prepare_data(symbols, start, end)
    rows = []
    fold_start = start
    fold = 1

    while fold_start + timedelta(days=args.train_days + args.test_days - 1) <= end:
        train_start = fold_start
        train_end = train_start + timedelta(days=args.train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=args.test_days - 1)

        best_tp, train_metrics = _choose_tp(
            data,
            symbols,
            train_start,
            train_end,
        )
        test_metrics = simulate_portfolio(
            data,
            symbols,
            test_start,
            test_end,
            best_tp,
        )

        row = {
            "fold": fold,
            "train_start": train_start.date(),
            "train_end": train_end.date(),
            "test_start": test_start.date(),
            "test_end": test_end.date(),
            "selected_tp": best_tp,
            "train_return": train_metrics["return"],
            "train_sharpe": train_metrics["sharpe"],
            "train_max_drawdown": train_metrics["max_drawdown"],
            "test_return": test_metrics["return"],
            "test_sharpe": test_metrics["sharpe"],
            "test_max_drawdown": test_metrics["max_drawdown"],
            "test_trades": test_metrics["trades"],
            "test_win_rate": test_metrics["win_rate"],
            "test_profit_factor": test_metrics["profit_factor"],
            "test_final_capital": test_metrics["final_capital"],
        }
        rows.append(row)

        print(
            f"Fold {fold}: train {train_start.date()}→{train_end.date()} | "
            f"TP={best_tp:.1f} | test {test_start.date()}→{test_end.date()} | "
            f"ret={test_metrics['return']:+.2%} | "
            f"DD={test_metrics['max_drawdown']:.2%} | "
            f"WR={test_metrics['win_rate']:.1%} | trades={test_metrics['trades']}"
        )

        fold += 1
        fold_start += timedelta(days=args.step_days)

    out = pd.DataFrame(rows)
    out.to_csv("walk_forward_results.csv", index=False)

    print("\nWALK-FORWARD RESULTS")
    if out.empty:
        print("Nenhum fold completo no intervalo informado.")
        return
    print(out.to_string(index=False))

    # Compound non-overlapping OOS returns when test windows do not overlap.
    compounded = float(np.prod(1.0 + out["test_return"].values) - 1.0)
    profitable_folds = int((out["test_return"] > 0).sum())

    print("\nSUMMARY")
    print(f"Folds: {len(out)}")
    print(f"Folds lucrativos: {profitable_folds}/{len(out)} ({profitable_folds/len(out):.1%})")
    print(f"Retorno OOS médio/fold: {out['test_return'].mean():+.2%}")
    print(f"Retorno OOS mediano/fold: {out['test_return'].median():+.2%}")
    print(f"Retorno OOS composto: {compounded:+.2%}")
    print(f"Sharpe OOS médio: {out['test_sharpe'].mean():.3f}")
    print(f"Pior drawdown OOS: {out['test_max_drawdown'].min():.2%}")
    print(f"Win rate OOS agregado (média dos folds): {out['test_win_rate'].mean():.1%}")
    print(f"Trades OOS: {int(out['test_trades'].sum())}")
    print("Resultados salvos em walk_forward_results.csv")


if __name__ == "__main__":
    main()
