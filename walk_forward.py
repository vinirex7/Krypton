"""Walk-forward validation for Krypton Spot strategy.

Realistic rolling validation on Binance Spot daily OHLCV.

Properties
----------
- Spot LONG-only.
- Signal on candle close, execution on next candle open.
- SL/TP use low/high; SL wins if both are touched in the same candle.
- Entry/exit fees, slippage and adverse gaps included.
- Risk is configurable and calculated from TOTAL portfolio equity.
- Notional is capped by available cash and normalized portfolio allocation.
- Portfolio-level simultaneous-position cap, circuit breaker and max-DD halt.
- Optional BTC 200-day SMA regime filter blocks new entries in risk-off regimes.
- TP is selected only on each training window and frozen for the next OOS window.

Live and historical validation use the same USDT pairs.
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
    ENTRY_SLIPPAGE_PCT,
    EXIT_SLIPPAGE_PCT,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MAX_DRAWDOWN_PCT,
    MAX_SIMULTANEOUS_POS,
    RISK_PER_TRADE,
    REGIME_FILTER,
    RSI_HIGH,
    RSI_LOW,
    RSI_PERIOD,
    STOP_LOSS_ATR_MULT,
    SUPERTREND_MULTIPLIER,
    SUPERTREND_PERIOD,
    TRADING_PAIRS,
)
from indicators import compute_atr, compute_signals

INITIAL_CAPITAL = 10_000.0
DEFAULT_CANDIDATE_TP = [3.0, 4.0, 4.5, 5.0]
TRAIN_DAYS = 365
TEST_DAYS = 180
STEP_DAYS = 180
MIN_NOTIONAL = 10.0

BASE_WEIGHTS = dict(TRADING_PAIRS)


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


def normalized_weights(symbols: list[str]) -> dict[str, float]:
    """Normalize configured live weights across the selected historical symbols."""
    raw = {s: BASE_WEIGHTS[s] for s in symbols}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("Soma dos pesos do portfolio precisa ser positiva.")
    return {s: w / total for s, w in raw.items()}


def _prepare_data(symbols: list[str], start: datetime, end: datetime):
    """Download market data once and precompute indicators over warm-up history."""
    data = {}
    warmup_start = start - timedelta(days=400)
    required = list(dict.fromkeys(symbols + (["BTCUSDT"] if "BTCUSDT" not in symbols else [])))

    for symbol in required:
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
            "sma200": df["close"].rolling(200, min_periods=200).mean(),
            "source": source,
        }
    return data


def _portfolio_mark_to_market(
    cash: float, positions: dict[str, Position], data, ts, price_field: str = "close"
) -> float:
    equity = cash
    for symbol, pos in positions.items():
        df = data[symbol]["df"]
        if ts in df.index:
            px = float(df.loc[ts, price_field])
        else:
            eligible = df.loc[:ts]
            px = pos.entry_price if eligible.empty else float(eligible["close"].iloc[-1])
        equity += pos.quantity * px
    return equity


def _exit_position(cash: float, pos: Position, exit_price: float) -> tuple[float, float]:
    gross_proceeds = pos.quantity * exit_price
    exit_fee = gross_proceeds * FEE_RATE
    new_cash = cash + gross_proceeds - exit_fee
    trade_pnl = (
        pos.quantity * (exit_price - pos.entry_price)
        - pos.quantity * pos.entry_price * FEE_RATE
        - exit_fee
    )
    return new_cash, trade_pnl


def _risk_on(data, ts) -> bool:
    """Risk-on regime: BTC daily close is above its 200-day SMA."""
    btc = data["BTCUSDT"]
    df = btc["df"]
    eligible = df.loc[:ts]
    if eligible.empty:
        return False
    btc_ts = eligible.index[-1]
    sma = btc["sma200"].loc[btc_ts]
    return pd.notna(sma) and float(df.loc[btc_ts, "close"]) > float(sma)


def simulate_portfolio(
    data,
    symbols,
    start,
    end,
    tp_mult,
    risk_per_trade=RISK_PER_TRADE,
    regime_filter=False,
    weights=None,
):
    """Run one portfolio simulation with fixed parameters."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    weights = weights or normalized_weights(symbols)
    calendar = sorted(set().union(*[set(data[s]["df"].loc[start_ts:end_ts].index) for s in symbols]))
    if len(calendar) < 20:
        return {"return": 0.0, "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "max_drawdown": 0.0, "sharpe": 0.0, "final_capital": INITIAL_CAPITAL,
            "halted": False, "trade_log": pd.DataFrame(), "equity_curve": pd.Series(dtype=float)}

    cash = INITIAL_CAPITAL
    positions: dict[str, Position] = {}
    pending_entries: dict[str, float] = {}
    trades: list[dict] = []
    equity_points = []
    peak_equity = INITIAL_CAPITAL
    daily_start_equity = INITIAL_CAPITAL
    daily_date = None
    halted = False

    for ts in calendar:
        pre_equity = _portfolio_mark_to_market(cash, positions, data, ts, price_field="open")
        if daily_date != ts.date():
            daily_date = ts.date()
            daily_start_equity = pre_equity

        # OPEN 1/2: exits generated at the previous close happen before intraday barriers.
        for symbol in list(positions):
            df = data[symbol]["df"]
            sig = data[symbol]["signals"]
            if ts not in df.index:
                continue
            loc = df.index.get_loc(ts)
            if isinstance(loc, slice) or loc == 0:
                continue
            prev_ts = df.index[loc - 1]
            if prev_ts >= start_ts and int(sig.loc[prev_ts]) != 1:
                pos = positions[symbol]
                exit_price = float(df.loc[ts, "open"]) * (1.0 - EXIT_SLIPPAGE_PCT)
                cash, pnl = _exit_position(cash, pos, exit_price)
                trades.append({"symbol": symbol, "entry_time": pos.entry_time,
                               "exit_time": ts, "pnl": pnl,
                               "portfolio_return": pnl / pos.equity_at_entry,
                               "reason": "Sig"})
                positions.pop(symbol, None)

        # OPEN 2/2: execute entries generated at the previous close.
        for symbol in list(pending_entries):
            if halted or len(positions) >= MAX_SIMULTANEOUS_POS or symbol in positions:
                pending_entries.pop(symbol, None)
                continue
            df = data[symbol]["df"]
            if ts not in df.index:
                continue
            current_equity = _portfolio_mark_to_market(cash, positions, data, ts, price_field="open")
            daily_loss = ((daily_start_equity - current_equity) / daily_start_equity
                          if daily_start_equity > 0 else 0.0)
            if daily_loss >= CIRCUIT_BREAKER_PCT:
                pending_entries.clear()
                break

            atr_value = pending_entries.pop(symbol)
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue
            entry_price = float(df.loc[ts, "open"]) * (1.0 + ENTRY_SLIPPAGE_PCT)
            sl_distance = atr_value * STOP_LOSS_ATR_MULT
            tp_distance = atr_value * tp_mult
            risk_amount = current_equity * risk_per_trade
            raw_qty = risk_amount / sl_distance
            allocation_cap = current_equity * weights[symbol]
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
                entry_fee=entry_fee,
                equity_at_entry=current_equity,
            )

        # Intraday exits. Conservative ordering: SL before TP.
        for symbol in list(positions):
            df = data[symbol]["df"]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            pos = positions[symbol]
            open_price = float(row["open"])
            if open_price <= pos.stop_loss:
                reason, exit_price = "SL_GAP", open_price * (1.0 - EXIT_SLIPPAGE_PCT)
            elif open_price >= pos.take_profit:
                reason, exit_price = "TP_GAP", pos.take_profit
            elif float(row["low"]) <= pos.stop_loss:
                reason, exit_price = "SL", pos.stop_loss * (1.0 - EXIT_SLIPPAGE_PCT)
            elif float(row["high"]) >= pos.take_profit:
                reason, exit_price = "TP", pos.take_profit
            else:
                continue
            if reason:
                cash, pnl = _exit_position(cash, pos, exit_price)
                trades.append({"symbol": symbol, "entry_time": pos.entry_time,
                               "exit_time": ts, "pnl": pnl,
                               "portfolio_return": pnl / pos.equity_at_entry,
                               "reason": reason})
                positions.pop(symbol, None)

        current_equity = _portfolio_mark_to_market(cash, positions, data, ts)
        peak_equity = max(peak_equity, current_equity)
        drawdown = ((peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0)
        daily_loss = ((daily_start_equity - current_equity) / daily_start_equity
                      if daily_start_equity > 0 else 0.0)

        # 20% is a trigger, not a guaranteed floor: an adverse candle can gap through it.
        if drawdown >= MAX_DRAWDOWN_PCT:
            halted = True
            pending_entries.clear()
        if daily_loss >= CIRCUIT_BREAKER_PCT:
            pending_entries.clear()

        # Current close schedules next open. Optional regime filter applies only to NEW entries.
        risk_on = (not regime_filter) or _risk_on(data, ts)
        if not halted and daily_loss < CIRCUIT_BREAKER_PCT and risk_on:
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

    # Liquidate remaining positions at the final available close.
    for symbol in list(positions):
        df = data[symbol]["df"].loc[:end_ts]
        if df.empty:
            continue
        exit_ts = df.index[-1]
        exit_price = float(df["close"].iloc[-1]) * (1.0 - EXIT_SLIPPAGE_PCT)
        pos = positions[symbol]
        cash, pnl = _exit_position(cash, pos, exit_price)
        trades.append({"symbol": symbol, "entry_time": pos.entry_time,
                       "exit_time": exit_ts, "pnl": pnl,
                       "portfolio_return": pnl / pos.equity_at_entry,
                       "reason": "EOD"})
        positions.pop(symbol, None)

    final_equity = cash
    if equity_points:
        equity_points[-1] = (equity_points[-1][0], final_equity)
    eq = pd.Series([v for _, v in equity_points], index=[t for t, _ in equity_points], dtype=float)
    total_return = final_equity / INITIAL_CAPITAL - 1.0
    if not eq.empty:
        max_dd = ((eq - eq.cummax()) / eq.cummax()).min()
        daily_returns = eq.pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(365)
                  if len(daily_returns) > 1 and daily_returns.std() > 0 else 0.0)
    else:
        max_dd = 0.0
        sharpe = 0.0

    pnl_values = [t["pnl"] for t in trades]
    wins = [x for x in pnl_values if x > 0]
    losses = [x for x in pnl_values if x <= 0]
    profit_factor = (sum(wins) / abs(sum(losses)) if losses and abs(sum(losses)) > 0
                     else float("inf") if wins else 0.0)
    return {
        "return": total_return,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "final_capital": final_equity,
        "halted": halted,
        "trade_log": pd.DataFrame(trades),
        "equity_curve": eq,
    }


def _choose_tp(data, symbols, train_start, train_end, candidate_tp,
               risk_per_trade, regime_filter, weights):
    """Select TP only from training data; ties prefer the less aggressive TP."""
    candidates = []
    for tp in candidate_tp:
        metrics = simulate_portfolio(
            data, symbols, train_start, train_end, tp,
            risk_per_trade=risk_per_trade,
            regime_filter=regime_filter,
            weights=weights,
        )
        candidates.append((metrics["return"], metrics["sharpe"], -tp, tp, metrics))
    _, _, _, best_tp, best_metrics = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    return best_tp, best_metrics


def run_walk_forward(data, symbols, start, end, train_days=TRAIN_DAYS,
                     test_days=TEST_DAYS, step_days=STEP_DAYS,
                     candidate_tp=None, risk_per_trade=RISK_PER_TRADE,
                     regime_filter=REGIME_FILTER, label="baseline"):
    candidate_tp = candidate_tp or DEFAULT_CANDIDATE_TP
    weights = normalized_weights(symbols)
    rows = []
    fold_start = start
    fold = 1
    while fold_start + timedelta(days=train_days + test_days - 1) <= end:
        train_start = fold_start
        train_end = train_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)
        best_tp, train_metrics = _choose_tp(
            data, symbols, train_start, train_end, candidate_tp,
            risk_per_trade, regime_filter, weights,
        )
        test_metrics = simulate_portfolio(
            data, symbols, test_start, test_end, best_tp,
            risk_per_trade=risk_per_trade,
            regime_filter=regime_filter,
            weights=weights,
        )
        rows.append({
            "variant": label,
            "fold": fold,
            "train_start": train_start.date(), "train_end": train_end.date(),
            "test_start": test_start.date(), "test_end": test_end.date(),
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
            "test_halted": test_metrics["halted"],
        })
        print(
            f"[{label}] Fold {fold}: TP={best_tp:.1f} | "
            f"OOS={test_metrics['return']:+.2%} | DD={test_metrics['max_drawdown']:.2%} | "
            f"Sharpe={test_metrics['sharpe']:.2f} | WR={test_metrics['win_rate']:.1%} | "
            f"trades={test_metrics['trades']}"
        )
        fold += 1
        fold_start += timedelta(days=step_days)
    return pd.DataFrame(rows)


def summarize(out: pd.DataFrame) -> dict:
    if out.empty:
        return {}
    compounded = float(np.prod(1.0 + out["test_return"].values) - 1.0)
    profitable = int((out["test_return"] > 0).sum())
    return {
        "folds": len(out),
        "profitable_folds": profitable,
        "profitable_pct": profitable / len(out),
        "mean_return": out["test_return"].mean(),
        "median_return": out["test_return"].median(),
        "compounded_return": compounded,
        "mean_sharpe": out["test_sharpe"].mean(),
        "worst_drawdown": out["test_max_drawdown"].min(),
        "mean_win_rate": out["test_win_rate"].mean(),
        "trades": int(out["test_trades"].sum()),
        "halted_folds": int(out["test_halted"].sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Krypton rolling walk-forward validation")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--symbols", nargs="+", default=list(TRADING_PAIRS))
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    parser.add_argument("--test-days", type=int, default=TEST_DAYS)
    parser.add_argument("--step-days", type=int, default=STEP_DAYS)
    parser.add_argument("--risk", type=float, default=RISK_PER_TRADE, help="Risco por trade; ex. 0.005 = 0,5%")
    parser.add_argument("--candidate-tp", nargs="+", type=float, default=DEFAULT_CANDIDATE_TP)
    parser.add_argument("--regime-filter", action=argparse.BooleanOptionalAction, default=REGIME_FILTER,
                        help="Só abre novas posições quando BTC > SMA200")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    symbols = [s.upper() for s in args.symbols]
    unknown = [s for s in symbols if s not in BASE_WEIGHTS]
    if unknown:
        raise ValueError(f"Símbolos sem peso configurado: {unknown}")
    if args.risk <= 0 or args.risk > 0.05:
        raise ValueError("--risk deve estar entre 0 e 0.05.")

    data = _prepare_data(symbols, start, end)
    out = run_walk_forward(
        data, symbols, start, end,
        train_days=args.train_days, test_days=args.test_days, step_days=args.step_days,
        candidate_tp=args.candidate_tp, risk_per_trade=args.risk,
        regime_filter=args.regime_filter,
    )
    out.to_csv("walk_forward_results.csv", index=False)
    print("\nWALK-FORWARD RESULTS")
    print(out.to_string(index=False) if not out.empty else "Nenhum fold completo.")
    s = summarize(out)
    if s:
        print("\nSUMMARY")
        print(f"Folds: {s['folds']}")
        print(f"Folds lucrativos: {s['profitable_folds']}/{s['folds']} ({s['profitable_pct']:.1%})")
        print(f"Retorno OOS médio/fold: {s['mean_return']:+.2%}")
        print(f"Retorno OOS mediano/fold: {s['median_return']:+.2%}")
        print(f"Produto das janelas OOS (não contínuo): {s['compounded_return']:+.2%}")
        print(f"Sharpe OOS médio: {s['mean_sharpe']:.3f}")
        print(f"Pior drawdown OOS: {s['worst_drawdown']:.2%}")
        print(f"Win rate OOS médio: {s['mean_win_rate']:.1%}")
        print(f"Trades OOS: {s['trades']}")
        print(f"Folds que acionaram halt de DD: {s['halted_folds']}")
        print("Resultados salvos em walk_forward_results.csv")


if __name__ == "__main__":
    main()
