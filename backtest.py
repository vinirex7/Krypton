# backtest.py — Backtest Spot (USDT) sem look-ahead
# Krypton TradeBot | Estratégia: Supertrend + RSI + MACD Filter
#
# Regras:
# - sinal no CLOSE do candle i -> entrada no OPEN de i+1;
# - SL/TP avaliados por LOW/HIGH; se ambos forem tocados no mesmo candle, SL vence;
# - apenas LONG/spot (sem short);
# - sizing = 1% do capital TOTAL por trade;
# - respeita MAX_SIMULTANEOUS_POS, CIRCUIT_BREAKER_PCT, MAX_DRAWDOWN_PCT,
#   STOP_LOSS_ATR_MULT e TAKE_PROFIT_ATR_MULT;
# - ATR calculado uma vez no dataset completo, com warm-up.

import argparse
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

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
    RSI_HIGH,
    RSI_LOW,
    RSI_PERIOD,
    STOP_LOSS_ATR_MULT,
    SUPERTREND_MULTIPLIER,
    SUPERTREND_PERIOD,
    TAKE_PROFIT_ATR_MULT,
)
from indicators import compute_atr, compute_signals

BINANCE_GLOBAL_URL = "https://api.binance.com/api/v3/klines"
BINANCE_US_URL = "https://api.binance.us/api/v3/klines"
BINANCE_US_SYMBOL_MAP = {"SOLUSDT": "SOLUSD", "BTCUSDT": "BTCUSD", "ETHUSDT": "ETHUSD", "BNBUSDT": "BNBUSD"}
YAHOO_SYMBOL_MAP = {"SOLUSDT": "SOL-USD", "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "BNBUSDT": "BNB-USD"}
WARMUP_DAYS = 300
INITIAL_CAPITAL = 10_000.0


def _date_to_ms(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _klines_to_df(klines: list) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame()
    df = pd.DataFrame(
        klines,
        columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def get_ohlcv_binance(symbol: str, start_str: str, end_str: str | None, base_url: str, symbol_map: dict | None = None) -> pd.DataFrame:
    symbol_api = symbol_map.get(symbol.upper(), symbol.upper()) if symbol_map else symbol.upper()
    start_ts = _date_to_ms(start_str)
    end_ts = _date_to_ms(end_str) if end_str else None
    all_klines = []
    current_ts = start_ts
    while True:
        params = {"symbol": symbol_api, "interval": "1d", "startTime": current_ts, "limit": 1000}
        if end_ts:
            params["endTime"] = end_ts
        try:
            response = requests.get(base_url, params=params, timeout=15)
            if response.status_code != 200:
                break
            klines = response.json()
        except Exception:
            time.sleep(2)
            break
        if not klines or isinstance(klines, dict):
            break
        all_klines.extend(klines)
        if len(klines) < 1000:
            break
        current_ts = klines[-1][0] + 86_400_000
        if end_ts and current_ts >= end_ts:
            break
        time.sleep(0.2)
    return _klines_to_df(all_klines)


def get_ohlcv_yahoo(symbol: str, start_str: str, end_str: str | None = None) -> pd.DataFrame:
    yf_symbol = YAHOO_SYMBOL_MAP.get(symbol.upper())
    if not yf_symbol:
        return pd.DataFrame()
    try:
        df = yf.download(yf_symbol, start=start_str, end=end_str, interval="1d", auto_adjust=False, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    required = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index, utc=True)
    return df[required].dropna()


def get_ohlcv(symbol: str, start_str: str, end_str: str | None = None, allow_proxy_data: bool = False) -> tuple[pd.DataFrame, str]:
    """Carrega o mercado exato da Binance; proxies USD são opt-in e claramente rotulados."""
    sources = [("Binance Global Spot", lambda: get_ohlcv_binance(symbol, start_str, end_str, BINANCE_GLOBAL_URL))]
    if allow_proxy_data:
        sources += [
            ("PROXY Binance US USD", lambda: get_ohlcv_binance(symbol, start_str, end_str, BINANCE_US_URL, BINANCE_US_SYMBOL_MAP)),
            ("PROXY Yahoo USD", lambda: get_ohlcv_yahoo(symbol, start_str, end_str)),
        ]
    best_df = pd.DataFrame()
    best_source = "nenhuma fonte"
    for source_name, loader in sources:
        print(f"\n  Tentando {source_name}...", end=" ", flush=True)
        df = loader()
        if len(df) > len(best_df):
            best_df, best_source = df, source_name
        if len(df) >= 50:
            print(f"✓ {len(df)} candles")
            return df, source_name
        print(f"{len(df)} candles")
    return best_df, best_source


def _buy_execution_price(open_price: float) -> float:
    return open_price * (1.0 + ENTRY_SLIPPAGE_PCT)


def _sell_execution_price(price: float) -> float:
    return price * (1.0 - EXIT_SLIPPAGE_PCT)


def _barrier_exit(row: pd.Series, position: dict) -> tuple[float, str] | None:
    """Modela gaps e mantém prioridade conservadora do SL quando ambos são tocados."""
    open_price = float(row["open"])
    if open_price <= position["sl"]:
        return _sell_execution_price(open_price), "SL_GAP"
    if open_price >= position["tp"]:
        return position["tp"], "TP_GAP"
    hit_sl = float(row["low"]) <= position["sl"]
    hit_tp = float(row["high"]) >= position["tp"]
    if hit_sl:
        return _sell_execution_price(position["sl"]), "SL"
    if hit_tp:
        return position["tp"], "TP"
    return None


def _close_position(cash: float, position: dict, exit_price: float) -> tuple[float, float]:
    proceeds = position["quantity"] * exit_price
    exit_fee = proceeds * FEE_RATE
    pnl = proceeds - exit_fee - position["cost_basis"]
    return cash + proceeds - exit_fee, pnl


def run_backtest(symbol: str, start: str, end: str | None = None) -> dict:
    print(f"\n{'=' * 60}")
    print(f"KRYPTON BACKTEST SPOT: {symbol} (quote USDT)")
    print(f"Período: {start} → {end or 'hoje'}")
    print(f"SL={STOP_LOSS_ATR_MULT}× ATR | TP={TAKE_PROFIT_ATR_MULT}× ATR | risco={RISK_PER_TRADE:.1%}")
    print(f"{'=' * 60}")

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    warmup_str = (start_dt - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    df_full, data_source = get_ohlcv(symbol, warmup_str, end)
    if len(df_full) < 50:
        print(f"\n❌ Dados insuficientes ({len(df_full)} candles).")
        return {}

    df_full = df_full.sort_index()
    signals_full = compute_signals(
        df_full,
        st_period=SUPERTREND_PERIOD,
        st_mult=SUPERTREND_MULTIPLIER,
        rsi_period=RSI_PERIOD,
        rsi_low=RSI_LOW,
        rsi_high=RSI_HIGH,
        macd_fast=MACD_FAST,
        macd_slow=MACD_SLOW,
        macd_sig=MACD_SIGNAL,
    )
    # ATR é calculado UMA vez sobre o dataset completo, antes do slice do período.
    atr_full = compute_atr(df_full["high"], df_full["low"], df_full["close"])
    start_ts = pd.Timestamp(start, tz="UTC")
    df = df_full.loc[start_ts:].copy()
    signals = signals_full.loc[df.index]
    atr = atr_full.loc[df.index]
    if len(df) < 10:
        print("❌ Período de backtest muito curto.")
        return {}

    cash = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    equity = []
    trades = []
    position = None
    halted = False
    daily_start_equity = INITIAL_CAPITAL
    daily_date = None
    entry_pending = None

    def reset_day(day):
        nonlocal daily_start_equity, daily_date
        if daily_date != day:
            daily_date = day
            daily_start_equity = current_equity

    def circuit_breaker_active():
        if daily_start_equity <= 0:
            return False
        return (daily_start_equity - current_equity) / daily_start_equity >= CIRCUIT_BREAKER_PCT

    for i in range(len(df)):
        row = df.iloc[i]
        current_equity = cash + (position["quantity"] * float(row["open"]) if position else 0.0)
        reset_day(df.index[i].date())

        # Eventos no OPEN: primeiro sai pelo sinal conhecido desde o close anterior.
        if position is not None and i > 0 and int(signals.iloc[i - 1]) != 1:
            exit_price = _sell_execution_price(float(row["open"]))
            cash, pnl = _close_position(cash, position, exit_price)
            trades.append({"entry_time": position["entry_time"], "exit_time": df.index[i], "pnl": pnl, "exit_reason": "Sig"})
            position = None

        current_equity = cash + (position["quantity"] * float(row["open"]) if position else 0.0)

        # Depois executa entradas geradas no close anterior.
        if entry_pending is not None and position is None and not halted and not circuit_breaker_active():
            entry_atr = entry_pending["atr"]
            entry = _buy_execution_price(float(row["open"]))
            sl_distance = entry_atr * STOP_LOSS_ATR_MULT
            tp_distance = entry_atr * TAKE_PROFIT_ATR_MULT
            if sl_distance > 0 and pd.notna(entry_atr):
                risk_amount = current_equity * RISK_PER_TRADE
                raw_qty = risk_amount / sl_distance
                max_qty = cash / (entry * (1 + FEE_RATE))
                qty = min(raw_qty, max_qty)
                notional = qty * entry
                if qty > 0 and notional >= 10:
                    entry_fee = notional * FEE_RATE
                    cost_basis = notional + entry_fee
                    cash -= cost_basis
                    position = {
                        "entry_price": entry,
                        "quantity": qty,
                        "sl": entry - sl_distance,
                        "tp": entry + tp_distance,
                        "entry_time": df.index[i],
                        "cost_basis": cost_basis,
                    }
            entry_pending = None

        # Só depois do OPEN processa a faixa intradiária.
        if position is not None:
            barrier = _barrier_exit(row, position)
            if barrier:
                exit_price, reason = barrier
                cash, pnl = _close_position(cash, position, exit_price)
                trades.append({"entry_time": position["entry_time"], "exit_time": df.index[i], "pnl": pnl, "exit_reason": reason})
                position = None

        current_equity = cash + (position["quantity"] * float(row["close"]) if position else 0.0)
        peak = max(peak, current_equity)
        if peak > 0 and (peak - current_equity) / peak >= MAX_DRAWDOWN_PCT:
            halted = True
            entry_pending = None

        # O sinal atual só pode abrir no próximo candle; nunca no mesmo close.
        if position is None and entry_pending is None and not halted and not circuit_breaker_active() and int(signals.iloc[i]) == 1 and pd.notna(atr.iloc[i]) and float(atr.iloc[i]) > 0 and i + 1 < len(df):
            entry_pending = {"atr": float(atr.iloc[i]), "signal_time": df.index[i]}

        equity.append(current_equity)

    if position is not None:
        exit_price = _sell_execution_price(float(df["close"].iloc[-1]))
        cash, pnl = _close_position(cash, position, exit_price)
        trades.append({"entry_time": position["entry_time"], "exit_time": df.index[-1], "pnl": pnl, "exit_reason": "EOD"})
        equity[-1] = cash

    eq = pd.Series(equity, index=df.index[:len(equity)])
    capital = cash
    ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    rets = eq.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(365) if rets.std() > 0 else 0
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    pf = sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses)) if losses else float("inf")
    wr = len(wins) / len(trades) if trades else 0
    bh_ret = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
    exit_reasons = {}
    for trade in trades:
        exit_reasons[trade["exit_reason"]] = exit_reasons.get(trade["exit_reason"], 0) + 1

    print(f"\n{'─' * 60}")
    print(f"{'MÉTRICA':<28} {'BOT':>10} {'BUY & HOLD':>12}")
    print(f"{'─' * 60}")
    print(f"{'Retorno Total':<28} {ret:>+9.1%} {bh_ret:>+11.1%}")
    print(f"{'Sharpe Ratio':<28} {sharpe:>10.3f} {'—':>12}")
    print(f"{'Max Drawdown':<28} {dd:>+9.1%} {'—':>12}")
    print(f"{'Win Rate':<28} {wr:>9.1%} {'—':>12}")
    print(f"{'Profit Factor':<28} {pf:>10.3f} {'—':>12}")
    print(f"{'Nº de Trades':<28} {len(trades):>10} {'—':>12}")
    print(f"{'Alpha vs B&H':<28} {ret - bh_ret:>+9.1%} {'—':>12}")
    print(f"{'Capital Final':<28} ${capital:>9,.2f} {'—':>12}")
    print(f"{'─' * 60}")
    print(f"Saídas: {exit_reasons}")
    print(f"{'=' * 60}\n")

    return {
        "symbol": symbol,
        "quote_asset": "USDT",
        "start": start,
        "end": end or str(date.today()),
        "data_source": data_source,
        "n_candles": len(df),
        "n_trades": len(trades),
        "return_total": ret,
        "sharpe_ratio": sharpe,
        "max_drawdown": dd,
        "win_rate": wr,
        "profit_factor": pf,
        "final_capital": capital,
        "bh_return": bh_ret,
        "alpha_vs_bh": ret - bh_ret,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Krypton TradeBot — Backtest Spot (USDT)")
    parser.add_argument("--symbol", default="SOLUSDT", choices=["SOLUSDT", "BTCUSDT", "BNBUSDT"])
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    if not run_backtest(args.symbol, args.start, args.end):
        raise SystemExit(2)
