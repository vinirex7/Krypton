# backtest.py — Backtesting Standalone com múltiplas fontes de dados
# Krypton TradeBot | Estratégia: Supertrend + RSI + MACD Filter
#
# Baixa dados históricos primeiro pela Binance Global, depois Binance US e,
# se necessário, usa Yahoo Finance como fallback. A estratégia do bot não foi alterada.
#
# Uso:
#   python backtest.py --symbol SOLUSDT --start 2022-01-01 --end 2026-06-01
#   python backtest.py --symbol BTCUSDT --start 2024-01-01
#   python backtest.py  # SOLUSDT desde 2022-01-01

import argparse
import time
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    FEE_RATE,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MAX_DRAWDOWN_PCT,
    RISK_PER_TRADE,
    RSI_HIGH,
    RSI_LOW,
    RSI_PERIOD,
    SUPERTREND_MULTIPLIER,
    SUPERTREND_PERIOD,
)
from indicators import compute_atr, compute_signals


BINANCE_GLOBAL_URL = "https://api.binance.com/api/v3/klines"
BINANCE_US_URL = "https://api.binance.us/api/v3/klines"

# Binance US usa pares USD em alguns ativos. Binance Global usa USDT.
BINANCE_US_SYMBOL_MAP = {
    "SOLUSDT": "SOLUSD",
    "BTCUSDT": "BTCUSD",
    "ETHUSDT": "ETHUSD",
    "BNBUSDT": "BNBUSD",
}

YAHOO_SYMBOL_MAP = {
    "SOLUSDT": "SOL-USD",
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "BNBUSDT": "BNB-USD",
}


def _date_to_ms(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d").timestamp() * 1000)


def _klines_to_df(klines: list) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df[["open", "high", "low", "close", "volume"]]


def get_ohlcv_binance(symbol: str, start_str: str, end_str: str | None, base_url: str, symbol_map: dict | None = None) -> pd.DataFrame:
    """
    Baixa OHLCV pela API pública da Binance/Binance US em lotes de 1000 candles.
    """
    symbol_api = symbol_map.get(symbol.upper(), symbol.upper()) if symbol_map else symbol.upper()
    start_ts = _date_to_ms(start_str)
    end_ts = _date_to_ms(end_str) if end_str else None

    all_klines = []
    current_ts = start_ts

    while True:
        params = {
            "symbol": symbol_api,
            "interval": "1d",
            "startTime": current_ts,
            "limit": 1000,
        }
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
    """
    Fallback via Yahoo Finance para períodos em que Binance US não possui candles.
    """
    yf_symbol = YAHOO_SYMBOL_MAP.get(symbol.upper())
    if not yf_symbol:
        return pd.DataFrame()

    try:
        df = yf.download(
            yf_symbol,
            start=start_str,
            end=end_str,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    return df[required_cols].dropna()


def get_ohlcv(symbol: str, start_str: str, end_str: str | None = None) -> tuple[pd.DataFrame, str]:
    """
    Busca dados em múltiplas fontes sem alterar a estratégia do backtest.
    """
    sources = [
        ("Binance Global", lambda: get_ohlcv_binance(symbol, start_str, end_str, BINANCE_GLOBAL_URL)),
        ("Binance US", lambda: get_ohlcv_binance(symbol, start_str, end_str, BINANCE_US_URL, BINANCE_US_SYMBOL_MAP)),
        ("Yahoo Finance", lambda: get_ohlcv_yahoo(symbol, start_str, end_str)),
    ]

    best_df = pd.DataFrame()
    best_source = "nenhuma fonte"

    for source_name, loader in sources:
        print(f"\n  Tentando {source_name}...", end=" ", flush=True)
        df = loader()
        if len(df) > len(best_df):
            best_df = df
            best_source = source_name

        if len(df) >= 50:
            print(f"✓ {len(df)} candles")
            return df, source_name

        print(f"{len(df)} candles")

    return best_df, best_source


def run_backtest(symbol: str, start: str, end: str | None = None) -> dict:
    """
    Executa backtest da estratégia Supertrend + RSI + MACD.

    Simula execução diária com:
      - Position sizing ATR-based (1% de risco por trade)
      - Stop Loss: 2× ATR(14)
      - Take Profit: 3× ATR(14)  |  R:R = 1,5:1
      - Taxas: 0,1% por lado (Binance spot)
      - Halt automático ao atingir max drawdown (-20%)
    """
    print(f"\n{'='*60}")
    print(f"KRYPTON BACKTEST: {symbol}")
    print(f"Período: {start} → {end or 'hoje'}")
    print(f"{'='*60}")
    print("Baixando dados históricos...", flush=True)

    df, data_source = get_ohlcv(symbol, start, end)

    if len(df) < 50:
        print(f"\n❌ Dados insuficientes ({len(df)} candles). Verifique o símbolo e as datas.")
        return {}

    print(f"\nFonte usada: {data_source}")
    print(f"✓ {len(df)} candles carregados.")
    print(f"  {df.index[0].date()} → {df.index[-1].date()}")

    # ─── Geração de Sinais ────────────────────────────────────────────────────
    signals = compute_signals(
        df,
        st_period  = SUPERTREND_PERIOD,
        st_mult    = SUPERTREND_MULTIPLIER,
        rsi_period = RSI_PERIOD,
        rsi_low    = RSI_LOW,
        rsi_high   = RSI_HIGH,
        macd_fast  = MACD_FAST,
        macd_slow  = MACD_SLOW,
        macd_sig   = MACD_SIGNAL,
    )

    # ─── Simulação de Trading ─────────────────────────────────────────────────
    capital  = 10_000.0
    peak     = capital
    equity   = [capital]
    trades   = []
    pos      = 0
    entry    = 0.0
    pos_size = 0.0

    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        sig   = signals.iloc[i]
        atr_v = compute_atr(df["high"], df["low"], df["close"]).iloc[i]

        # Gerenciar posição aberta
        if pos != 0:
            sl      = entry - 2 * atr_v if pos ==  1 else entry + 2 * atr_v
            tp      = entry + 3 * atr_v if pos ==  1 else entry - 3 * atr_v
            hit_sl  = (pos ==  1 and price <= sl) or (pos == -1 and price >= sl)
            hit_tp  = (pos ==  1 and price >= tp) or (pos == -1 and price <= tp)
            exit_sg = (pos != sig and sig != 0)

            if hit_sl or hit_tp or exit_sg:
                fee  = pos_size * price * FEE_RATE
                pnl  = (pos_size * (price - entry) if pos == 1
                        else pos_size * (entry - price)) - fee
                capital += pnl
                trades.append({
                    "pnl"        : pnl,
                    "exit_reason": "SL" if hit_sl else "TP" if hit_tp else "Sig",
                })
                pos  = 0
                peak = max(peak, capital)

        # Halt por max drawdown
        if peak > 0 and (peak - capital) / peak >= MAX_DRAWDOWN_PCT:
            equity.append(capital)
            continue

        # Abrir nova posição
        if pos == 0 and sig != 0 and pd.notna(atr_v) and atr_v > 0:
            entry    = price * (1.0005 if sig == 1 else 0.9995)
            pos_size = (capital * RISK_PER_TRADE) / (atr_v * 2)
            capital -= pos_size * entry * FEE_RATE
            pos      = int(sig)

        equity.append(capital)

    # ─── Métricas ─────────────────────────────────────────────────────────────
    eq     = pd.Series(equity, index=df.index[: len(equity)])
    ret    = (capital - 10_000) / 10_000
    rets   = eq.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    dd     = ((eq - eq.cummax()) / eq.cummax()).min()
    wins   = [t for t in trades if t["pnl"] >  0]
    losses = [t for t in trades if t["pnl"] <= 0]
    pf     = (
        sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))
        if losses else float("inf")
    )
    wr     = len(wins) / len(trades) if trades else 0
    bh_ret = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]

    exit_reasons: dict = {}
    for t in trades:
        exit_reasons[t["exit_reason"]] = exit_reasons.get(t["exit_reason"], 0) + 1

    # ─── Impressão dos Resultados ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"{'MÉTRICA':<28} {'BOT':>10} {'BUY & HOLD':>12}")
    print(f"{'─'*60}")
    print(f"{'Retorno Total':<28} {ret:>+9.1%} {bh_ret:>+11.1%}")
    print(f"{'Sharpe Ratio':<28} {sharpe:>10.3f} {'—':>12}")
    print(f"{'Max Drawdown':<28} {dd:>+9.1%} {'—':>12}")
    print(f"{'Win Rate':<28} {wr:>9.1%} {'—':>12}")
    print(f"{'Profit Factor':<28} {pf:>10.3f} {'—':>12}")
    print(f"{'Nº de Trades':<28} {len(trades):>10} {'—':>12}")
    print(f"{'Alpha vs B&H':<28} {ret - bh_ret:>+9.1%} {'—':>12}")
    print(f"{'Capital Final':<28} ${capital:>9,.2f} {'—':>12}")
    print(f"{'─'*60}")
    print(f"Saídas: {exit_reasons}")
    print(f"{'='*60}\n")

    return {
        "symbol"        : symbol,
        "start"         : start,
        "end"           : end or str(date.today()),
        "data_source"   : data_source,
        "n_candles"     : len(df),
        "n_trades"      : len(trades),
        "return_total"  : ret,
        "sharpe_ratio"  : sharpe,
        "max_drawdown"  : dd,
        "win_rate"      : wr,
        "profit_factor" : pf,
        "final_capital" : capital,
        "bh_return"     : bh_ret,
        "alpha_vs_bh"   : ret - bh_ret,
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Krypton TradeBot — Backtest Standalone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python backtest.py --symbol SOLUSDT --start 2022-01-01 --end 2026-06-01
  python backtest.py --symbol BTCUSDT --start 2024-01-01
  python backtest.py  # SOLUSDT desde 2022-01-01
        """,
    )
    parser.add_argument(
        "--symbol", default="SOLUSDT",
        choices=["SOLUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT"],
        help="Par de trading (padrão: SOLUSDT)",
    )
    parser.add_argument(
        "--start", default="2022-01-01",
        help="Data de início YYYY-MM-DD (padrão: 2022-01-01)",
    )
    parser.add_argument(
        "--end", default=None,
        help="Data de fim YYYY-MM-DD (padrão: hoje)",
    )
    args = parser.parse_args()
    run_backtest(args.symbol, args.start, args.end)
