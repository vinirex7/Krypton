# backtest.py — Backtesting Standalone com Dados da Binance US
# Krypton TradeBot | Estratégia: Supertrend + RSI + MACD Filter
#
# Usa a API pública da Binance US (sem autenticação) para baixar dados históricos.
# A API global da Binance (api.binance.com) pode ser bloqueada dependendo da região —
# a Binance US (api.binance.us) não tem essa restrição geográfica.
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

# Mapeamento Binance global → Binance US
SYMBOL_MAP = {
    "SOLUSDT": "SOLUSD",
    "BTCUSDT": "BTCUSD",
    "ETHUSDT": "ETHUSD",
    "BNBUSDT": "BNBUSD",
}

BINANCE_US_URL = "https://api.binance.us/api/v3/klines"


def get_ohlcv_binance_us(symbol: str, start_str: str, end_str: str | None = None) -> pd.DataFrame:
    """
    Baixa dados históricos OHLCV da Binance US (API pública, sem autenticação).

    Busca em lotes de 1000 candles para cobrir períodos longos.
    Sem restrição geográfica.
    """
    symbol_us = SYMBOL_MAP.get(symbol.upper(), symbol)
    start_ts  = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts    = int(datetime.strptime(end_str,   "%Y-%m-%d").timestamp() * 1000) if end_str else None

    all_klines  = []
    current_ts  = start_ts

    while True:
        params = {
            "symbol"   : symbol_us,
            "interval" : "1d",
            "startTime": current_ts,
            "limit"    : 1000,
        }
        if end_ts:
            params["endTime"] = end_ts

        try:
            r      = requests.get(BINANCE_US_URL, params=params, timeout=15)
            klines = r.json()
        except Exception as e:
            print(f"  Erro ao buscar dados: {e}. Tentando novamente em 5s...")
            time.sleep(5)
            continue

        if not klines or isinstance(klines, dict):
            break

        all_klines.extend(klines)

        if len(klines) < 1000:
            break

        current_ts = klines[-1][0] + 86_400_000  # avança 1 dia em ms
        if end_ts and current_ts >= end_ts:
            break

    if not all_klines:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_klines,
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
    print("Baixando dados históricos via Binance US...", end=" ", flush=True)

    df = get_ohlcv_binance_us(symbol, start, end)

    if len(df) < 50:
        print(f"\n❌ Dados insuficientes ({len(df)} candles). Verifique o símbolo e as datas.")
        return {}

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
        description="Krypton TradeBot — Backtest Standalone (Binance US API)",
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
