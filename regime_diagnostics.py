"""Ex-post market-regime diagnostics for Krypton.

Purpose
-------
Measure whether the fixed strategy behaves as intended across realized market
conditions:
- bull: capture upside;
- bear: preserve capital / minimize losses;
- sideways: keep exposure and turnover low.

Regime labels use the realized BTC return of each historical window. That is
INTENTIONALLY ex-post and is never used by the simulator to make trading
decisions. Therefore this file is a diagnostic tool, not a live regime filter.

No parameter search is performed here. TP stays frozen at 3x ATR and all
strategy candidates are the already-declared BTC/individual/breadth variants.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import research_validation as rv
import walk_forward as wf
from config import TAKE_PROFIT_ATR_MULT, TRADING_PAIRS
from research_metrics import concentration_metrics

DEFAULT_WINDOW_DAYS = 180
DEFAULT_BULL_THRESHOLD = 0.20
DEFAULT_BEAR_THRESHOLD = -0.20


def classify_period(btc_return: float, bull_threshold: float = DEFAULT_BULL_THRESHOLD,
                    bear_threshold: float = DEFAULT_BEAR_THRESHOLD) -> str:
    if bull_threshold <= bear_threshold:
        raise ValueError("bull_threshold deve ser maior que bear_threshold.")
    if btc_return >= bull_threshold:
        return "bull"
    if btc_return <= bear_threshold:
        return "bear"
    return "sideways"


def exposure_ratio(exposure: pd.DataFrame, n_symbols: int) -> float:
    if exposure is None or exposure.empty or n_symbols <= 0:
        return 0.0
    simultaneous = exposure.astype(float).sum(axis=1)
    return float((simultaneous / float(n_symbols)).mean())


def _periods(start: datetime, end: datetime, window_days: int):
    if window_days < 30:
        raise ValueError("window_days deve ser >= 30.")
    cursor = start
    period = 1
    while cursor <= end:
        period_end = min(cursor + timedelta(days=window_days - 1), end)
        if (period_end - cursor).days + 1 < max(30, window_days // 2):
            break
        yield period, cursor, period_end
        cursor = period_end + timedelta(days=1)
        period += 1


def _btc_period_return(data, start, end) -> float:
    df = data["BTCUSDT"]["df"].loc[pd.Timestamp(start):pd.Timestamp(end)]
    if len(df) < 2:
        return np.nan
    return float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0)


def run_diagnostics(data, symbols, start, end, window_days=DEFAULT_WINDOW_DAYS,
                    bull_threshold=DEFAULT_BULL_THRESHOLD,
                    bear_threshold=DEFAULT_BEAR_THRESHOLD):
    rows = []
    for period, period_start, period_end in _periods(start, end, window_days):
        btc_return = _btc_period_return(data, period_start, period_end)
        if not np.isfinite(btc_return):
            continue
        market_regime = classify_period(btc_return, bull_threshold, bear_threshold)

        for mode in rv.REGIME_MODES:
            label = f"regime_{mode}"
            metrics = rv.simulate(
                data,
                symbols,
                period_start,
                period_end,
                regime_mode=mode,
                drawdown_overlay=False,
            )
            concentration = concentration_metrics(
                metrics["trade_log"], metrics["equity_curve"], rv.INITIAL_CAPITAL
            )
            exp = exposure_ratio(metrics["exposure"], len(symbols))
            row = {
                "period": period,
                "start": period_start.date().isoformat(),
                "end": period_end.date().isoformat(),
                "market_regime": market_regime,
                "candidate": label,
                "btc_return": btc_return,
                "strategy_return": metrics["return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "trades": metrics["trades"],
                "exposure_ratio": exp,
                "inactivity_ratio": 1.0 - exp,
                "bull_capture_ratio": (
                    metrics["return"] / btc_return
                    if market_regime == "bull" and btc_return > 0 else np.nan
                ),
                "bear_protection_spread": (
                    metrics["return"] - btc_return if market_regime == "bear" else np.nan
                ),
                **concentration,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_regimes(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby(["candidate", "market_regime"])
        .agg(
            periods=("period", "count"),
            median_strategy_return=("strategy_return", "median"),
            mean_strategy_return=("strategy_return", "mean"),
            worst_strategy_return=("strategy_return", "min"),
            worst_drawdown=("max_drawdown", "min"),
            mean_sharpe=("sharpe", "mean"),
            total_trades=("trades", "sum"),
            mean_exposure=("exposure_ratio", "mean"),
            median_btc_return=("btc_return", "median"),
            mean_bull_capture=("bull_capture_ratio", "mean"),
            mean_bear_protection_spread=("bear_protection_spread", "mean"),
            median_return_without_best_5d=("return_without_best_5d", "median"),
            median_top5_trade_pnl_share=("top_5pct_trade_pnl_share", "median"),
        )
        .reset_index()
    )


def main():
    if abs(TAKE_PROFIT_ATR_MULT - rv.FIXED_TP) > 1e-12:
        raise RuntimeError("regime_diagnostics exige TAKE_PROFIT_ATR_MULT=3.0.")

    p = argparse.ArgumentParser(description="Krypton ex-post bull/bear/sideways diagnostics")
    p.add_argument("--start", default="2020-08-01")
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    p.add_argument("--symbols", nargs="+", default=list(TRADING_PAIRS))
    p.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    p.add_argument("--bull-threshold", type=float, default=DEFAULT_BULL_THRESHOLD)
    p.add_argument("--bear-threshold", type=float, default=DEFAULT_BEAR_THRESHOLD)
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    symbols = [s.upper() for s in args.symbols]
    if set(symbols) != set(TRADING_PAIRS):
        raise ValueError("Diagnóstico comparável exige os mesmos 3 ativos de TRADING_PAIRS.")

    data = wf._prepare_data(symbols, start, end)
    results = run_diagnostics(
        data,
        symbols,
        start,
        end,
        window_days=args.window_days,
        bull_threshold=args.bull_threshold,
        bear_threshold=args.bear_threshold,
    )
    summary = summarize_regimes(results)
    results.to_csv("regime_period_results.csv", index=False)
    summary.to_csv("regime_summary.csv", index=False)

    print("\nREGIME PERIOD RESULTS")
    print(results.to_string(index=False) if not results.empty else "Sem períodos completos.")
    print("\nREGIME SUMMARY")
    print(summary.to_string(index=False) if not summary.empty else "Sem resumo disponível.")
    print("\nArquivos: regime_period_results.csv | regime_summary.csv")
    print("Nota: os rótulos bull/bear/sideways são ex-post e NÃO são usados nos sinais do bot.")


if __name__ == "__main__":
    main()
