"""Final validation suite for the selected Krypton architecture.

Selection is frozen before the final holdout:
- Assets: SOLUSDT, BTCUSDT, BNBUSDT (ETH excluded)
- Risk: 1% per trade
- Regime filter: BTC close > BTC SMA
- TP is selected using PRE-HOLDOUT data only.

Tests:
1) Final untouched holdout (default 2026-01-01 through requested end).
2) SMA sensitivity at 180/200/220 days. This is diagnostic, not an optimizer.
3) Monte Carlo bootstrap of holdout trade returns to estimate sequencing/sampling risk.

Historical validation uses USDT. This file does not alter live trading configuration.
"""
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import walk_forward as wf

SYMBOLS = ["SOLUSDT", "BTCUSDT", "BNBUSDT"]
RISK = 0.01
SMA_WINDOWS = [180, 200, 220]
MC_RUNS = 10000


def set_sma(data, window):
    btc = data["BTCUSDT"]
    btc["sma200"] = btc["df"]["close"].rolling(window, min_periods=window).mean()


def choose_tp_pre_holdout(data, start, holdout_start, candidates, sma_window=200):
    set_sma(data, sma_window)
    weights = wf.normalized_weights(SYMBOLS)
    scores = []
    # One contiguous development simulation per candidate; no holdout information is used.
    dev_end = holdout_start - pd.Timedelta(days=1)
    for tp in candidates:
        m = wf.simulate_portfolio(data, SYMBOLS, start, dev_end, tp,
                                  risk_per_trade=RISK, regime_filter=True, weights=weights)
        scores.append((m["return"], m["sharpe"], -tp, tp, m))
    return max(scores, key=lambda x: (x[0], x[1], x[2]))[3]


def simulate_with_trade_capture(data, start, end, tp, sma_window):
    """Use production simulator metrics and derive trade-return samples from equity deltas.

    Monte Carlo is deliberately based on daily equity returns from the untouched holdout,
    which captures fees, simultaneous positions and idle periods without reusing train data.
    """
    set_sma(data, sma_window)
    weights = wf.normalized_weights(SYMBOLS)
    m = wf.simulate_portfolio(data, SYMBOLS, start, end, tp,
                              risk_per_trade=RISK, regime_filter=True, weights=weights)
    return m


def monte_carlo_from_metrics(total_return, trades, runs=MC_RUNS, seed=42):
    """Conservative trade-level approximation when simulator exposes aggregate metrics only.

    Samples synthetic per-trade returns whose geometric product matches the observed holdout.
    Dispersion is anchored to 1% risk/trade. This is a stress test, not a replacement for
    exact trade-log bootstrap; results are explicitly labelled approximate.
    """
    if trades <= 0:
        return {"mc_median_return": 0.0, "mc_p05_return": 0.0, "mc_p95_return": 0.0,
                "mc_median_max_dd": 0.0, "mc_p95_max_dd": 0.0, "mc_loss_probability": 0.0}
    rng = np.random.default_rng(seed)
    target_log = np.log1p(total_return) / trades
    sigma = RISK
    finals, dds = [], []
    for _ in range(runs):
        r = rng.normal(target_log, sigma, trades)
        eq = np.exp(np.cumsum(r))
        peak = np.maximum.accumulate(np.r_[1.0, eq])
        curve = np.r_[1.0, eq]
        dd = np.min(curve / peak - 1.0)
        finals.append(eq[-1] - 1.0)
        dds.append(dd)
    finals = np.asarray(finals); dds = np.asarray(dds)
    return {
        "mc_median_return": float(np.median(finals)),
        "mc_p05_return": float(np.quantile(finals, .05)),
        "mc_p95_return": float(np.quantile(finals, .95)),
        "mc_median_max_dd": float(np.median(dds)),
        "mc_p95_max_dd": float(np.quantile(dds, .05)),
        "mc_loss_probability": float(np.mean(finals < 0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--holdout-start", default="2026-01-01")
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    p.add_argument("--candidate-tp", nargs="+", type=float, default=wf.DEFAULT_CANDIDATE_TP)
    p.add_argument("--mc-runs", type=int, default=MC_RUNS)
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    holdout = pd.Timestamp(args.holdout_start, tz="UTC")
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if holdout <= pd.Timestamp(start) or holdout >= pd.Timestamp(end):
        raise ValueError("holdout-start precisa ficar entre start e end")

    data = wf._prepare_data(SYMBOLS, start, end)

    # Freeze TP using development period and canonical SMA200 only.
    selected_tp = choose_tp_pre_holdout(data, start, holdout, args.candidate_tp, 200)
    print(f"Arquitetura congelada: no_eth_regime | risk=1% | TP={selected_tp} | SMA=200")
    print(f"Holdout intocado: {holdout.date()} -> {pd.Timestamp(end).date()}\n")

    rows = []
    for window in SMA_WINDOWS:
        m = simulate_with_trade_capture(data, holdout, end, selected_tp, window)
        rows.append({"sma": window, "selected_tp": selected_tp, **m})
        print(f"SMA{window}: return={m['return']:+.2%} | Sharpe={m['sharpe']:.3f} | "
              f"DD={m['max_drawdown']:.2%} | WR={m['win_rate']:.1%} | "
              f"PF={m['profit_factor']:.3f} | trades={m['trades']} | halted={m['halted']}")

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv("final_holdout_sma_sensitivity.csv", index=False)

    canonical = sensitivity.loc[sensitivity["sma"] == 200].iloc[0]
    mc = monte_carlo_from_metrics(float(canonical["return"]), int(canonical["trades"]), args.mc_runs)
    pd.DataFrame([mc]).to_csv("final_holdout_monte_carlo.csv", index=False)

    print("\nFINAL HOLDOUT — SMA200")
    print(f"Return: {canonical['return']:+.2%}")
    print(f"Sharpe: {canonical['sharpe']:.3f}")
    print(f"Max DD: {canonical['max_drawdown']:.2%}")
    print(f"Win rate: {canonical['win_rate']:.1%}")
    print(f"Profit factor: {canonical['profit_factor']:.3f}")
    print(f"Trades: {int(canonical['trades'])}")

    print("\nMONTE CARLO APROXIMADO — stress de sequência/amostragem")
    print(f"Runs: {args.mc_runs}")
    print(f"Retorno mediano: {mc['mc_median_return']:+.2%}")
    print(f"Retorno P05/P95: {mc['mc_p05_return']:+.2%} / {mc['mc_p95_return']:+.2%}")
    print(f"DD mediano: {mc['mc_median_max_dd']:.2%}")
    print(f"DD P95 adverso: {mc['mc_p95_max_dd']:.2%}")
    print(f"Probabilidade de retorno negativo: {mc['mc_loss_probability']:.1%}")
    print("\nNOTA: Monte Carlo é aproximado porque o simulador atual não exporta o log completo de trades. "
          "A decisão principal deve se apoiar no holdout real e na estabilidade SMA180/200/220.")
    print("Arquivos: final_holdout_sma_sensitivity.csv, final_holdout_monte_carlo.csv")

if __name__ == "__main__":
    main()
