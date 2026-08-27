"""Frozen validation for Spot + USD-M perpetual market-neutral carry."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import adaptive_portfolio as ap
import backtest
import carry_market_neutral as cmn
import cross_asset_hybrid_v2 as hv2
import deep_validation as dv
import walk_forward as wf

LIVE_SYMBOLS = list(wf.BASE_WEIGHTS)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
ENTRY_APRS = (0.08, 0.12, 0.16)
CARRY_WEIGHTS = (0.10, 0.20, 0.30)
CARRY_CENTER_APR = 0.12
NOTIONAL_FRACTION = 0.50
CARRY_REBALANCE_DAYS = 7
CARRY_LOOKBACK_DAYS = 7
CHALLENGER_ALPHA_WEIGHT = 0.10
CHALLENGER_ALPHA_CADENCE = 45
SLEEVE_REBALANCE_DAYS = 90
TRANSFER_COST = 0.003
MAX_DD_GATE = -0.18
TARGET_CAGR = 0.15

backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"


def _period_table(curves, start, end):
    windows = {
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2024": ("2024-01-01", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
        "2025_h1": ("2025-01-01", "2025-06-30"),
        "2025_plus": ("2025-01-01", end),
        "2026_diagnostic": ("2026-01-01", end),
        "full": (start, end),
    }
    rows = []
    for name, curve in curves.items():
        for period, (a, b) in windows.items():
            rows.append({"variant": name, "period": period,
                         **ap.performance_metrics(ap.slice_and_rebase(curve, a, b))})
    return pd.DataFrame(rows)


def _challenger_curve(data, s, e):
    continuity = ap.simulate_tactical(
        data, LIVE_SYMBOLS, s, e, cost_aware=True,
        entry_permission=ap.persistent_state_permission(data, LIVE_SYMBOLS),
    )
    alpha = hv2.simulate_breadth_allocator(
        data, SYMBOLS, s, e, target_vol=0.15, top_n=2,
        min_selected=2, rebalance_days=CHALLENGER_ALPHA_CADENCE,
    )
    return hv2.combine_rebalanced_sleeves(
        continuity["equity_curve"], alpha["equity_curve"], CHALLENGER_ALPHA_WEIGHT,
        rebalance_days=SLEEVE_REBALANCE_DAYS, transfer_cost=TRANSFER_COST,
    )


def _promotion_checks(periods, carry_stats):
    p = periods.set_index(["variant", "period"])
    base = p.loc[("challenger_v1", "full")]
    base_25 = p.loc[("challenger_v1", "2025_plus")]
    out = {}
    for w in CARRY_WEIGHTS:
        name = f"challenger_carry_{int(w*100)}"
        full = p.loc[(name, "full")]
        p25 = p.loc[(name, "2025_plus")]
        d26 = p.loc[(name, "2026_diagnostic")]
        rules = {
            "cagr_target_15pct": bool(full["cagr"] >= TARGET_CAGR),
            "cagr_beats_challenger": bool(full["cagr"] > base["cagr"]),
            "sharpe_at_least_1_10": bool(full["sharpe"] >= 1.10),
            "calmar_beats_challenger": bool(full["calmar"] > base["calmar"]),
            "max_dd_under_18pct": bool(full["max_drawdown"] >= MAX_DD_GATE),
            "2025_plus_not_worse": bool(p25["return"] >= base_25["return"] - 0.01),
            "2026_positive": bool(d26["return"] > 0),
            "carry_no_liquidations": bool(carry_stats["liquidation_events"] == 0),
        }
        out[name] = {"passed": all(rules.values()), **rules}
    return out


def run(start, end, mc_runs=5000):
    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    data = wf._prepare_data(SYMBOLS, s, e)

    frozen = wf.simulate_portfolio(data, LIVE_SYMBOLS, s, e, 3.0,
                                   risk_per_trade=0.01, regime_filter=True)
    baseline = ap.simulate_tactical(data, LIVE_SYMBOLS, s, e)
    if not np.isclose(frozen["final_capital"], baseline["final_capital"], rtol=0, atol=1e-7):
        raise AssertionError("baseline divergiu")

    challenger = _challenger_curve(data, s, e)

    futures = {}
    for sym in SYMBOLS:
        perp, funding = cmn.download_perp_and_funding(sym, s, e)
        futures[sym] = {"perp": perp, "funding": funding}
        if sym in {"BTCUSDT", "ETHUSDT", "BNBUSDT"} and (perp.empty or funding.empty):
            raise AssertionError(f"dados futures ausentes para {sym}")

    curves = {"challenger_v1": challenger}
    carry_results = {}
    for threshold in ENTRY_APRS:
        result = cmn.simulate_funding_carry(
            data, futures, SYMBOLS, s, e, entry_apr=threshold, exit_apr=0.0,
            lookback_days=CARRY_LOOKBACK_DAYS, top_n=2,
            rebalance_days=CARRY_REBALANCE_DAYS,
            notional_fraction=NOTIONAL_FRACTION,
        )
        name = f"carry_apr_{int(threshold*100)}"
        carry_results[name] = result
        curves[name] = result["equity_curve"]

    center = carry_results["carry_apr_12"]
    for w in CARRY_WEIGHTS:
        name = f"challenger_carry_{int(w*100)}"
        curves[name] = hv2.combine_rebalanced_sleeves(
            challenger, center["equity_curve"], w,
            rebalance_days=SLEEVE_REBALANCE_DAYS, transfer_cost=TRANSFER_COST,
        )

    periods = _period_table(curves, s, e)
    full = pd.DataFrame([{"variant": n, **ap.performance_metrics(c)}
                         for n, c in curves.items()])
    threshold_robust = all(
        carry_results[f"carry_apr_{int(t*100)}"]["return"] > 0 and
        carry_results[f"carry_apr_{int(t*100)}"]["liquidation_events"] == 0
        for t in ENTRY_APRS
    )
    checks = _promotion_checks(periods, center)
    for value in checks.values():
        value["carry_threshold_family_positive"] = bool(threshold_robust)
        value["passed"] = bool(value["passed"] and threshold_robust)

    mc_names = ["challenger_v1"] + [f"challenger_carry_{int(w*100)}" for w in CARRY_WEIGHTS]
    mc = {name: dv.block_bootstrap_monte_carlo(
        curves[name].pct_change().dropna(), runs=mc_runs, seed=42)
        for name in mc_names}

    full.to_csv("carry_market_neutral_full_results.csv", index=False)
    periods.to_csv("carry_market_neutral_period_results.csv", index=False)
    pd.DataFrame(curves).sort_index().ffill().to_csv("carry_market_neutral_equity_curves.csv")
    center["trade_log"].to_csv("carry_market_neutral_trades.csv", index=False)
    funding_summary = []
    for sym in SYMBOLS:
        f = futures[sym]["funding"]
        if f.empty:
            continue
        funding_summary.append({
            "symbol": sym, "events_days": int((f != 0).sum()),
            "mean_daily_funding": float(f.mean()),
            "annualized_mean": float(f.mean() * 365),
            "positive_day_ratio": float((f > 0).mean()),
        })
    pd.DataFrame(funding_summary).to_csv("carry_market_neutral_funding_summary.csv", index=False)

    report = {
        "start": start, "end": end,
        "design": {
            "market_neutral": "long Spot + equal-base short USD-M perpetual",
            "symbols": SYMBOLS, "entry_apr_family": list(ENTRY_APRS),
            "center_entry_apr": CARRY_CENTER_APR, "exit_apr": 0.0,
            "lookback_days": CARRY_LOOKBACK_DAYS,
            "carry_rebalance_days": CARRY_REBALANCE_DAYS,
            "notional_fraction": NOTIONAL_FRACTION,
            "futures_taker_fee": cmn.FUTURES_TAKER_FEE,
            "futures_slippage": cmn.FUTURES_SLIPPAGE,
            "spot_fee": cmn.FEE_RATE,
            "spot_entry_slippage": cmn.ENTRY_SLIPPAGE_PCT,
            "spot_exit_slippage": cmn.EXIT_SLIPPAGE_PCT,
        },
        "threshold_family_positive": threshold_robust,
        "carry_component": {
            name: {k: v for k, v in result.items() if k not in {"equity_curve", "trade_log"}}
            for name, result in carry_results.items()
        },
        "promotion_checks": checks,
        "monte_carlo": mc,
        "guardrails": {
            "main_changed": False, "live_changed": False,
            "derivatives_research_only": True,
            "max_portfolio_dd_gate": abs(MAX_DD_GATE),
            "2026_is_pristine_holdout": False,
            "promotion_requires_forward_paper": True,
        },
    }
    with open("carry_market_neutral_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("\nFULL RESULTS\n", full.to_string(index=False))
    print("\nYEAR/PERIOD RETURNS\n", periods.pivot(index="variant", columns="period", values="return").to_string())
    print("\nTHRESHOLD FAMILY POSITIVE", threshold_robust)
    print("\nPROMOTION CHECKS\n", json.dumps(checks, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2020-08-01")
    p.add_argument("--end", default="2026-08-25")
    p.add_argument("--mc-runs", type=int, default=5000)
    a = p.parse_args()
    run(a.start, a.end, a.mc_runs)


if __name__ == "__main__":
    main()
