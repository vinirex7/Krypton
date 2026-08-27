"""Frozen validation for complementary Spot mean reversion versus Challenger v1."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import adaptive_portfolio as ap
import backtest
import cross_asset_hybrid_v2 as hv2
import deep_validation as dv
import mean_reversion_alpha as mr
import walk_forward as wf

LIVE_SYMBOLS = list(wf.BASE_WEIGHTS)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
MR_WEIGHTS = (0.05, 0.10, 0.15)
TRANSFER_COST = 0.003
SLEEVE_REBALANCE_DAYS = 90
backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"


def metrics(curve):
    return ap.performance_metrics(curve)


def period_table(curves, start, end):
    windows = {
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2024": ("2024-01-01", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
        "2025_h1": ("2025-01-01", "2025-06-30"),
        "2025_plus": ("2025-01-01", end),
        "2026": ("2026-01-01", end),
        "full": (start, end),
    }
    rows = []
    for name, curve in curves.items():
        for period, (a, b) in windows.items():
            rows.append({"variant": name, "period": period,
                         **metrics(ap.slice_and_rebase(curve, a, b))})
    return pd.DataFrame(rows)


def promotion_checks(periods, trade_log):
    p = periods.set_index(["variant", "period"])
    v1 = p.loc[("challenger_v1", "full")]
    v1_val = p.loc[("challenger_v1", "2025_plus")]
    v1_h1 = p.loc[("challenger_v1", "2025_h1")]
    out = {}
    ntrades = len(trade_log)
    for w in MR_WEIGHTS:
        name = f"challenger_mr_{int(w*100)}"
        full = p.loc[(name, "full")]
        val = p.loc[(name, "2025_plus")]
        h1 = p.loc[(name, "2025_h1")]
        b22 = p.loc[(name, "2022")]
        y26 = p.loc[(name, "2026")]
        rules = {
            "cagr_ge_15pct": bool(full["cagr"] >= 0.15),
            "cagr_better_than_v1": bool(full["cagr"] > v1["cagr"]),
            "sharpe_ge_v1": bool(full["sharpe"] >= v1["sharpe"]),
            "calmar_ge_v1": bool(full["calmar"] >= v1["calmar"]),
            "max_dd_under_15pct": bool(full["max_drawdown"] >= -0.15),
            "2025_plus_ge_v1": bool(val["return"] >= v1_val["return"]),
            "2025_h1_not_2pp_worse": bool(h1["return"] >= v1_h1["return"] - 0.02),
            "2022_gt_minus5pct": bool(b22["return"] >= -0.05),
            "2026_positive": bool(y26["return"] > 0),
            "mr_sample_at_least_30_trades": bool(ntrades >= 30),
        }
        out[name] = {"passed": all(rules.values()), **rules}
    return out


def run(start, end, mc_runs=5000):
    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    data = wf._prepare_data(SYMBOLS, s, e)

    baseline = ap.simulate_tactical(data, LIVE_SYMBOLS, s, e)
    continuity = ap.simulate_tactical(
        data, LIVE_SYMBOLS, s, e, cost_aware=True,
        entry_permission=ap.persistent_state_permission(data, LIVE_SYMBOLS),
    )
    trend_alpha = hv2.simulate_breadth_allocator(
        data, SYMBOLS, s, e, target_vol=0.15, top_n=2,
        min_selected=2, rebalance_days=45,
    )
    challenger = hv2.combine_rebalanced_sleeves(
        continuity["equity_curve"], trend_alpha["equity_curve"], 0.10,
        rebalance_days=90, transfer_cost=TRANSFER_COST,
    )

    mr_sleeve = mr.simulate_mean_reversion(
        data, SYMBOLS, s, e, max_positions=1,
        sleeve_target=1.0, max_hold_days=5, stop_atr=2.0,
    )

    curves = {
        "baseline": baseline["equity_curve"],
        "challenger_v1": challenger,
        "mean_reversion_sleeve": mr_sleeve["equity_curve"],
    }
    for w in MR_WEIGHTS:
        curves[f"challenger_mr_{int(w*100)}"] = hv2.combine_rebalanced_sleeves(
            challenger, mr_sleeve["equity_curve"], w,
            rebalance_days=SLEEVE_REBALANCE_DAYS,
            transfer_cost=TRANSFER_COST,
        )

    periods = period_table(curves, s, e)
    checks = promotion_checks(periods, mr_sleeve["trade_log"])
    full = pd.DataFrame([{"variant": n, **metrics(c)} for n, c in curves.items()])
    mc = {
        n: dv.block_bootstrap_monte_carlo(c.pct_change().dropna(), runs=mc_runs, seed=42)
        for n, c in curves.items() if n != "mean_reversion_sleeve"
    }

    selected = None
    for w in MR_WEIGHTS:
        n = f"challenger_mr_{int(w*100)}"
        if checks[n]["passed"]:
            selected = n
            break

    full.to_csv("mean_reversion_full.csv", index=False)
    periods.to_csv("mean_reversion_periods.csv", index=False)
    pd.DataFrame(curves).sort_index().ffill().to_csv("mean_reversion_curves.csv")
    mr_sleeve["trade_log"].to_csv("mean_reversion_trades.csv", index=False)
    report = {
        "start": start, "end": end,
        "hypothesis": "daily pullback reversion only while frozen trend alpha is ineligible",
        "frozen_rules": {
            "asset_above_own_sma200": True,
            "three_day_return_lte": -0.05,
            "rsi2_lte": 10,
            "exit": "next open after close>=SMA5 or 5 bars; 2ATR intraday stop",
            "spot_only": True, "leverage": 1.0,
            "mr_weights": list(MR_WEIGHTS),
        },
        "mr_trade_count": int(len(mr_sleeve["trade_log"])),
        "promotion_checks": checks,
        "selected_candidate": selected,
        "monte_carlo": mc,
        "guardrails": {"live_changed": False, "next_open": True,
                       "parameters_frozen_before_results": True,
                       "promotion_requires_forward_paper": True},
    }
    with open("mean_reversion_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nFULL\n", full.to_string(index=False))
    print("\nPERIOD RETURNS\n", periods.pivot(index="variant", columns="period", values="return").to_string())
    print("\nTRADES", len(mr_sleeve["trade_log"]))
    print("\nCHECKS\n", json.dumps(checks, indent=2))
    print("\nSELECTED", selected or "NONE")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2020-08-01")
    p.add_argument("--end", default="2026-08-25")
    p.add_argument("--mc-runs", type=int, default=5000)
    a = p.parse_args(); run(a.start, a.end, a.mc_runs)


if __name__ == "__main__":
    main()
