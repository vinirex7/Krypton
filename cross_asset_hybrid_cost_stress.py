"""Cost-stress validation for the frozen cross-asset hybrid candidate.

Candidate parameters are fixed from the prior slow-cadence plateau:
- 10% cross-asset alpha sleeve
- 45-day allocator rebalance
- 15% target volatility
- top 2 assets, breadth >= 2
- 90-day sleeve rebalance

Only execution costs are stressed. The baseline is left unchanged because this
experiment asks whether the incremental alpha remains worthwhile under adverse
cost assumptions.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import adaptive_portfolio as ap
import backtest
import cross_asset_allocation as ca
import cross_asset_hybrid_v2 as hv2
import deep_validation as dv
import walk_forward as wf

LIVE_SYMBOLS = list(wf.BASE_WEIGHTS)
ALLOC_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
CADENCE = 45
ALPHA_WEIGHT = 0.10
TARGET_VOL = 0.15
SLEEVE_REBALANCE_DAYS = 90
BASE_TRANSFER_COST = 0.003
backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"

# Pre-declared stress grid: no signal/weight parameters vary.
SCENARIOS = {
    "standard": {"exchange_mult": 1.0, "transfer_cost": 0.003},
    "double_transfer": {"exchange_mult": 1.0, "transfer_cost": 0.006},
    "double_exchange": {"exchange_mult": 2.0, "transfer_cost": 0.003},
    "double_all_costs": {"exchange_mult": 2.0, "transfer_cost": 0.006},
}


def period_table(curves, start, end):
    windows = {
        "development": (start, "2024-12-31"),
        "2022_bear": ("2022-01-01", "2022-12-31"),
        "2025_h1": ("2025-01-01", "2025-06-30"),
        "2025_h2": ("2025-07-01", "2025-12-31"),
        "validation_2025_plus": ("2025-01-01", end),
        "2026_diagnostic": ("2026-01-01", end),
        "full": (start, end),
    }
    rows = []
    for name, curve in curves.items():
        for period, (a, b) in windows.items():
            rows.append({"variant": name, "period": period,
                         **ap.performance_metrics(ap.slice_and_rebase(curve, a, b))})
    return pd.DataFrame(rows)


def checks(periods):
    p = periods.set_index(["variant", "period"])
    base_full = p.loc[("baseline", "full")]
    base_val = p.loc[("baseline", "validation_2025_plus")]
    base_h1 = p.loc[("baseline", "2025_h1")]
    out = {}
    for scenario in SCENARIOS:
        name = f"hybrid_{scenario}"
        full = p.loc[(name, "full")]
        dev = p.loc[(name, "development")]
        val = p.loc[(name, "validation_2025_plus")]
        bear = p.loc[(name, "2022_bear")]
        h1 = p.loc[(name, "2025_h1")]
        d26 = p.loc[(name, "2026_diagnostic")]
        rules = {
            "dev_profitable": bool(dev["return"] > 0),
            "full_cagr_15pct_better": bool(full["cagr"] > base_full["cagr"] * 1.15),
            "full_calmar_better": bool(full["calmar"] > base_full["calmar"]),
            "validation_beats_baseline": bool(val["return"] > base_val["return"]),
            "max_dd_under_15pct": bool(full["max_drawdown"] >= -0.15),
            "bear_2022_above_minus_5pct": bool(bear["return"] >= -0.05),
            "h1_2025_not_over_2pp_worse": bool(h1["return"] >= base_h1["return"] - 0.02),
            "diagnostic_2026_profitable": bool(d26["return"] > 0),
        }
        out[name] = {"passed": all(rules.values()), **rules}
    return out


def _simulate_alpha(data, s, e, exchange_mult):
    old_fee, old_entry, old_exit = ca.FEE_RATE, ca.ENTRY_SLIPPAGE_PCT, ca.EXIT_SLIPPAGE_PCT
    try:
        ca.FEE_RATE = old_fee * exchange_mult
        ca.ENTRY_SLIPPAGE_PCT = old_entry * exchange_mult
        ca.EXIT_SLIPPAGE_PCT = old_exit * exchange_mult
        return hv2.simulate_breadth_allocator(
            data, ALLOC_SYMBOLS, s, e, target_vol=TARGET_VOL,
            top_n=2, min_selected=2, rebalance_days=CADENCE,
        )
    finally:
        ca.FEE_RATE, ca.ENTRY_SLIPPAGE_PCT, ca.EXIT_SLIPPAGE_PCT = old_fee, old_entry, old_exit


def run(start, end, mc_runs=5000):
    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    data = wf._prepare_data(ALLOC_SYMBOLS, s, e)

    frozen = wf.simulate_portfolio(data, LIVE_SYMBOLS, s, e, 3.0,
                                   risk_per_trade=0.01, regime_filter=True)
    baseline = ap.simulate_tactical(data, LIVE_SYMBOLS, s, e)
    if not np.isclose(frozen["final_capital"], baseline["final_capital"], rtol=0, atol=1e-7):
        raise AssertionError("baseline divergiu")
    continuity = ap.simulate_tactical(
        data, LIVE_SYMBOLS, s, e, cost_aware=True,
        entry_permission=ap.persistent_state_permission(data, LIVE_SYMBOLS),
    )

    curves = {"baseline": baseline["equity_curve"],
              "continuity_tactical": continuity["equity_curve"]}
    logs = []
    for scenario, cfg in SCENARIOS.items():
        alpha = _simulate_alpha(data, s, e, cfg["exchange_mult"])
        name = f"hybrid_{scenario}"
        curves[name] = hv2.combine_rebalanced_sleeves(
            continuity["equity_curve"], alpha["equity_curve"], ALPHA_WEIGHT,
            rebalance_days=SLEEVE_REBALANCE_DAYS,
            transfer_cost=cfg["transfer_cost"],
        )
        if not alpha["rebalance_log"].empty:
            x = alpha["rebalance_log"].copy()
            x["scenario"] = scenario
            logs.append(x)

    periods = period_table(curves, s, e)
    gate = checks(periods)
    stress_passed = all(gate[f"hybrid_{s}"]["passed"] for s in SCENARIOS)
    full = pd.DataFrame([{"variant": n, **ap.performance_metrics(c)} for n, c in curves.items()])
    mc = {n: dv.block_bootstrap_monte_carlo(c.pct_change().dropna(), runs=mc_runs, seed=42)
          for n, c in curves.items()}

    full.to_csv("cross_asset_hybrid_cost_stress_full.csv", index=False)
    periods.to_csv("cross_asset_hybrid_cost_stress_periods.csv", index=False)
    pd.DataFrame(curves).sort_index().ffill().to_csv("cross_asset_hybrid_cost_stress_curves.csv")
    (pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()).to_csv(
        "cross_asset_hybrid_cost_stress_rebalances.csv", index=False)
    report = {
        "start": start, "end": end, "candidate": "hybrid_a10_r45",
        "scenarios": SCENARIOS, "stress_passed_all": stress_passed,
        "promotion_checks": gate, "monte_carlo": mc,
        "guardrails": {"live_changed": False, "spot_only": True, "leverage": 1.0,
                       "signals_unchanged": True, "stress_grid_frozen_before_results": True,
                       "2026_is_pristine_holdout": False,
                       "promotion_requires_future_paper": True},
    }
    with open("cross_asset_hybrid_cost_stress_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nFULL\n", full.to_string(index=False))
    print("\nRETURNS\n", periods.pivot(index="variant", columns="period", values="return").to_string())
    print("\nCHECKS\n", json.dumps(gate, indent=2))
    print("\nSTRESS_PASSED_ALL", stress_passed)
    return report


def main():
    p = argparse.ArgumentParser(); p.add_argument("--start", default="2020-08-01")
    p.add_argument("--end", default="2026-08-25"); p.add_argument("--mc-runs", type=int, default=5000)
    a = p.parse_args(); run(a.start, a.end, a.mc_runs)


if __name__ == "__main__":
    main()
