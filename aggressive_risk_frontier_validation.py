"""Frozen aggressive-risk research for Krypton.

Goal: maximize robust CAGR after the user explicitly accepted more crypto
volatility. This is research-only: no live config or order path is changed.

The experiment changes risk budgets, not signal thresholds. Three pre-declared
risk tiers are tested as a frontier rather than selecting a magic parameter:
  A: tactical 1.25% risk/trade + 25% cross-asset sleeve at 20% target vol
  B: tactical 1.50% risk/trade + 35% cross-asset sleeve at 25% target vol
  C: tactical 2.00% risk/trade + 45% cross-asset sleeve at 30% target vol
All use the already-validated continuity gate, 45-bar cross-asset cadence,
90-day sleeve rebalance, next-open execution and existing costs.
"""
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
import walk_forward as wf

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
LIVE = list(wf.BASE_WEIGHTS)
CADENCE = 45
SLEEVE_DAYS = 90
TRANSFER_COST = 0.003
AGGRESSIVE_DD_CAP = 0.30

# Pre-declared before results. Do not optimize these after observing the holdout.
TIERS = {
    "aggressive_A": {"risk_per_trade": 0.0125, "alpha_weight": 0.25, "target_vol": 0.20},
    "aggressive_B": {"risk_per_trade": 0.0150, "alpha_weight": 0.35, "target_vol": 0.25},
    "aggressive_C": {"risk_per_trade": 0.0200, "alpha_weight": 0.45, "target_vol": 0.30},
}

# New objective: accept materially more volatility only for materially more CAGR.
GATES = {
    "cagr_hurdle": 0.18,
    "max_drawdown_floor": -0.30,
    "sharpe_floor": 0.90,
    "calmar_floor": 0.60,
    "year_2022_floor": -0.15,
}

backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"


def _periods(curves, start, end):
    windows = {
        "2021": ("2021-01-01", "2021-12-31"),
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2024": ("2024-01-01", "2024-12-31"),
        "2025_h1": ("2025-01-01", "2025-06-30"),
        "2025": ("2025-01-01", "2025-12-31"),
        "2025_plus": ("2025-01-01", end),
        "2026": ("2026-01-01", end),
        "full": (start, end),
    }
    rows = []
    for name, curve in curves.items():
        for period, (a, b) in windows.items():
            rows.append({"variant": name, "period": period,
                         **ap.performance_metrics(ap.slice_and_rebase(curve, a, b))})
    return pd.DataFrame(rows)


def _gate(metrics, periods):
    p = periods.set_index("period")
    checks = {
        "cagr_ge_18": bool(metrics["cagr"] >= GATES["cagr_hurdle"]),
        "dd_le_30": bool(metrics["max_drawdown"] >= GATES["max_drawdown_floor"]),
        "sharpe_ge_090": bool(metrics["sharpe"] >= GATES["sharpe_floor"]),
        "calmar_ge_060": bool(metrics["calmar"] >= GATES["calmar_floor"]),
        "2022_gt_minus15": bool(p.loc["2022", "return"] >= GATES["year_2022_floor"]),
        "2025_plus_positive": bool(p.loc["2025_plus", "return"] > 0),
        "2026_positive": bool(p.loc["2026", "return"] > 0),
    }
    return {"passed": all(checks.values()), **checks}


def run(start="2020-08-01", end="2026-08-25", mc_runs=5000):
    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    data = wf._prepare_data(ASSETS, s, e)

    # Frozen current engine reproduction.
    frozen = wf.simulate_portfolio(data, LIVE, s, e, 3.0, risk_per_trade=.01, regime_filter=True)
    baseline = ap.simulate_tactical(data, LIVE, s, e)
    if not np.isclose(frozen["final_capital"], baseline["final_capital"], rtol=0, atol=1e-7):
        raise AssertionError("baseline divergiu do simulador congelado")

    permission = ap.persistent_state_permission(data, LIVE)
    continuity = ap.simulate_tactical(data, LIVE, s, e, cost_aware=True,
                                      entry_permission=permission, risk_per_trade=.01)
    alpha15 = hv2.simulate_breadth_allocator(data, ASSETS, s, e, target_vol=.15,
                                             top_n=2, min_selected=2,
                                             rebalance_days=CADENCE)
    challenger = hv2.combine_rebalanced_sleeves(
        continuity["equity_curve"], alpha15["equity_curve"], .10,
        rebalance_days=SLEEVE_DAYS, transfer_cost=TRANSFER_COST)

    curves = {"baseline": baseline["equity_curve"], "challenger_v1": challenger}
    component_rows = []

    # Research-only change: the simulator may continue until a 30% portfolio
    # drawdown instead of the live/default cap. No production constant is edited.
    original_dd_cap = ap.MAX_DRAWDOWN_PCT
    try:
        ap.MAX_DRAWDOWN_PCT = AGGRESSIVE_DD_CAP
        for name, cfg in TIERS.items():
            tactical = ap.simulate_tactical(
                data, LIVE, s, e, cost_aware=True, entry_permission=permission,
                risk_per_trade=cfg["risk_per_trade"])
            alpha = hv2.simulate_breadth_allocator(
                data, ASSETS, s, e, target_vol=cfg["target_vol"], top_n=2,
                min_selected=2, rebalance_days=CADENCE)
            combined = hv2.combine_rebalanced_sleeves(
                tactical["equity_curve"], alpha["equity_curve"], cfg["alpha_weight"],
                rebalance_days=SLEEVE_DAYS, transfer_cost=TRANSFER_COST)
            curves[name] = combined
            component_rows.append({
                "variant": name,
                **cfg,
                "tactical_return": tactical["return"],
                "tactical_cagr": tactical["cagr"],
                "tactical_dd": tactical["max_drawdown"],
                "tactical_trades": tactical["trades"],
                "alpha_return": alpha["return"],
                "alpha_cagr": alpha["cagr"],
                "alpha_dd": alpha["max_drawdown"],
            })
    finally:
        ap.MAX_DRAWDOWN_PCT = original_dd_cap

    full = pd.DataFrame([{"variant": n, **ap.performance_metrics(c)} for n, c in curves.items()])
    periods = _periods(curves, start, end)
    full_ix = full.set_index("variant")
    gates = {}
    for name in TIERS:
        gates[name] = _gate(full_ix.loc[name].to_dict(), periods[periods.variant == name])

    # Robust frontier requirement: promotion is the lowest-risk tier that passes.
    selected = next((name for name in TIERS if gates[name]["passed"]), None)

    mc = {}
    for name in ["challenger_v1", *TIERS.keys()]:
        mc[name] = dv.block_bootstrap_monte_carlo(
            curves[name].pct_change().dropna(), runs=mc_runs, seed=42)

    # Return concentration: strategy should not exist only because of a handful of days.
    concentration = {}
    for name, curve in curves.items():
        r = curve.pct_change().dropna().sort_values(ascending=False)
        def compound_without(k):
            x = r.iloc[k:]
            return float((1.0 + x).prod() - 1.0) if len(x) else 0.0
        concentration[name] = {
            "return_without_best_1d": compound_without(1),
            "return_without_best_5d": compound_without(5),
            "return_without_best_20d": compound_without(20),
        }

    report = {
        "start": start, "end": end,
        "objective": "maximize robust CAGR with max drawdown <= 30%",
        "tiers_frozen_before_results": TIERS,
        "gates": gates,
        "selected_candidate": selected,
        "monte_carlo": mc,
        "concentration": concentration,
        "guardrails": {
            "live_changed": False,
            "leverage_directional": False,
            "max_research_drawdown": AGGRESSIVE_DD_CAP,
            "next_open_execution": True,
            "fees_and_slippage": True,
            "promotion_requires_forward_paper": True,
        },
    }
    full.to_csv("aggressive_risk_frontier_full.csv", index=False)
    periods.to_csv("aggressive_risk_frontier_periods.csv", index=False)
    pd.DataFrame(curves).to_csv("aggressive_risk_frontier_curves.csv")
    pd.DataFrame(component_rows).to_csv("aggressive_risk_frontier_components.csv", index=False)
    with open("aggressive_risk_frontier_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("FULL")
    print(full.to_string(index=False))
    print("\nPERIOD RETURNS")
    print(periods.pivot(index="variant", columns="period", values="return").to_string())
    print("\nCOMPONENTS")
    print(pd.DataFrame(component_rows).to_string(index=False))
    print("\nREPORT")
    print(json.dumps(report, indent=2, default=str))
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
