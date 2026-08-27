"""Robustness check around pre-declared aggressive tier B.

This is not a parameter search. A single risk-budget scalar (0.9/1.0/1.1) is
applied coherently to tactical risk, alpha sleeve weight and alpha target vol.
Tier B is also re-run with doubled spot fees/slippage and doubled sleeve transfer
cost. Signals, lookbacks and cadence remain frozen.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

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
BASE_TRANSFER = 0.003
DD_CAP = 0.30
BASE = {"risk_per_trade": .015, "alpha_weight": .35, "target_vol": .25}
SCALES = (0.90, 1.00, 1.10)
backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"


def simulate_variant(data, s, e, cfg, *, transfer_cost=BASE_TRANSFER):
    permission = ap.persistent_state_permission(data, LIVE)
    tactical = ap.simulate_tactical(data, LIVE, s, e, cost_aware=True,
                                    entry_permission=permission,
                                    risk_per_trade=cfg["risk_per_trade"])
    alpha = hv2.simulate_breadth_allocator(data, ASSETS, s, e,
                                           target_vol=cfg["target_vol"], top_n=2,
                                           min_selected=2, rebalance_days=CADENCE)
    curve = hv2.combine_rebalanced_sleeves(
        tactical["equity_curve"], alpha["equity_curve"], cfg["alpha_weight"],
        rebalance_days=SLEEVE_DAYS, transfer_cost=transfer_cost)
    return curve


def run(start="2020-08-01", end="2026-08-25", mc_runs=5000):
    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    data = wf._prepare_data(ASSETS, s, e)
    original_dd = ap.MAX_DRAWDOWN_PCT
    ap.MAX_DRAWDOWN_PCT = DD_CAP
    curves = {}
    try:
        for scale in SCALES:
            cfg = {
                "risk_per_trade": BASE["risk_per_trade"] * scale,
                "alpha_weight": BASE["alpha_weight"] * scale,
                "target_vol": BASE["target_vol"] * scale,
            }
            curves[f"B_x{scale:.2f}"] = simulate_variant(data, s, e, cfg)

        # Cost stress: all spot exchange friction used by the tactical and alpha
        # engines is doubled, as is the transfer cost between sleeves.
        originals = {
            "ap_fee": ap.FEE_RATE, "ap_entry": ap.ENTRY_SLIPPAGE_PCT,
            "ap_exit": ap.EXIT_SLIPPAGE_PCT, "hv_fee": hv2.FEE_RATE,
            "hv_entry": hv2.ENTRY_SLIPPAGE_PCT, "hv_exit": hv2.EXIT_SLIPPAGE_PCT,
        }
        try:
            ap.FEE_RATE *= 2.0
            ap.ENTRY_SLIPPAGE_PCT *= 2.0
            ap.EXIT_SLIPPAGE_PCT *= 2.0
            hv2.FEE_RATE *= 2.0
            hv2.ENTRY_SLIPPAGE_PCT *= 2.0
            hv2.EXIT_SLIPPAGE_PCT *= 2.0
            curves["B_costs_2x"] = simulate_variant(data, s, e, BASE,
                                                      transfer_cost=BASE_TRANSFER * 2.0)
        finally:
            ap.FEE_RATE = originals["ap_fee"]
            ap.ENTRY_SLIPPAGE_PCT = originals["ap_entry"]
            ap.EXIT_SLIPPAGE_PCT = originals["ap_exit"]
            hv2.FEE_RATE = originals["hv_fee"]
            hv2.ENTRY_SLIPPAGE_PCT = originals["hv_entry"]
            hv2.EXIT_SLIPPAGE_PCT = originals["hv_exit"]
    finally:
        ap.MAX_DRAWDOWN_PCT = original_dd

    rows, mc = [], {}
    for name, curve in curves.items():
        m = ap.performance_metrics(curve)
        rows.append({"variant": name, **m})
        mc[name] = dv.block_bootstrap_monte_carlo(curve.pct_change().dropna(),
                                                   runs=mc_runs, seed=43)
    full = pd.DataFrame(rows)

    # Plateau criterion: all neighboring risk budgets must remain economically
    # useful; cost stress must not erase the 18% CAGR hurdle or breach 30% DD.
    checks = {}
    for row in rows:
        checks[row["variant"]] = {
            "cagr_ge_18": bool(row["cagr"] >= .18),
            "dd_le_30": bool(row["max_drawdown"] >= -.30),
            "sharpe_ge_090": bool(row["sharpe"] >= .90),
        }
        checks[row["variant"]]["passed"] = all(checks[row["variant"]].values())
    report = {
        "start": start, "end": end,
        "base_tier": BASE,
        "risk_budget_scales": SCALES,
        "checks": checks,
        "plateau_passed": all(checks[f"B_x{x:.2f}"]["passed"] for x in SCALES),
        "cost_stress_passed": checks["B_costs_2x"]["passed"],
        "monte_carlo": mc,
        "guardrails": {"live_changed": False, "signals_changed": False,
                       "cost_stress_effective_required": True},
    }
    if not full.loc[full.variant.eq("B_costs_2x"), "final_capital"].iloc[0] < full.loc[full.variant.eq("B_x1.00"), "final_capital"].iloc[0]:
        raise AssertionError("cost stress inefetivo")

    full.to_csv("aggressive_risk_robustness_full.csv", index=False)
    pd.DataFrame(curves).to_csv("aggressive_risk_robustness_curves.csv")
    with open("aggressive_risk_robustness_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(full.to_string(index=False))
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
