"""Frozen validation for Cross-Asset Alpha v2.

Challenger v1 is the benchmark. Feature ablations are diagnostic only. Promotion
uses one predeclared FULL-v2 candidate; only if its 10% hybrid beats Challenger
v1 are 15/20/25% weights eligible. The smallest weight passing every gate wins.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import adaptive_portfolio as ap
import backtest
import cross_asset_alpha_v2 as av2
import cross_asset_hybrid_v2 as hv2
import deep_validation as dv
import walk_forward as wf

LIVE_SYMBOLS = list(wf.BASE_WEIGHTS)
ALLOC_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
CADENCE = 45
TARGET_VOL = 0.15
SLEEVE_REBALANCE_DAYS = 90
TRANSFER_COST = 0.003
WEIGHTS = (0.10, 0.15, 0.20, 0.25)
PROJECT_CAGR_HURDLE = 0.15
backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"

VARIANTS = {
    "alpha_v2_tsmom": dict(use_tsmom=True),
    "alpha_v2_vol": dict(use_vol_regime=True),
    "alpha_v2_dispersion": dict(use_dispersion=True),
    "alpha_v2_persistent": dict(use_persistent_regime=True),
    "alpha_v2_full": dict(use_tsmom=True, use_vol_regime=True,
                          use_dispersion=True, use_persistent_regime=True),
}


def period_table(curves, start, end):
    windows = {"development": (start, "2024-12-31"), "2022_bear": ("2022-01-01", "2022-12-31"),
               "2023": ("2023-01-01", "2023-12-31"), "2024": ("2024-01-01", "2024-12-31"),
               "2025_h1": ("2025-01-01", "2025-06-30"), "2025": ("2025-01-01", "2025-12-31"),
               "validation_2025_plus": ("2025-01-01", end), "2026_diagnostic": ("2026-01-01", end),
               "full": (start, end)}
    rows = []
    for name, curve in curves.items():
        for period, (a, b) in windows.items():
            rows.append({"variant": name, "period": period,
                         **ap.performance_metrics(ap.slice_and_rebase(curve, a, b))})
    return pd.DataFrame(rows)


def architecture_gate(periods):
    p = periods.set_index(["variant", "period"])
    v1f, v2f = p.loc[("challenger_v1", "full")], p.loc[("alpha_v2_full_a10", "full")]
    v1v, v2v = p.loc[("challenger_v1", "validation_2025_plus")], p.loc[("alpha_v2_full_a10", "validation_2025_plus")]
    rules = {"cagr_better_than_v1": bool(v2f["cagr"] > v1f["cagr"]),
             "sharpe_better_than_v1": bool(v2f["sharpe"] > v1f["sharpe"]),
             "calmar_better_than_v1": bool(v2f["calmar"] > v1f["calmar"]),
             "max_dd_under_15pct": bool(v2f["max_drawdown"] >= -0.15),
             "validation_not_worse": bool(v2v["return"] >= v1v["return"]),
             "2026_profitable": bool(p.loc[("alpha_v2_full_a10", "2026_diagnostic"), "return"] > 0)}
    return {"passed": all(rules.values()), **rules}


def weight_gates(periods, architecture_passed):
    p = periods.set_index(["variant", "period"])
    v1f = p.loc[("challenger_v1", "full")]; v1h1 = p.loc[("challenger_v1", "2025_h1")]
    v1v = p.loc[("challenger_v1", "validation_2025_plus")]
    out = {}
    for w in WEIGHTS[1:]:
        name = f"alpha_v2_full_a{int(w*100)}"; full = p.loc[(name, "full")]
        h1 = p.loc[(name, "2025_h1")]; val = p.loc[(name, "validation_2025_plus")]
        rules = {"architecture_10pct_passed_first": bool(architecture_passed),
                 "cagr_at_least_15pct": bool(full["cagr"] >= PROJECT_CAGR_HURDLE),
                 "sharpe_at_least_v1": bool(full["sharpe"] >= v1f["sharpe"]),
                 "calmar_at_least_v1": bool(full["calmar"] >= v1f["calmar"]),
                 "max_dd_under_15pct": bool(full["max_drawdown"] >= -0.15),
                 "validation_not_worse_than_v1": bool(val["return"] >= v1v["return"]),
                 "h1_2025_not_over_2pp_worse": bool(h1["return"] >= v1h1["return"] - 0.02),
                 "2022_above_minus_5pct": bool(p.loc[(name, "2022_bear"), "return"] >= -0.05),
                 "2026_profitable": bool(p.loc[(name, "2026_diagnostic"), "return"] > 0)}
        out[name] = {"passed": all(rules.values()), **rules}
    return out


def run(start, end, mc_runs=5000):
    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    data = wf._prepare_data(ALLOC_SYMBOLS, s, e)
    frozen = wf.simulate_portfolio(data, LIVE_SYMBOLS, s, e, 3.0, risk_per_trade=0.01, regime_filter=True)
    baseline = ap.simulate_tactical(data, LIVE_SYMBOLS, s, e)
    if not np.isclose(frozen["final_capital"], baseline["final_capital"], rtol=0, atol=1e-7):
        raise AssertionError("baseline divergiu")
    continuity = ap.simulate_tactical(data, LIVE_SYMBOLS, s, e, cost_aware=True,
        entry_permission=ap.persistent_state_permission(data, LIVE_SYMBOLS))
    alpha_v1 = hv2.simulate_breadth_allocator(data, ALLOC_SYMBOLS, s, e, target_vol=TARGET_VOL,
        top_n=2, min_selected=2, rebalance_days=CADENCE)
    challenger = hv2.combine_rebalanced_sleeves(continuity["equity_curve"], alpha_v1["equity_curve"], 0.10,
        rebalance_days=SLEEVE_REBALANCE_DAYS, transfer_cost=TRANSFER_COST)
    curves = {"baseline": baseline["equity_curve"], "continuity_tactical": continuity["equity_curve"],
              "challenger_v1": challenger}
    logs = []; full_alpha_curve = None
    for variant, flags in VARIANTS.items():
        sim = av2.simulate_alpha_v2(data, ALLOC_SYMBOLS, s, e, rebalance_days=CADENCE,
            target_vol=TARGET_VOL, top_n=2, min_selected=2, **flags)
        curves[f"{variant}_a10"] = hv2.combine_rebalanced_sleeves(continuity["equity_curve"], sim["equity_curve"], 0.10,
            rebalance_days=SLEEVE_REBALANCE_DAYS, transfer_cost=TRANSFER_COST)
        if variant == "alpha_v2_full": full_alpha_curve = sim["equity_curve"]
        if not sim["rebalance_log"].empty:
            x = sim["rebalance_log"].copy(); x["variant"] = variant; logs.append(x)
    if full_alpha_curve is None: raise AssertionError("full Alpha v2 ausente")
    for w in WEIGHTS[1:]:
        curves[f"alpha_v2_full_a{int(w*100)}"] = hv2.combine_rebalanced_sleeves(
            continuity["equity_curve"], full_alpha_curve, w,
            rebalance_days=SLEEVE_REBALANCE_DAYS, transfer_cost=TRANSFER_COST)
    periods = period_table(curves, s, e); arch = architecture_gate(periods); gates = weight_gates(periods, arch["passed"])
    selected = next((n for n in ["alpha_v2_full_a15", "alpha_v2_full_a20", "alpha_v2_full_a25"] if gates[n]["passed"]), None)
    full = pd.DataFrame([{"variant": n, **ap.performance_metrics(c)} for n, c in curves.items()])
    mc_names = ["baseline", "challenger_v1", "alpha_v2_full_a10"] + ([selected] if selected else [])
    mc = {n: dv.block_bootstrap_monte_carlo(curves[n].pct_change().dropna(), runs=mc_runs, seed=42) for n in mc_names}
    full.to_csv("cross_asset_alpha_v2_full.csv", index=False); periods.to_csv("cross_asset_alpha_v2_periods.csv", index=False)
    pd.DataFrame(curves).sort_index().ffill().to_csv("cross_asset_alpha_v2_curves.csv")
    (pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()).to_csv("cross_asset_alpha_v2_rebalances.csv", index=False)
    report = {"start": start, "end": end, "cadence": CADENCE, "target_vol": TARGET_VOL,
              "sleeve_rebalance_days": SLEEVE_REBALANCE_DAYS, "transfer_cost": TRANSFER_COST,
              "weights": list(WEIGHTS), "project_cagr_hurdle": PROJECT_CAGR_HURDLE,
              "architecture_gate": arch, "weight_gates": gates, "selected_candidate": selected,
              "monte_carlo": mc, "guardrails": {"live_changed": False, "spot_only": True, "leverage": 1.0,
              "features_frozen_before_results": True, "ablations_diagnostic_only": True,
              "smallest_passing_weight_wins": True, "2026_is_pristine_holdout": False,
              "promotion_requires_future_paper": True}}
    with open("cross_asset_alpha_v2_report.json", "w") as f: json.dump(report, f, indent=2, default=str)
    print("\nFULL\n", full.to_string(index=False)); print("\nRETURNS\n", periods.pivot(index="variant", columns="period", values="return").to_string())
    print("\nARCHITECTURE\n", json.dumps(arch, indent=2)); print("\nWEIGHT GATES\n", json.dumps(gates, indent=2)); print("\nSELECTED", selected)
    return report


def main():
    p = argparse.ArgumentParser(); p.add_argument("--start", default="2020-08-01"); p.add_argument("--end", default="2026-08-25")
    p.add_argument("--mc-runs", type=int, default=5000); a = p.parse_args(); run(a.start, a.end, a.mc_runs)

if __name__ == "__main__": main()
