"""Validate cross-asset allocation v1 without changing Krypton live."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import adaptive_core_v3 as corev3
import adaptive_portfolio as ap
import backtest
import cross_asset_allocation as ca
import deep_validation as dv
import walk_forward as wf

LIVE_SYMBOLS = list(wf.BASE_WEIGHTS)
ALLOC_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
TARGET_VOLS = (0.15, 0.20, 0.25)
REBALANCE_DAYS = (7, 30)
backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"


def periods_for(curves, start, end):
    windows = {
        "development": (start, "2024-12-31"),
        "2021": ("2021-01-01", "2021-12-31"),
        "2022_bear": ("2022-01-01", "2022-12-31"),
        "2023_2024": ("2023-01-01", "2024-12-31"),
        "2025_h1": ("2025-01-01", "2025-06-30"),
        "2025_h2": ("2025-07-01", "2025-12-31"),
        "validation_2025_plus": ("2025-01-01", end),
        "2026_diagnostic": ("2026-01-01", end),
        "full": (start, end),
    }
    rows = []
    for name, curve in curves.items():
        for period, (a, b) in windows.items():
            rows.append({
                "variant": name, "period": period,
                **ap.performance_metrics(ap.slice_and_rebase(curve, a, b)),
            })
    return pd.DataFrame(rows)


def checks_for(periods):
    p = periods.set_index(["variant", "period"])
    base_full = p.loc[("baseline", "full")]
    base_val = p.loc[("baseline", "validation_2025_plus")]
    base_h1 = p.loc[("baseline", "2025_h1")]
    out = {}
    for name in periods.variant.unique():
        if not name.startswith("alloc_"):
            continue
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


def run(start, end, mc_runs=5000):
    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    data = wf._prepare_data(ALLOC_SYMBOLS, s, e)

    frozen = wf.simulate_portfolio(
        data, LIVE_SYMBOLS, s, e, 3.0, risk_per_trade=0.01, regime_filter=True
    )
    baseline = ap.simulate_tactical(data, LIVE_SYMBOLS, s, e)
    if not np.isclose(frozen["final_capital"], baseline["final_capital"], rtol=0, atol=1e-7):
        raise AssertionError("baseline divergiu do motor congelado")

    continuity = ap.simulate_tactical(
        data, LIVE_SYMBOLS, s, e, cost_aware=True,
        entry_permission=ap.persistent_state_permission(data, LIVE_SYMBOLS),
    )

    curves = {
        "baseline": baseline["equity_curve"],
        "continuity_tactical": continuity["equity_curve"],
        "btc_buy_hold": ca.simulate_buy_hold(data, ["BTCUSDT"], s, e),
        "equal_weight_4": ca.simulate_buy_hold(data, ALLOC_SYMBOLS, s, e),
    }

    # Previously rejected comparator, reproduced without changing its rules.
    core = corev3.simulate_btc_trend_core(data, s, e, entry_atr_mult=1.0, slope_lookback=20)
    curves["rejected_corev3_10_cont90"] = ap.combine_sleeves(
        {"core": core["equity_curve"], "sat": continuity["equity_curve"]},
        {"core": 0.10, "sat": 0.90},
    )

    allocation_logs = []
    allocation_states = []
    for reb in REBALANCE_DAYS:
        for tv in TARGET_VOLS:
            name = f"alloc_tv{int(tv*100)}_r{reb}"
            result = ca.simulate_allocator(
                data, ALLOC_SYMBOLS, s, e,
                target_vol=tv, top_n=2, rebalance_days=reb,
            )
            curves[name] = result["equity_curve"]
            if not result["rebalance_log"].empty:
                x = result["rebalance_log"].copy()
                x["variant"] = name
                allocation_logs.append(x)
            if not result["allocations"].empty:
                x = result["allocations"].copy()
                x["variant"] = name
                x["time"] = x.index
                allocation_states.append(x.reset_index(drop=True))

    periods = periods_for(curves, s, e)
    checks = checks_for(periods)

    # Robust promotion: same target-vol must pass at both pre-declared cadences.
    robust = []
    for tv in TARGET_VOLS:
        names = [f"alloc_tv{int(tv*100)}_r{reb}" for reb in REBALANCE_DAYS]
        if all(checks[n]["passed"] for n in names):
            robust.append(tv)
    selected_tv = min(robust) if robust else None
    selected = None if selected_tv is None else f"alloc_tv{int(selected_tv*100)}_r30"

    full = pd.DataFrame([
        {"variant": name, **ap.performance_metrics(curve)}
        for name, curve in curves.items()
    ])

    mc = {}
    for name in ["baseline", "continuity_tactical"] + [n for n in checks]:
        returns = curves[name].pct_change().dropna()
        mc[name] = dv.block_bootstrap_monte_carlo(
            returns, runs=mc_runs, seed=42
        )

    full.to_csv("cross_asset_allocation_full_results.csv", index=False)
    periods.to_csv("cross_asset_allocation_period_results.csv", index=False)
    pd.DataFrame(curves).sort_index().ffill().to_csv(
        "cross_asset_allocation_equity_curves.csv"
    )
    (pd.concat(allocation_logs, ignore_index=True) if allocation_logs else pd.DataFrame()).to_csv(
        "cross_asset_allocation_rebalances.csv", index=False
    )
    (pd.concat(allocation_states, ignore_index=True) if allocation_states else pd.DataFrame()).to_csv(
        "cross_asset_allocation_states.csv", index=False
    )

    report = {
        "start": start,
        "end": end,
        "assets": ALLOC_SYMBOLS,
        "momentum_windows": [30, 90, 180],
        "trend_filter": "asset close > own SMA200",
        "selection": "top 2 positive average momentum",
        "sizing": "inverse 20d realized vol, scaled by 60d covariance portfolio vol",
        "target_vols": list(TARGET_VOLS),
        "rebalance_days": list(REBALANCE_DAYS),
        "selected_candidate": selected,
        "robust_target_vols": robust,
        "promotion_checks": checks,
        "monte_carlo": mc,
        "guardrails": {
            "live_changed": False,
            "spot_only": True,
            "leverage": 1.0,
            "cash_allowed": True,
            "execution": "next daily open",
            "fees_and_slippage": True,
            "2025_plus_is_not_pristine_holdout": True,
            "promotion_requires_future_paper": True,
        },
    }
    with open("cross_asset_allocation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\nCROSS-ASSET FULL RESULTS\n", full.to_string(index=False))
    print("\nPERIOD RETURNS\n",
          periods.pivot(index="variant", columns="period", values="return").to_string())
    print("\nCHECKS\n", json.dumps(checks, indent=2))
    print("\nSELECTED", selected or "NONE")
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
