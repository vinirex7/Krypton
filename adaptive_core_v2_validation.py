"""Validate a slope-confirmed BTC core combined with the continuity tactical sleeve."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import adaptive_core_v2 as corev2
import adaptive_portfolio as ap
import backtest
import deep_validation as dv
import walk_forward as wf


SYMBOLS = list(wf.BASE_WEIGHTS)
WEIGHTS = (0.10, 0.20, 0.30)
backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"


def period_rows(curves: dict[str, pd.Series], start, end) -> pd.DataFrame:
    periods = {
        "development": (start, "2024-12-31"),
        "2022_bear": ("2022-01-01", "2022-12-31"),
        "2025_h1": ("2025-01-01", "2025-06-30"),
        "2025_h2": ("2025-07-01", "2025-12-31"),
        "validation_2025_plus": ("2025-01-01", end),
        "2026_diagnostic": ("2026-01-01", end),
        "full": (start, end),
    }
    rows = []
    for variant, curve in curves.items():
        for period, (p_start, p_end) in periods.items():
            sliced = ap.slice_and_rebase(curve, p_start, p_end)
            rows.append({"variant": variant, "period": period, **ap.performance_metrics(sliced)})
    return pd.DataFrame(rows)


def promotion_checks(periods: pd.DataFrame) -> dict:
    pivot = periods.set_index(["variant", "period"])
    baseline_full = pivot.loc[("baseline", "full")]
    baseline_val = pivot.loc[("baseline", "validation_2025_plus")]
    baseline_h1 = pivot.loc[("baseline", "2025_h1")]
    checks = {}
    for variant in periods["variant"].unique():
        if variant in {"baseline", "continuity_tactical", "trend_core"}:
            continue
        full = pivot.loc[(variant, "full")]
        dev = pivot.loc[(variant, "development")]
        val = pivot.loc[(variant, "validation_2025_plus")]
        bear = pivot.loc[(variant, "2022_bear")]
        h1 = pivot.loc[(variant, "2025_h1")]
        diag = pivot.loc[(variant, "2026_diagnostic")]
        rules = {
            "dev_profitable": bool(dev["return"] > 0),
            "full_cagr_15pct_better": bool(full["cagr"] > baseline_full["cagr"] * 1.15),
            "full_calmar_better": bool(full["calmar"] > baseline_full["calmar"]),
            "validation_beats_baseline": bool(val["return"] > baseline_val["return"]),
            "max_dd_under_15pct": bool(full["max_drawdown"] >= -0.15),
            "bear_2022_above_minus_5pct": bool(bear["return"] >= -0.05),
            "h1_2025_not_over_2pp_worse": bool(h1["return"] >= baseline_h1["return"] - 0.02),
            "diagnostic_2026_profitable": bool(diag["return"] > 0),
        }
        checks[variant] = {"passed": all(rules.values()), **rules}
    return checks


def run(start: str, end: str, mc_runs: int = 5000) -> dict:
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    data = wf._prepare_data(SYMBOLS, start_dt, end_dt)

    frozen = wf.simulate_portfolio(data, SYMBOLS, start_dt, end_dt, 3.0,
                                   risk_per_trade=0.01, regime_filter=True)
    baseline = ap.simulate_tactical(data, SYMBOLS, start_dt, end_dt)
    if not np.isclose(frozen["final_capital"], baseline["final_capital"], rtol=0, atol=1e-7):
        raise AssertionError("baseline divergiu do simulador congelado")

    continuity = ap.simulate_tactical(
        data, SYMBOLS, start_dt, end_dt, cost_aware=True,
        entry_permission=ap.persistent_state_permission(data, SYMBOLS),
    )
    trend_core = corev2.simulate_btc_trend_core(data, start_dt, end_dt, slope_lookback=20)

    curves: dict[str, pd.Series] = {
        "baseline": baseline["equity_curve"],
        "continuity_tactical": continuity["equity_curve"],
        "trend_core": trend_core["equity_curve"],
    }
    for weight in WEIGHTS:
        core_pct = int(round(weight * 100))
        sat_pct = 100 - core_pct
        name = f"trendcore{core_pct}_continuity{sat_pct}"
        curves[name] = ap.combine_sleeves(
            {"core": trend_core["equity_curve"], "sat": continuity["equity_curve"]},
            {"core": weight, "sat": 1.0 - weight},
        )

    periods = period_rows(curves, start_dt, end_dt)
    checks = promotion_checks(periods)
    passing = [name for name, rule in checks.items() if rule["passed"]]

    # Conservative selection: smallest passing core weight.  Neighboring
    # sensitivity is reported separately; no parameter is optimized on return.
    selected = None
    for weight in WEIGHTS:
        name = f"trendcore{int(round(weight * 100))}_continuity{100-int(round(weight * 100))}"
        if name in passing:
            selected = name
            break

    stability = {
        "variants_tested": len(WEIGHTS),
        "pass_count": len(passing),
        "at_least_two_weights_pass": len(passing) >= 2,
        "passing_variants": passing,
    }

    monte_carlo = {}
    for name in ["baseline", "continuity_tactical", *[f"trendcore{int(w*100)}_continuity{100-int(w*100)}" for w in WEIGHTS]]:
        returns = curves[name].pct_change().dropna()
        monte_carlo[name] = dv.block_bootstrap_monte_carlo(returns, runs=mc_runs, seed=42)

    full = pd.DataFrame([{"variant": name, **ap.performance_metrics(curve)} for name, curve in curves.items()])
    full.to_csv("adaptive_core_v2_full_results.csv", index=False)
    periods.to_csv("adaptive_core_v2_period_results.csv", index=False)
    pd.DataFrame(curves).sort_index().ffill().to_csv("adaptive_core_v2_equity_curves.csv")
    trend_core["trade_log"].to_csv("adaptive_core_v2_trades.csv", index=False)

    report = {
        "start": start,
        "end": end,
        "hypothesis": "BTC core only when close>SMA200 and SMA200 has positive 20d slope",
        "weights_tested": list(WEIGHTS),
        "selected_candidate": selected,
        "promotion_checks": checks,
        "stability": stability,
        "monte_carlo": monte_carlo,
        "guardrails": {
            "live_changed": False,
            "spot_only": True,
            "leverage": 1.0,
            "2026_is_pristine_holdout": False,
            "promotion_requires_future_paper": True,
        },
    }
    with open("adaptive_core_v2_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("\nCORE V2 FULL RESULTS")
    print(full.to_string(index=False))
    print("\nCORE V2 PERIOD RETURNS")
    print(periods.pivot(index="variant", columns="period", values="return").to_string())
    print("\nPROMOTION CHECKS")
    print(json.dumps(checks, indent=2))
    print("\nSTABILITY")
    print(json.dumps(stability, indent=2))
    print("\nSELECTED:", selected or "NONE")
    print("\nMONTE CARLO")
    print(json.dumps(monte_carlo, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-08-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--mc-runs", type=int, default=5000)
    args = parser.parse_args()
    run(args.start, args.end, args.mc_runs)


if __name__ == "__main__":
    main()
