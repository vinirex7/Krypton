"""End-to-end research pipeline for the seven adaptive Krypton experiments."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import adaptive_portfolio as ap
import backtest
import deep_validation as dv
import range_grid as rg
import walk_forward as wf
from adaptive_ml import (
    audit_dqn_reward_timing,
    probability_permission,
    walk_forward_xgboost_probabilities,
)

SYMBOLS = list(wf.BASE_WEIGHTS)

# GitHub-hosted runners are geo-blocked by api.binance.com.  This is Binance's
# official public market-data endpoint and is already used by the validated
# forensic CI entrypoints.
backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"


def _clean_metrics(metrics: dict) -> dict:
    return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items() if k not in {"equity_curve", "trade_log"}}


def _period_rows(curves: dict[str, pd.Series], start, end) -> list[dict]:
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
    for name, curve in curves.items():
        for period, (p_start, p_end) in periods.items():
            sliced = ap.slice_and_rebase(curve, p_start, p_end)
            rows.append({"variant": name, "period": period, **ap.performance_metrics(sliced)})
    return rows


def _select_candidate(periods: pd.DataFrame) -> tuple[str | None, dict]:
    """Pre-declared, risk-aware promotion gate; 2026 is diagnostic, not pristine."""
    pivot = periods.set_index(["variant", "period"])
    baseline_full = pivot.loc[("baseline", "full")]
    baseline_val = pivot.loc[("baseline", "validation_2025_plus")]
    baseline_h1 = pivot.loc[("baseline", "2025_h1")]
    eligible = []
    checks = {}
    for variant in periods["variant"].unique():
        if variant == "baseline":
            continue
        full = pivot.loc[(variant, "full")]
        dev = pivot.loc[(variant, "development")]
        val = pivot.loc[(variant, "validation_2025_plus")]
        bear = pivot.loc[(variant, "2022_bear")]
        h1 = pivot.loc[(variant, "2025_h1")]
        diag26 = pivot.loc[(variant, "2026_diagnostic")]
        rules = {
            "dev_profitable": bool(dev["return"] > 0),
            "full_cagr_15pct_better": bool(full["cagr"] > baseline_full["cagr"] * 1.15),
            "full_calmar_better": bool(full["calmar"] > baseline_full["calmar"]),
            "validation_beats_baseline": bool(val["return"] > baseline_val["return"]),
            "max_dd_under_15pct": bool(full["max_drawdown"] >= -0.15),
            "bear_2022_above_minus_5pct": bool(bear["return"] >= -0.05),
            "h1_2025_not_over_2pp_worse": bool(h1["return"] >= baseline_h1["return"] - 0.02),
            "diagnostic_2026_profitable": bool(diag26["return"] > 0),
        }
        passed = all(rules.values())
        checks[variant] = {"passed": passed, **rules}
        if passed:
            eligible.append((float(dev["calmar"]), float(dev["cagr"]), variant))
    winner = max(eligible)[2] if eligible else None
    return winner, checks


def run(start, end, mc_runs: int = 2000) -> dict:
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    data = wf._prepare_data(SYMBOLS, start_dt, end_dt)

    frozen = wf.simulate_portfolio(
        data, SYMBOLS, start_dt, end_dt, 3.0,
        risk_per_trade=0.01, regime_filter=True,
    )
    baseline = ap.simulate_tactical(data, SYMBOLS, start_dt, end_dt)
    if not np.isclose(frozen["final_capital"], baseline["final_capital"], rtol=0, atol=1e-7):
        raise AssertionError(
            f"simulador adaptativo divergiu da baseline: {baseline['final_capital']} vs {frozen['final_capital']}"
        )

    cost_aware = ap.simulate_tactical(data, SYMBOLS, start_dt, end_dt, cost_aware=True)
    runner = ap.simulate_tactical(data, SYMBOLS, start_dt, end_dt, runner=True, cost_aware=True)
    core = ap.simulate_btc_core(data, start_dt, end_dt)

    hourly = rg.download_ohlcv("BTCUSDT", start_dt - timedelta(days=2), end_dt, "1h")
    grid = rg.simulate_range_grid(hourly, data["BTCUSDT"]["df"], start_dt, end_dt)

    xgb = walk_forward_xgboost_probabilities(data, SYMBOLS, start_dt, end_dt)
    xgb_tactical = ap.simulate_tactical(
        data, SYMBOLS, start_dt, end_dt, cost_aware=True,
        entry_permission=probability_permission(xgb.probabilities, threshold=0.55),
    )

    curves = {
        "baseline": baseline["equity_curve"],
        "cost_aware_tactical": cost_aware["equity_curve"],
        "runner_tactical": runner["equity_curve"],
        "core40_sat60": ap.combine_sleeves(
            {"core": core["equity_curve"], "sat": cost_aware["equity_curve"]},
            {"core": 0.40, "sat": 0.60},
        ),
        "sat90_grid10": ap.combine_sleeves(
            {"sat": cost_aware["equity_curve"], "grid": grid["equity_curve"]},
            {"sat": 0.90, "grid": 0.10},
        ),
        "core40_sat50_grid10": ap.combine_sleeves(
            {"core": core["equity_curve"], "sat": cost_aware["equity_curve"],
             "grid": grid["equity_curve"]},
            {"core": 0.40, "sat": 0.50, "grid": 0.10},
        ),
        "core40_runner50_grid10": ap.combine_sleeves(
            {"core": core["equity_curve"], "runner": runner["equity_curve"],
             "grid": grid["equity_curve"]},
            {"core": 0.40, "runner": 0.50, "grid": 0.10},
        ),
        "core40_xgb50_grid10": ap.combine_sleeves(
            {"core": core["equity_curve"], "xgb": xgb_tactical["equity_curve"],
             "grid": grid["equity_curve"]},
            {"core": 0.40, "xgb": 0.50, "grid": 0.10},
        ),
    }

    periods = pd.DataFrame(_period_rows(curves, start_dt, end_dt))
    winner, checks = _select_candidate(periods)
    selected_curve = curves[winner] if winner else curves["baseline"]
    mc = dv.block_bootstrap_monte_carlo(selected_curve.pct_change().dropna(), runs=mc_runs, seed=42)
    dqn_audit = audit_dqn_reward_timing(data["BTCUSDT"]["df"].loc[as_utc(start):as_utc(end)])

    # Auditable outputs.
    periods.to_csv("adaptive_period_results.csv", index=False)
    pd.DataFrame([{"variant": name, **ap.performance_metrics(curve)} for name, curve in curves.items()]) \
        .to_csv("adaptive_full_results.csv", index=False)
    pd.DataFrame({name: curve for name, curve in curves.items()}).sort_index().ffill() \
        .to_csv("adaptive_equity_curves.csv")
    baseline["trade_log"].to_csv("adaptive_baseline_trades.csv", index=False)
    runner["trade_log"].to_csv("adaptive_runner_trades.csv", index=False)
    grid["trade_log"].to_csv("adaptive_grid_fills.csv", index=False)
    xgb.folds.to_csv("adaptive_xgb_folds.csv", index=False)

    report = {
        "start": start,
        "end": end,
        "baseline_reproduction_error": float(abs(frozen["final_capital"] - baseline["final_capital"])),
        "component_metrics": {
            "baseline": _clean_metrics(baseline),
            "cost_aware": _clean_metrics(cost_aware),
            "runner": _clean_metrics(runner),
            "core": _clean_metrics(core),
            "grid": _clean_metrics(grid),
            "xgb_tactical": _clean_metrics(xgb_tactical),
        },
        "selected_candidate": winner,
        "promotion_checks": checks,
        "monte_carlo": _clean_metrics(mc),
        "dqn_reward_timing_audit": {
            "same_bar": _clean_metrics(dqn_audit["same_bar"]),
            "lagged_next_bar": _clean_metrics(dqn_audit["lagged_next_bar"]),
            "action_counts": dqn_audit["action_counts"],
        },
        "guardrails": {
            "live_changed": False,
            "spot_only": True,
            "leverage": 1.0,
            "2026_is_pristine_holdout": False,
            "promotion_requires_future_paper": True,
        },
    }
    with open("adaptive_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report


def main():
    parser = argparse.ArgumentParser(description="Krypton adaptive portfolio research")
    parser.add_argument("--start", default="2020-08-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--mc-runs", type=int, default=2000)
    args = parser.parse_args()
    report = run(args.start, args.end, args.mc_runs)
    full = pd.read_csv("adaptive_full_results.csv")
    periods = pd.read_csv("adaptive_period_results.csv")
    print("\nADAPTIVE FULL RESULTS")
    print(full.to_string(index=False))
    print("\nPERIOD RETURNS")
    print(periods.pivot(index="variant", columns="period", values="return").to_string())
    print("\nSELECTED CANDIDATE:", report["selected_candidate"] or "NONE")
    print("PROMOTION CHECKS")
    print(json.dumps(report["promotion_checks"], indent=2))
    print("\nDQN REWARD-TIMING AUDIT")
    print(json.dumps(report["dqn_reward_timing_audit"], indent=2))
    print("\nMONTE CARLO")
    print(json.dumps(report["monte_carlo"], indent=2))


if __name__ == "__main__":
    main()
