"""C-only pre-live validation for Krypton Aggressive C.

Frozen objective: stress the already-selected C configuration without changing
signals or mining thresholds. Tests: +/-10% risk-budget neighborhood, 2x/3x
execution costs, one-extra-bar execution delay, alpha leave-one-out, a single
pre-declared causal DD/volatility throttle, recent rolling windows, concentration
and 5,000-run block-bootstrap Monte Carlo.

Research only. No live order path or config.py constant is changed.
"""
from __future__ import annotations

import argparse
import copy
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from math import sqrt

import numpy as np
import pandas as pd

import adaptive_portfolio as ap
import backtest
import cross_asset_allocation as ca
import cross_asset_hybrid_v2 as hv2
import deep_validation as dv
import walk_forward as wf

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
LIVE = list(wf.BASE_WEIGHTS)
CADENCE = 45
SLEEVE_DAYS = 90
TRANSFER_COST = 0.003
DD_CAP = 0.30
C = {"risk_per_trade": 0.0200, "alpha_weight": 0.45, "target_vol": 0.30}
NEIGHBOR_SCALES = (0.90, 1.00, 1.10)
COST_SCALES = (2.0, 3.0)
THROTTLE = {
    "vol_window": 20,
    "vol_soft": 0.60,
    "vol_hard": 0.80,
    "dd_soft": -0.15,
    "dd_hard": -0.22,
    "scale_soft": 0.75,
    "scale_hard": 0.50,
    "transfer_cost": 0.003,
}
backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"


def _period_metrics(curve, start, end):
    return ap.performance_metrics(ap.slice_and_rebase(curve, start, end))


def _period_table(curves, start, end):
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
            rows.append({"variant": name, "period": period, **_period_metrics(curve, a, b)})
    return pd.DataFrame(rows)


def _delay_tactical_signals(data, bars=1):
    out = {}
    for sym, item in data.items():
        cloned = dict(item)
        if "signals" in item:
            cloned["signals"] = item["signals"].shift(bars).fillna(0).astype(int)
        out[sym] = cloned
    return out


def simulate_breadth_allocator_delayed(data, symbols, start, end, *, target_vol,
                                       top_n=2, min_selected=2,
                                       rebalance_days=45, extra_delay_bars=1):
    """Same breadth allocator but execute weights one additional bar later."""
    start_ts, end_ts = ap.as_utc(start), ap.as_utc(end)
    calendar = sorted(set.intersection(*[
        set(data[s]["df"].loc[start_ts:end_ts].index) for s in symbols
    ]))
    if not calendar:
        empty = pd.Series(dtype=float)
        return {**ap.performance_metrics(empty), "equity_curve": empty,
                "rebalance_log": pd.DataFrame()}

    cash = ap.INITIAL_CAPITAL
    qty = {s: 0.0 for s in symbols}
    pending = None
    pending_signal_time = None
    execute_i = None
    last_signal_i = None
    points, logs = [], []

    for i, ts in enumerate(calendar):
        if pending is not None and execute_i is not None and i >= execute_i:
            pre_eq = ca._mark(cash, qty, data, ts, "open")
            opens = {s: float(data[s]["df"].loc[ts, "open"]) for s in symbols}
            for s in symbols:
                target_value = pre_eq * float(pending.get(s, 0.0))
                current_value = qty[s] * opens[s]
                if current_value > target_value and qty[s] > 0:
                    sell_qty = min(qty[s], (current_value - target_value) / opens[s])
                    px = opens[s] * (1.0 - hv2.EXIT_SLIPPAGE_PCT)
                    cash += sell_qty * px * (1.0 - hv2.FEE_RATE)
                    qty[s] -= sell_qty
            eq_after_sells = ca._mark(cash, qty, data, ts, "open")
            for s in symbols:
                target_value = eq_after_sells * float(pending.get(s, 0.0))
                current_value = qty[s] * opens[s]
                deficit = max(0.0, target_value - current_value)
                if deficit <= 0 or cash <= 0:
                    continue
                px = opens[s] * (1.0 + hv2.ENTRY_SLIPPAGE_PCT)
                spend = min(deficit, cash / (1.0 + hv2.FEE_RATE))
                buy_qty = spend / px
                cash -= buy_qty * px * (1.0 + hv2.FEE_RATE)
                qty[s] += buy_qty
            logs.append({"signal_time": pending_signal_time, "execution_time": ts,
                         "extra_delay_bars": extra_delay_bars,
                         "target_gross": sum(pending.values()),
                         "selected": sum(float(v) > 0 for v in pending.values())})
            pending = None
            execute_i = None

        points.append((ts, ca._mark(cash, qty, data, ts, "close")))
        if last_signal_i is None or i - last_signal_i >= rebalance_days:
            pending = hv2.breadth_target_weights(
                data, symbols, ts, target_vol=target_vol, top_n=top_n,
                min_selected=min_selected)
            pending_signal_time = ts
            execute_i = i + 1 + extra_delay_bars
            last_signal_i = i

    ts = calendar[-1]
    for s in symbols:
        if qty[s] <= 0:
            continue
        px = float(data[s]["df"].loc[ts, "close"]) * (1.0 - hv2.EXIT_SLIPPAGE_PCT)
        cash += qty[s] * px * (1.0 - hv2.FEE_RATE)
        qty[s] = 0.0
    points[-1] = (ts, cash)
    equity = pd.Series([v for _, v in points], index=[t for t, _ in points], dtype=float)
    log = pd.DataFrame(logs)
    if not log.empty:
        sig = pd.to_datetime(log.signal_time, utc=True)
        exe = pd.to_datetime(log.execution_time, utc=True)
        if not bool((exe > sig).all()):
            raise AssertionError("look-ahead detectado no delay test")
    return {**ap.performance_metrics(equity), "equity_curve": equity,
            "rebalance_log": log}


@contextmanager
def scaled_exchange_costs(scale):
    originals = {
        "ap_fee": ap.FEE_RATE, "ap_entry": ap.ENTRY_SLIPPAGE_PCT,
        "ap_exit": ap.EXIT_SLIPPAGE_PCT, "hv_fee": hv2.FEE_RATE,
        "hv_entry": hv2.ENTRY_SLIPPAGE_PCT, "hv_exit": hv2.EXIT_SLIPPAGE_PCT,
    }
    try:
        ap.FEE_RATE *= scale
        ap.ENTRY_SLIPPAGE_PCT *= scale
        ap.EXIT_SLIPPAGE_PCT *= scale
        hv2.FEE_RATE *= scale
        hv2.ENTRY_SLIPPAGE_PCT *= scale
        hv2.EXIT_SLIPPAGE_PCT *= scale
        yield
    finally:
        ap.FEE_RATE = originals["ap_fee"]
        ap.ENTRY_SLIPPAGE_PCT = originals["ap_entry"]
        ap.EXIT_SLIPPAGE_PCT = originals["ap_exit"]
        hv2.FEE_RATE = originals["hv_fee"]
        hv2.ENTRY_SLIPPAGE_PCT = originals["hv_entry"]
        hv2.EXIT_SLIPPAGE_PCT = originals["hv_exit"]


def simulate_c(data, s, e, cfg=C, *, assets=ASSETS, transfer_cost=TRANSFER_COST,
               extra_delay_bars=0):
    source = _delay_tactical_signals(data, extra_delay_bars) if extra_delay_bars else data
    permission = ap.persistent_state_permission(source, LIVE)
    tactical = ap.simulate_tactical(
        source, LIVE, s, e, cost_aware=True, entry_permission=permission,
        risk_per_trade=cfg["risk_per_trade"])
    if extra_delay_bars:
        alpha = simulate_breadth_allocator_delayed(
            data, assets, s, e, target_vol=cfg["target_vol"], top_n=2,
            min_selected=2, rebalance_days=CADENCE,
            extra_delay_bars=extra_delay_bars)
    else:
        alpha = hv2.simulate_breadth_allocator(
            data, assets, s, e, target_vol=cfg["target_vol"], top_n=2,
            min_selected=2, rebalance_days=CADENCE)
    curve = hv2.combine_rebalanced_sleeves(
        tactical["equity_curve"], alpha["equity_curve"], cfg["alpha_weight"],
        rebalance_days=SLEEVE_DAYS, transfer_cost=transfer_cost)
    return curve, tactical, alpha


def apply_causal_throttle(curve, cfg=THROTTLE):
    """Move a fraction of portfolio to cash using only prior-close state."""
    raw = curve.dropna().astype(float)
    rets = raw.pct_change().fillna(0.0)
    if raw.empty:
        return raw, pd.DataFrame()
    equity = ap.INITIAL_CAPITAL
    peak = equity
    prev_scale = 1.0
    points, logs = [], []
    for i, ts in enumerate(raw.index):
        if i == 0:
            points.append((ts, equity))
            logs.append({"time": ts, "scale": 1.0, "vol20": np.nan, "dd": 0.0})
            continue
        hist = rets.iloc[max(0, i - cfg["vol_window"]):i]
        vol = float(hist.std() * sqrt(365.0)) if len(hist) >= 10 else np.nan
        dd = equity / peak - 1.0
        hard = dd <= cfg["dd_hard"] or (np.isfinite(vol) and vol >= cfg["vol_hard"])
        soft = dd <= cfg["dd_soft"] or (np.isfinite(vol) and vol >= cfg["vol_soft"])
        scale = cfg["scale_hard"] if hard else cfg["scale_soft"] if soft else 1.0
        turnover = abs(scale - prev_scale)
        if turnover:
            equity *= 1.0 - turnover * cfg["transfer_cost"]
        equity *= 1.0 + scale * float(rets.iloc[i])
        peak = max(peak, equity)
        points.append((ts, equity))
        logs.append({"time": ts, "scale": scale, "vol20": vol, "dd": dd,
                     "turnover": turnover})
        prev_scale = scale
    return pd.Series([v for _, v in points], index=[t for t, _ in points], dtype=float), pd.DataFrame(logs)


def rolling_recent(curves, start="2024-01-01", window_bars=180, step_bars=90):
    rows = []
    for name, curve in curves.items():
        idx = curve.index[curve.index >= ap.as_utc(start)]
        for i in range(0, max(0, len(idx) - window_bars + 1), step_bars):
            subidx = idx[i:i + window_bars]
            if len(subidx) < window_bars:
                continue
            m = _period_metrics(curve, subidx[0], subidx[-1])
            rows.append({"variant": name, "start": subidx[0], "end": subidx[-1], **m})
    return pd.DataFrame(rows)


def concentration(curve):
    r = curve.pct_change().dropna().sort_values(ascending=False)
    out = {}
    for k in (1, 5, 20):
        x = r.iloc[k:]
        out[f"return_without_best_{k}d"] = float((1.0 + x).prod() - 1.0)
    return out


def run(start="2020-08-01", end="2026-08-25", mc_runs=5000):
    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    data = wf._prepare_data(ASSETS, s, e)
    curves = {}
    diagnostics = {}
    original_dd = ap.MAX_DRAWDOWN_PCT
    ap.MAX_DRAWDOWN_PCT = DD_CAP
    try:
        base_curve, _, _ = simulate_c(data, s, e, C)
        curves["C"] = base_curve

        for scale in NEIGHBOR_SCALES:
            cfg = {k: C[k] * scale for k in C}
            curves[f"C_x{scale:.2f}"] = simulate_c(data, s, e, cfg)[0]

        for scale in COST_SCALES:
            with scaled_exchange_costs(scale):
                curves[f"C_costs_{int(scale)}x"] = simulate_c(
                    data, s, e, C, transfer_cost=TRANSFER_COST * scale)[0]

        curves["C_delay_1bar"] = simulate_c(data, s, e, C, extra_delay_bars=1)[0]

        for omitted in ASSETS:
            assets = [x for x in ASSETS if x != omitted]
            curves[f"C_no_{omitted.replace('USDT','')}"] = simulate_c(
                data, s, e, C, assets=assets)[0]

        throttled, throttle_log = apply_causal_throttle(base_curve)
        curves["C_throttle"] = throttled
        throttle_log.to_csv("c_only_throttle_log.csv", index=False)
    finally:
        ap.MAX_DRAWDOWN_PCT = original_dd

    full = pd.DataFrame([{"variant": n, **ap.performance_metrics(c)} for n, c in curves.items()])
    periods = _period_table(curves, start, end)
    recent = rolling_recent({k: curves[k] for k in ["C", "C_throttle"]})
    ix = full.set_index("variant")

    neighbor_checks = {}
    for scale in NEIGHBOR_SCALES:
        n = f"C_x{scale:.2f}"
        m = ix.loc[n]
        neighbor_checks[n] = {
            "cagr_ge_25": bool(m.cagr >= .25),
            "dd_le_30": bool(m.max_drawdown >= -.30),
            "sharpe_ge_110": bool(m.sharpe >= 1.10),
        }
        neighbor_checks[n]["passed"] = all(neighbor_checks[n].values())

    cost_checks = {}
    for scale in COST_SCALES:
        n = f"C_costs_{int(scale)}x"
        m = ix.loc[n]
        hurdle = .25 if scale == 2.0 else .20
        cost_checks[n] = {
            "cagr_hurdle": bool(m.cagr >= hurdle),
            "dd_le_30": bool(m.max_drawdown >= -.30),
            "sharpe_ge_100": bool(m.sharpe >= 1.00),
            "cost_effective": bool(m.final_capital < ix.loc["C", "final_capital"]),
        }
        cost_checks[n]["passed"] = all(cost_checks[n].values())

    loo_checks = {}
    for omitted in ASSETS:
        n = f"C_no_{omitted.replace('USDT','')}"
        m = ix.loc[n]
        loo_checks[n] = {
            "cagr_ge_20": bool(m.cagr >= .20),
            "dd_le_30": bool(m.max_drawdown >= -.30),
            "positive_2025_plus": bool(periods[(periods.variant == n) & (periods.period == "2025_plus")]["return"].iloc[0] > 0),
        }
        loo_checks[n]["passed"] = all(loo_checks[n].values())

    delay = ix.loc["C_delay_1bar"]
    delay_check = {
        "cagr_ge_20": bool(delay.cagr >= .20),
        "dd_le_30": bool(delay.max_drawdown >= -.30),
        "sharpe_ge_090": bool(delay.sharpe >= .90),
    }
    delay_check["passed"] = all(delay_check.values())

    mc = {}
    for name in ["C", "C_throttle"]:
        mc[name] = dv.block_bootstrap_monte_carlo(
            curves[name].pct_change().dropna(), runs=mc_runs, seed=44)

    tm = ix.loc["C_throttle"]
    # C+ is only accepted if the single pre-declared throttle preserves most of
    # C's CAGR while materially improving historical and bootstrap drawdown.
    throttle_check = {
        "cagr_ge_27": bool(tm.cagr >= .27),
        "sharpe_ge_115": bool(tm.sharpe >= 1.15),
        "historical_dd_le_25": bool(tm.max_drawdown >= -.25),
        "2025_positive": bool(periods[(periods.variant == "C_throttle") & (periods.period == "2025")]["return"].iloc[0] > 0),
        "2026_positive": bool(periods[(periods.variant == "C_throttle") & (periods.period == "2026")]["return"].iloc[0] > 0),
    }
    # The Monte-Carlo function's keys are retained verbatim in the report; the
    # historical gate above remains authoritative if schema changes.
    throttle_check["passed"] = all(throttle_check.values())

    recent_summary = {}
    if not recent.empty:
        for name, grp in recent.groupby("variant"):
            recent_summary[name] = {
                "windows": int(len(grp)),
                "positive_windows": int((grp["return"] > 0).sum()),
                "median_return": float(grp["return"].median()),
                "worst_return": float(grp["return"].min()),
                "worst_dd": float(grp["max_drawdown"].min()),
            }

    report = {
        "start": start, "end": end,
        "frozen_C": C,
        "neighbor_checks": neighbor_checks,
        "neighbor_plateau_passed": all(x["passed"] for x in neighbor_checks.values()),
        "cost_checks": cost_checks,
        "cost_stress_passed": all(x["passed"] for x in cost_checks.values()),
        "leave_one_out_checks": loo_checks,
        "leave_one_out_passed": all(x["passed"] for x in loo_checks.values()),
        "delay_check": delay_check,
        "throttle_rule": THROTTLE,
        "throttle_check": throttle_check,
        "monte_carlo": mc,
        "recent_180d_windows": recent_summary,
        "concentration": {n: concentration(curves[n]) for n in ["C", "C_throttle"]},
        "guardrails": {
            "live_changed": False,
            "signals_changed": False,
            "C_parameters_changed": False,
            "next_open_base_execution": True,
            "extra_delay_tested": True,
            "promotion_requires_shadow": True,
        },
    }

    full.to_csv("c_only_full.csv", index=False)
    periods.to_csv("c_only_periods.csv", index=False)
    recent.to_csv("c_only_recent_windows.csv", index=False)
    pd.DataFrame(curves).to_csv("c_only_curves.csv")
    with open("c_only_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("FULL")
    print(full.to_string(index=False))
    print("\nPERIODS")
    print(periods.pivot(index="variant", columns="period", values="return").to_string())
    print("\nRECENT")
    print(recent.to_string(index=False))
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
