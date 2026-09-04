"""Cross-asset hybrid v2: breadth-confirmed alpha sleeve plus tactical continuity."""
from __future__ import annotations

import numpy as np
import pandas as pd

import adaptive_portfolio as ap
import cross_asset_allocation as ca
from config import ENTRY_SLIPPAGE_PCT, EXIT_SLIPPAGE_PCT, FEE_RATE

INITIAL_CAPITAL = ap.INITIAL_CAPITAL


def breadth_target_weights(data, symbols, ts, *, target_vol=0.15, top_n=2,
                           min_selected=2):
    weights = ca.target_weights(
        data, symbols, ts, target_vol=target_vol, top_n=top_n
    )
    selected = sum(float(v) > 0 for v in weights.values())
    if selected < min_selected:
        return {s: 0.0 for s in symbols}
    return weights


def simulate_breadth_allocator(data, symbols, start, end, *, target_vol=0.15,
                               top_n=2, min_selected=2, rebalance_days=7):
    start_ts, end_ts = ap.as_utc(start), ap.as_utc(end)
    calendar = sorted(set.intersection(*[
        set(data[s]["df"].loc[start_ts:end_ts].index) for s in symbols
    ]))
    if not calendar:
        empty = pd.Series(dtype=float)
        return {**ap.performance_metrics(empty), "equity_curve": empty,
                "rebalance_log": pd.DataFrame()}

    cash = INITIAL_CAPITAL
    qty = {s: 0.0 for s in symbols}
    pending = None
    pending_signal_time = None
    last_signal_i = None
    points, logs = [], []

    for i, ts in enumerate(calendar):
        if pending is not None:
            pre_eq = ca._mark(cash, qty, data, ts, "open")
            opens = {s: float(data[s]["df"].loc[ts, "open"]) for s in symbols}
            for s in symbols:
                target_value = pre_eq * float(pending.get(s, 0.0))
                current_value = qty[s] * opens[s]
                if current_value > target_value and qty[s] > 0:
                    sell_qty = min(qty[s], (current_value - target_value) / opens[s])
                    px = opens[s] * (1.0 - EXIT_SLIPPAGE_PCT)
                    gross = sell_qty * px
                    cash += gross * (1.0 - FEE_RATE)
                    qty[s] -= sell_qty

            eq_after_sells = ca._mark(cash, qty, data, ts, "open")
            for s in symbols:
                target_value = eq_after_sells * float(pending.get(s, 0.0))
                current_value = qty[s] * opens[s]
                deficit = max(0.0, target_value - current_value)
                if deficit <= 0 or cash <= 0:
                    continue
                px = opens[s] * (1.0 + ENTRY_SLIPPAGE_PCT)
                spend = min(deficit, cash / (1.0 + FEE_RATE))
                buy_qty = spend / px
                cash -= buy_qty * px * (1.0 + FEE_RATE)
                qty[s] += buy_qty

            logs.append({
                "signal_time": pending_signal_time,
                "execution_time": ts,
                "target_gross": sum(pending.values()),
                "selected": sum(float(v) > 0 for v in pending.values()),
                **{f"target_{s}": pending.get(s, 0.0) for s in symbols},
            })
            pending = None

        points.append((ts, ca._mark(cash, qty, data, ts, "close")))
        if last_signal_i is None or i - last_signal_i >= rebalance_days:
            pending = breadth_target_weights(
                data, symbols, ts, target_vol=target_vol, top_n=top_n,
                min_selected=min_selected,
            )
            pending_signal_time = ts
            last_signal_i = i

    ts = calendar[-1]
    for s in symbols:
        if qty[s] <= 0:
            continue
        px = float(data[s]["df"].loc[ts, "close"]) * (1.0 - EXIT_SLIPPAGE_PCT)
        cash += qty[s] * px * (1.0 - FEE_RATE)
        qty[s] = 0.0
    points[-1] = (ts, cash)

    equity = pd.Series([v for _, v in points], index=[t for t, _ in points], dtype=float)
    log = pd.DataFrame(logs)
    if not log.empty:
        if not bool((pd.to_datetime(log["execution_time"], utc=True) >
                     pd.to_datetime(log["signal_time"], utc=True)).all()):
            raise AssertionError("look-ahead detectado no breadth allocator")
    return {**ap.performance_metrics(equity), "equity_curve": equity,
            "rebalance_log": log}


def combine_rebalanced_sleeves(tactical, alpha, alpha_weight, *,
                               rebalance_days=90, transfer_cost=0.003):
    """Combine two independently simulated sleeves with periodic capital reset.

    Transfer cost is charged on one-way capital moved between sleeves. Underlying
    sleeve returns already include their own exchange fees/slippage.
    """
    if not 0 <= alpha_weight <= 1:
        raise ValueError("alpha_weight fora de [0,1]")
    if rebalance_days < 1 or transfer_cost < 0:
        raise ValueError("parametros de rebalance invalidos")
    idx = tactical.dropna().index.intersection(alpha.dropna().index).sort_values()
    if idx.empty:
        return pd.Series(dtype=float)
    tr = tactical.loc[idx].pct_change().fillna(0.0)
    ar = alpha.loc[idx].pct_change().fillna(0.0)
    tcap = INITIAL_CAPITAL * (1.0 - alpha_weight)
    acap = INITIAL_CAPITAL * alpha_weight
    last_rebalance = idx[0]
    points = []
    for i, ts in enumerate(idx):
        if i:
            tcap *= 1.0 + float(tr.loc[ts])
            acap *= 1.0 + float(ar.loc[ts])
        if i and (ts - last_rebalance).days >= rebalance_days:
            total = tcap + acap
            target_alpha = total * alpha_weight
            transfer = abs(target_alpha - acap)
            total -= transfer * transfer_cost
            acap = total * alpha_weight
            tcap = total * (1.0 - alpha_weight)
            last_rebalance = ts
        points.append(tcap + acap)
    return pd.Series(points, index=idx, dtype=float)
