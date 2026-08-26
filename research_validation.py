"""Krypton stages 3-6 research pipeline.

Stage 3: concentration + return/position correlation.
Stage 4: save every tested candidate + Deflated Sharpe + White Reality Check.
Stage 5: isolated regime experiment: BTC SMA200 vs individual SMA200 vs breadth 2/3.
Stage 6: only after regime selection, test the predefined drawdown overlay:
         DD >= 10% => 0.5% risk; DD >= 15% => no new entries.

The final reserve is excluded from every selection step and opened exactly once
for the final selected candidate. TP is FIXED at 3x ATR throughout this file.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import walk_forward as wf
from config import (
    CIRCUIT_BREAKER_PCT,
    ENTRY_SLIPPAGE_PCT,
    EXIT_SLIPPAGE_PCT,
    FEE_RATE,
    MAX_SIMULTANEOUS_POS,
    RISK_PER_TRADE,
    STOP_LOSS_ATR_MULT,
    TAKE_PROFIT_ATR_MULT,
    TRADING_PAIRS,
)
from research_metrics import concentration_metrics, correlation_report, deflated_sharpe_ratio, white_reality_check

INITIAL_CAPITAL = 10_000.0
FIXED_TP = 3.0
MIN_NOTIONAL = 10.0
REGIME_MODES = ("btc", "individual", "breadth")


@dataclass
class Position:
    symbol: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp
    equity_at_entry: float


def _as_utc(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _latest_bar(data, symbol, ts):
    df = data[symbol]["df"].loc[:ts]
    return None if df.empty else df.index[-1]


def _asset_above_sma(data, symbol, ts) -> bool:
    t = _latest_bar(data, symbol, ts)
    if t is None:
        return False
    sma = data[symbol]["sma200"].loc[t]
    close = data[symbol]["df"].loc[t, "close"]
    return pd.notna(sma) and float(close) > float(sma)


def regime_allows(data, symbols, symbol, ts, mode: str) -> bool:
    if mode == "btc":
        return _asset_above_sma(data, "BTCUSDT", ts)
    if mode == "individual":
        return _asset_above_sma(data, symbol, ts)
    if mode == "breadth":
        return sum(_asset_above_sma(data, s, ts) for s in symbols) >= 2
    raise ValueError(f"regime_mode inválido: {mode}")


def _mark(cash, positions, data, ts, field="close"):
    equity = cash
    for symbol, pos in positions.items():
        df = data[symbol]["df"]
        if ts in df.index:
            px = float(df.loc[ts, field])
        else:
            eligible = df.loc[:ts]
            px = pos.entry_price if eligible.empty else float(eligible["close"].iloc[-1])
        equity += pos.quantity * px
    return equity


def _exit(cash, pos, price):
    gross = pos.quantity * price
    exit_fee = gross * FEE_RATE
    pnl = pos.quantity * (price - pos.entry_price) - pos.quantity * pos.entry_price * FEE_RATE - exit_fee
    return cash + gross - exit_fee, pnl


def simulate(data, symbols, start, end, regime_mode="btc", drawdown_overlay=False,
             base_risk=RISK_PER_TRADE, weights=None):
    start_ts, end_ts = _as_utc(start), _as_utc(end)
    weights = weights or wf.normalized_weights(symbols)
    calendar = sorted(set().union(*[set(data[s]["df"].loc[start_ts:end_ts].index) for s in symbols]))
    if len(calendar) < 20:
        return {"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "trades": 0,
                "final_capital": INITIAL_CAPITAL, "trade_log": pd.DataFrame(),
                "equity_curve": pd.Series(dtype=float), "exposure": pd.DataFrame()}

    cash = INITIAL_CAPITAL
    positions = {}
    pending = {}
    trades = []
    equity_points = []
    exposure_rows = []
    peak = INITIAL_CAPITAL
    daily_start = INITIAL_CAPITAL
    daily_date = None
    hard_halt = False

    for ts in calendar:
        pre_eq = _mark(cash, positions, data, ts, "open")
        if daily_date != ts.date():
            daily_date, daily_start = ts.date(), pre_eq

        for symbol in list(positions):
            df, sig = data[symbol]["df"], data[symbol]["signals"]
            if ts not in df.index:
                continue
            loc = df.index.get_loc(ts)
            if isinstance(loc, slice) or loc == 0:
                continue
            prev = df.index[loc - 1]
            if prev >= start_ts and int(sig.loc[prev]) != 1:
                pos = positions.pop(symbol)
                px = float(df.loc[ts, "open"]) * (1.0 - EXIT_SLIPPAGE_PCT)
                cash, pnl = _exit(cash, pos, px)
                trades.append({"symbol": symbol, "entry_time": pos.entry_time, "exit_time": ts,
                               "pnl": pnl, "portfolio_return": pnl / pos.equity_at_entry, "reason": "Sig"})

        for symbol in list(pending):
            if hard_halt or len(positions) >= MAX_SIMULTANEOUS_POS or symbol in positions:
                pending.pop(symbol, None)
                continue
            df = data[symbol]["df"]
            if ts not in df.index:
                continue
            current_eq = _mark(cash, positions, data, ts, "open")
            daily_loss = (daily_start - current_eq) / daily_start if daily_start > 0 else 0.0
            if daily_loss >= CIRCUIT_BREAKER_PCT:
                pending.clear()
                break
            dd = (peak - current_eq) / peak if peak > 0 else 0.0
            if drawdown_overlay and dd >= 0.15:
                pending.pop(symbol, None)
                continue
            risk = 0.005 if drawdown_overlay and dd >= 0.10 else base_risk
            atr_value = pending.pop(symbol)
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue
            entry = float(df.loc[ts, "open"]) * (1.0 + ENTRY_SLIPPAGE_PCT)
            sl_dist = atr_value * STOP_LOSS_ATR_MULT
            tp_dist = atr_value * FIXED_TP
            raw_qty = current_eq * risk / sl_dist
            cap = min(current_eq * weights[symbol], cash / (1.0 + FEE_RATE))
            qty = min(raw_qty, cap / entry if entry > 0 else 0.0)
            notional = qty * entry
            if qty <= 0 or notional < MIN_NOTIONAL:
                continue
            debit = notional * (1.0 + FEE_RATE)
            if debit > cash:
                continue
            cash -= debit
            positions[symbol] = Position(symbol, entry, qty, entry - sl_dist, entry + tp_dist, ts, current_eq)

        for symbol in list(positions):
            df = data[symbol]["df"]
            if ts not in df.index:
                continue
            row, pos = df.loc[ts], positions[symbol]
            op = float(row["open"])
            if op <= pos.stop_loss:
                reason, px = "SL_GAP", op * (1.0 - EXIT_SLIPPAGE_PCT)
            elif op >= pos.take_profit:
                reason, px = "TP_GAP", pos.take_profit
            elif float(row["low"]) <= pos.stop_loss:
                reason, px = "SL", pos.stop_loss * (1.0 - EXIT_SLIPPAGE_PCT)
            elif float(row["high"]) >= pos.take_profit:
                reason, px = "TP", pos.take_profit
            else:
                continue
            positions.pop(symbol)
            cash, pnl = _exit(cash, pos, px)
            trades.append({"symbol": symbol, "entry_time": pos.entry_time, "exit_time": ts,
                           "pnl": pnl, "portfolio_return": pnl / pos.equity_at_entry, "reason": reason})

        current_eq = _mark(cash, positions, data, ts)
        peak = max(peak, current_eq)
        daily_loss = (daily_start - current_eq) / daily_start if daily_start > 0 else 0.0
        if daily_loss >= CIRCUIT_BREAKER_PCT:
            pending.clear()

        # Stage 6 overlay is intentionally NOT a permanent halt at -15%: it blocks
        # new entries while DD is >=15%, allowing recovery of existing positions.
        dd = (peak - current_eq) / peak if peak > 0 else 0.0
        block_new = drawdown_overlay and dd >= 0.15
        if not hard_halt and not block_new and daily_loss < CIRCUIT_BREAKER_PCT:
            for symbol in symbols:
                if symbol in positions or symbol in pending:
                    continue
                if len(positions) + len(pending) >= MAX_SIMULTANEOUS_POS:
                    break
                if not regime_allows(data, symbols, symbol, ts, regime_mode):
                    continue
                df, sig, atr = data[symbol]["df"], data[symbol]["signals"], data[symbol]["atr"]
                if ts not in df.index:
                    continue
                loc = df.index.get_loc(ts)
                if isinstance(loc, slice) or loc + 1 >= len(df) or df.index[loc + 1] > end_ts:
                    continue
                av = float(atr.loc[ts]) if pd.notna(atr.loc[ts]) else np.nan
                if int(sig.loc[ts]) == 1 and np.isfinite(av) and av > 0:
                    pending[symbol] = av

        equity_points.append((ts, current_eq))
        exposure_rows.append({"time": ts, **{s: int(s in positions) for s in symbols}})

    for symbol in list(positions):
        df = data[symbol]["df"].loc[:end_ts]
        if df.empty:
            continue
        ts = df.index[-1]
        pos = positions.pop(symbol)
        px = float(df["close"].iloc[-1]) * (1.0 - EXIT_SLIPPAGE_PCT)
        cash, pnl = _exit(cash, pos, px)
        trades.append({"symbol": symbol, "entry_time": pos.entry_time, "exit_time": ts,
                       "pnl": pnl, "portfolio_return": pnl / pos.equity_at_entry, "reason": "EOD"})

    if equity_points:
        equity_points[-1] = (equity_points[-1][0], cash)
    eq = pd.Series([v for _, v in equity_points], index=[t for t, _ in equity_points], dtype=float)
    rets = eq.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(365)) if len(rets) > 1 and rets.std() > 0 else 0.0
    max_dd = float(((eq - eq.cummax()) / eq.cummax()).min()) if not eq.empty else 0.0
    exposure = pd.DataFrame(exposure_rows).set_index("time") if exposure_rows else pd.DataFrame()
    return {"return": cash / INITIAL_CAPITAL - 1.0, "sharpe": sharpe, "max_drawdown": max_dd,
            "trades": len(trades), "final_capital": cash, "trade_log": pd.DataFrame(trades),
            "equity_curve": eq, "exposure": exposure}


def _folds(start, selection_end, train_days, test_days, step_days):
    fold_start, fold = start, 1
    while fold_start + timedelta(days=train_days + test_days - 1) <= selection_end:
        train_end = fold_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)
        yield fold, test_start, test_end
        fold += 1
        fold_start += timedelta(days=step_days)


def evaluate_candidate(data, symbols, folds, label, regime_mode, drawdown_overlay,
                       max_top5_share=0.50):
    rows, return_parts, exposures, trades = [], [], [], []
    for fold, test_start, test_end in folds:
        m = simulate(data, symbols, test_start, test_end, regime_mode, drawdown_overlay)
        c = concentration_metrics(m["trade_log"], m["equity_curve"], INITIAL_CAPITAL)
        passed = (c["return_without_best_trade"] > 0 and c["return_without_best_5d"] > 0 and
                  c["return_without_best_20d"] > 0 and c["top_5pct_trade_pnl_share"] <= max_top5_share)
        rows.append({"candidate": label, "fold": fold, "test_start": test_start.date(), "test_end": test_end.date(),
                     "regime_mode": regime_mode, "drawdown_overlay": drawdown_overlay,
                     "tp_atr": FIXED_TP, "risk_base": RISK_PER_TRADE, "return": m["return"],
                     "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"], "trades": m["trades"],
                     **c, "concentration_pass": passed})
        if not m["equity_curve"].empty:
            return_parts.append(m["equity_curve"].pct_change().dropna())
        if not m["exposure"].empty:
            exposures.append(m["exposure"])
        if not m["trade_log"].empty:
            trades.append(m["trade_log"])
    returns = pd.concat(return_parts).sort_index() if return_parts else pd.Series(dtype=float)
    exposure = pd.concat(exposures).sort_index() if exposures else pd.DataFrame()
    trade_log = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return pd.DataFrame(rows), returns, exposure, trade_log


def main():
    if abs(TAKE_PROFIT_ATR_MULT - FIXED_TP) > 1e-12:
        raise RuntimeError("research_validation exige TAKE_PROFIT_ATR_MULT=3.0; TP não será pesquisado.")
    p = argparse.ArgumentParser(description="Krypton stages 3-6 robust validation")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    p.add_argument("--symbols", nargs="+", default=list(TRADING_PAIRS))
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--test-days", type=int, default=180)
    p.add_argument("--step-days", type=int, default=180)
    p.add_argument("--reserve-days", type=int, default=180)
    p.add_argument("--max-top5-share", type=float, default=0.50)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    reserve_start = end - timedelta(days=args.reserve_days - 1)
    selection_end = reserve_start - timedelta(days=1)
    symbols = [s.upper() for s in args.symbols]
    if set(symbols) != set(TRADING_PAIRS):
        raise ValueError("Experimento breadth exige exatamente os 3 ativos configurados em TRADING_PAIRS.")

    data = wf._prepare_data(symbols, start, end)
    folds = list(_folds(start, selection_end, args.train_days, args.test_days, args.step_days))
    if not folds:
        raise RuntimeError("Nenhum fold OOS completo antes da reserva final.")

    all_rows, candidate_returns, candidate_exposure = [], {}, {}
    # Stage 5: compare only the three predeclared regime hypotheses.
    for mode in REGIME_MODES:
        label = f"regime_{mode}"
        rows, rets, exposure, _ = evaluate_candidate(data, symbols, folds, label, mode, False, args.max_top5_share)
        all_rows.append(rows)
        candidate_returns[label] = rets
        candidate_exposure[label] = exposure

    first = pd.concat(all_rows, ignore_index=True)
    summary = first.groupby("candidate").agg(mean_return=("return", "mean"), median_return=("return", "median"),
                                               mean_sharpe=("sharpe", "mean"), worst_dd=("max_drawdown", "min"),
                                               trades=("trades", "sum"), concentration_pass=("concentration_pass", "all")).reset_index()
    eligible = summary[summary["concentration_pass"]]
    if eligible.empty:
        regime_winner = str(summary.sort_values(["median_return", "mean_sharpe"], ascending=False).iloc[0]["candidate"])
        selection_warning = "Nenhum regime passou concentração; vencedor provisório NÃO promovível."
    else:
        regime_winner = str(eligible.sort_values(["median_return", "mean_sharpe"], ascending=False).iloc[0]["candidate"])
        selection_warning = None
    selected_mode = regime_winner.replace("regime_", "")

    # Stage 6: one predefined overlay only, no grid.
    dd_label = f"{regime_winner}_dd_10_15"
    dd_rows, dd_rets, dd_exposure, _ = evaluate_candidate(data, symbols, folds, dd_label, selected_mode, True, args.max_top5_share)
    all_rows.append(dd_rows)
    candidate_returns[dd_label] = dd_rets
    candidate_exposure[dd_label] = dd_exposure

    results = pd.concat(all_rows, ignore_index=True)
    n_trials = len(candidate_returns)
    dsr = {name: deflated_sharpe_ratio(rets, n_trials) for name, rets in candidate_returns.items()}
    wrc = white_reality_check(candidate_returns, bootstrap_samples=args.bootstrap)
    results["deflated_sharpe_ratio"] = results["candidate"].map(dsr)
    results.to_csv("config_search_results.csv", index=False)

    final_summary = results.groupby("candidate").agg(mean_return=("return", "mean"), median_return=("return", "median"),
                                                       mean_sharpe=("sharpe", "mean"), worst_dd=("max_drawdown", "min"),
                                                       trades=("trades", "sum"), concentration_pass=("concentration_pass", "all"),
                                                       dsr=("deflated_sharpe_ratio", "first")).reset_index()
    final_eligible = final_summary[final_summary["concentration_pass"]]
    if final_eligible.empty:
        final_candidate = regime_winner
    else:
        final_candidate = str(final_eligible.sort_values(["median_return", "mean_sharpe", "dsr"], ascending=False).iloc[0]["candidate"])

    final_mode = selected_mode if final_candidate == dd_label else final_candidate.replace("regime_", "")
    final_dd = final_candidate == dd_label

    # Stage 3 correlation report for the selected configuration on selection OOS only.
    corr = correlation_report(data, symbols, folds[0][1], folds[-1][2], candidate_exposure.get(final_candidate))
    corr["daily_return_correlation"].to_csv("correlation_daily_returns.csv")
    corr["position_correlation"].to_csv("correlation_positions.csv")

    # Locked reserve: touched only after every selection above is complete.
    reserve = simulate(data, symbols, reserve_start, end, final_mode, final_dd)
    reserve_conc = concentration_metrics(reserve["trade_log"], reserve["equity_curve"], INITIAL_CAPITAL)
    report = {
        "tp_frozen_atr": FIXED_TP,
        "selection_period_end": selection_end.date().isoformat(),
        "reserve_start": reserve_start.date().isoformat(),
        "reserve_end": end.date().isoformat(),
        "regime_winner_before_dd": regime_winner,
        "final_candidate": final_candidate,
        "selection_warning": selection_warning,
        "multiple_testing": {"n_trials": n_trials, "dsr": dsr, "white_reality_check": wrc},
        "reserve": {"return": reserve["return"], "sharpe": reserve["sharpe"],
                    "max_drawdown": reserve["max_drawdown"], "trades": reserve["trades"], **reserve_conc},
    }
    with open("research_validation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    final_summary.to_csv("config_search_summary.csv", index=False)
    print(final_summary.to_string(index=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
