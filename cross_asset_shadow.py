"""Forward shadow logger for the frozen cross-asset hybrid candidate.

Read-only by design: public market data only, no Binance trading client and no
order submission. The logger freezes and records the information/decisions that
were actually available at each completed daily close for a future OOS paper
evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import adaptive_portfolio as ap
import backtest
import cross_asset_hybrid_v2 as hv2
import walk_forward as wf

PAPER_START = pd.Timestamp("2026-08-27", tz="UTC")
LIVE_SYMBOLS = list(wf.BASE_WEIGHTS)
ALPHA_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

ALPHA_WEIGHT = 0.10
TACTICAL_WEIGHT = 0.90
TARGET_VOL = 0.15
TOP_N = 2
MIN_SELECTED = 2
ALLOCATOR_REBALANCE_BARS = 45
SLEEVE_REBALANCE_DAYS = 90
TRANSFER_COST = 0.003
MOMENTUM_WINDOWS = (30, 90, 180)
VOL_WINDOW = 20
COV_WINDOW = 60

backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"


def frozen_candidate() -> dict:
    return {
        "paper_start": PAPER_START.strftime("%Y-%m-%d"),
        "live_symbols": LIVE_SYMBOLS,
        "alpha_symbols": ALPHA_SYMBOLS,
        "alpha_weight": ALPHA_WEIGHT,
        "tactical_weight": TACTICAL_WEIGHT,
        "target_vol": TARGET_VOL,
        "top_n": TOP_N,
        "min_selected": MIN_SELECTED,
        "allocator_rebalance_bars": ALLOCATOR_REBALANCE_BARS,
        "sleeve_rebalance_days": SLEEVE_REBALANCE_DAYS,
        "transfer_cost": TRANSFER_COST,
        "momentum_windows": list(MOMENTUM_WINDOWS),
        "vol_window": VOL_WINDOW,
        "cov_window": COV_WINDOW,
        "execution": "next_daily_open",
        "spot_only": True,
        "leverage": 1.0,
    }


def strategy_fingerprint() -> str:
    payload = json.dumps(frozen_candidate(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def as_utc(value=None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def last_completed_daily_close(as_of=None) -> pd.Timestamp:
    ts = as_utc(as_of)
    return ts.normalize() - pd.Timedelta(days=1)


def common_paper_calendar(data, signal_time, paper_start=PAPER_START) -> pd.DatetimeIndex:
    """Return common completed bars exactly as the research allocator does."""
    ts = as_utc(signal_time).normalize()
    start = as_utc(paper_start).normalize()
    common = set.intersection(*[
        set(data[s]["df"].loc[start:ts].index) for s in ALPHA_SYMBOLS
    ])
    return pd.DatetimeIndex(sorted(common))


def rebalance_due_from_calendar(calendar: pd.DatetimeIndex, signal_time) -> tuple[bool, int]:
    """Match research cadence: first common paper bar, then every 45 bars."""
    ts = as_utc(signal_time).normalize()
    if ts not in calendar:
        return False, -1
    loc = calendar.get_loc(ts)
    if isinstance(loc, slice):
        raise ValueError("duplicate shadow timestamps")
    bars_since_start = int(loc)
    return bars_since_start % ALLOCATOR_REBALANCE_BARS == 0, bars_since_start


def _risk_on(data, ts) -> bool:
    btc = data["BTCUSDT"]
    if ts not in btc["df"].index or ts not in btc["sma200"].index:
        return False
    sma = btc["sma200"].loc[ts]
    return bool(pd.notna(sma) and float(btc["df"].loc[ts, "close"]) > float(sma))


def build_snapshot(data, signal_time, *, paper_start=PAPER_START) -> dict:
    ts = as_utc(signal_time).normalize()
    calendar = common_paper_calendar(data, ts, paper_start)
    due, bars_since_start = rebalance_due_from_calendar(calendar, ts)

    if due:
        alpha_target = hv2.breadth_target_weights(
            data, ALPHA_SYMBOLS, ts,
            target_vol=TARGET_VOL,
            top_n=TOP_N,
            min_selected=MIN_SELECTED,
        )
    else:
        alpha_target = None

    gross = None if alpha_target is None else float(sum(alpha_target.values()))
    portfolio_alpha = None if alpha_target is None else {
        s: float(ALPHA_WEIGHT * alpha_target.get(s, 0.0)) for s in ALPHA_SYMBOLS
    }

    continuity_permission = ap.persistent_state_permission(data, LIVE_SYMBOLS)
    risk_on = _risk_on(data, ts)
    tactical = {}
    for symbol in LIVE_SYMBOLS:
        signal_series = data[symbol]["signals"]
        signal = int(signal_series.loc[ts]) if ts in signal_series.index else 0
        allowed = bool(continuity_permission(symbol, ts))
        tactical[symbol] = {
            "signal": signal,
            "continuity_allowed": allowed,
            "regime_risk_on": risk_on,
            "entry_candidate": bool(signal == 1 and allowed and risk_on),
        }

    return {
        "strategy_fingerprint": strategy_fingerprint(),
        "candidate": frozen_candidate(),
        "signal_time": ts.isoformat(),
        "execution_assumption": "next_daily_open",
        "paper_start": as_utc(paper_start).normalize().isoformat(),
        "bars_since_paper_start": bars_since_start,
        "alpha_rebalance_due": due,
        "alpha_target_within_sleeve": alpha_target,
        "alpha_target_gross": gross,
        "alpha_cash_within_sleeve": None if gross is None else float(max(0.0, 1.0 - gross)),
        "portfolio_alpha_target": portfolio_alpha,
        "tactical": tactical,
        "read_only": True,
        "orders_submitted": 0,
    }


def _append_jsonl(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def run(as_of=None, output="cross_asset_shadow_decisions.jsonl", *, paper_start=PAPER_START) -> dict:
    asof = as_utc(as_of)
    signal_time = last_completed_daily_close(asof)
    start = as_utc(paper_start).normalize()

    if signal_time < start:
        snapshot = {
            "strategy_fingerprint": strategy_fingerprint(),
            "candidate": frozen_candidate(),
            "as_of": asof.isoformat(),
            "last_completed_daily_close": signal_time.isoformat(),
            "paper_start": start.isoformat(),
            "status": "waiting_for_paper_start",
            "read_only": True,
            "orders_submitted": 0,
        }
        _append_jsonl(output, snapshot)
        return snapshot

    data = wf._prepare_data(ALPHA_SYMBOLS, start.to_pydatetime(), signal_time.to_pydatetime())
    snapshot = build_snapshot(data, signal_time, paper_start=start)
    snapshot["as_of"] = asof.isoformat()
    snapshot["status"] = "shadow_decision_recorded"
    _append_jsonl(output, snapshot)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description="Krypton read-only cross-asset shadow logger")
    parser.add_argument("--as-of", default=None, help="UTC timestamp for deterministic replay")
    parser.add_argument("--paper-start", default=PAPER_START.strftime("%Y-%m-%d"))
    parser.add_argument("--output", default="cross_asset_shadow_decisions.jsonl")
    args = parser.parse_args()
    snapshot = run(args.as_of, args.output, paper_start=as_utc(args.paper_start))
    print(json.dumps(snapshot, indent=2, default=str))


if __name__ == "__main__":
    main()
