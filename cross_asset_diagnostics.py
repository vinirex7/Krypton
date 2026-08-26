"""Cross-asset opportunity diagnostics for Krypton research only.

The old ex-post label bull/bear/sideways uses BTC endpoint return only. That is
not sufficient for a BTC/SOL/BNB portfolio: BTC can finish a 180-day window
flat while altcoins experience large tradable trends.

This module measures continuous, ex-post diagnostics without changing signals:
- endpoint return per asset;
- maximum 30/60/90-day rallies and drawdowns;
- own-SMA200 occupancy and crossings;
- trend efficiency and realized volatility per asset;
- mean cross-asset SMA200 breadth;
- gap between BTC risk-on occupancy and portfolio breadth;
- long-trend opportunity count across assets.

No parameter search, no holdout access, no live changes.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import forensic_diagnostics as fd
import regime_diagnostics as rd
import walk_forward as wf
from config import TRADING_PAIRS

WINDOW_DAYS = 180
RALLY_THRESHOLD = 0.20
LOOKBACKS = (30, 60, 90)


def _as_utc(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _trend_efficiency(close: pd.Series) -> float:
    close = close.dropna().astype(float)
    if len(close) < 2:
        return np.nan
    path = float(close.diff().abs().sum())
    return abs(float(close.iloc[-1] - close.iloc[0])) / path if path > 0 else 0.0


def _asset_metrics(data, symbol, start, end):
    item = data[symbol]
    df = item["df"].loc[_as_utc(start):_as_utc(end)].copy()
    if len(df) < 2:
        return {}
    close = df["close"].astype(float)
    sma = item["sma200"].reindex(df.index)
    above = (close > sma).where(sma.notna())
    valid_above = above.dropna()
    crossings = int((valid_above.astype(int).diff().abs() == 1).sum()) if len(valid_above) else 0
    rets = close.pct_change().dropna()
    out = {
        "return": float(close.iloc[-1] / close.iloc[0] - 1.0),
        "sma200_above_fraction": float(valid_above.mean()) if len(valid_above) else np.nan,
        "sma200_crossings": crossings,
        "trend_efficiency": _trend_efficiency(close),
        "realized_vol": float(rets.std() * np.sqrt(365.0)) if len(rets) > 1 else 0.0,
        "max_drawdown": float((close / close.cummax() - 1.0).min()),
    }
    for days in LOOKBACKS:
        roll = close.pct_change(days)
        out[f"best_{days}d_return"] = float(roll.max()) if roll.notna().any() else np.nan
        out[f"worst_{days}d_return"] = float(roll.min()) if roll.notna().any() else np.nan
    return out


def _cross_asset_daily(data, symbols, start, end):
    frames = []
    for symbol in symbols:
        item = data[symbol]
        df = item["df"].loc[_as_utc(start):_as_utc(end), ["close"]].copy()
        df[f"above_{symbol}"] = (df["close"] > item["sma200"].reindex(df.index)).astype(float)
        df[f"mom90_{symbol}"] = (df["close"] > df["close"].shift(90)).astype(float)
        frames.append(df[[f"above_{symbol}", f"mom90_{symbol}"]])
    joined = pd.concat(frames, axis=1).dropna(how="all")
    if joined.empty:
        return pd.DataFrame()
    above_cols = [f"above_{s}" for s in symbols]
    mom_cols = [f"mom90_{s}" for s in symbols]
    joined["sma_breadth"] = joined[above_cols].mean(axis=1)
    joined["momentum90_breadth"] = joined[mom_cols].mean(axis=1)
    return joined


def classify_opportunity(asset_metrics: dict[str, dict]) -> str:
    """Diagnostic-only label based on realized 60-day upside opportunity."""
    count = sum(
        np.isfinite(m.get("best_60d_return", np.nan)) and m.get("best_60d_return", 0.0) >= RALLY_THRESHOLD
        for m in asset_metrics.values()
    )
    if count >= 2:
        return "broad_long_opportunity"
    if count == 1:
        return "narrow_long_opportunity"
    return "low_long_opportunity"


def run(data, symbols, start, end, window_days=WINDOW_DAYS):
    rows = []
    for period, ps, pe in rd._periods(start, end, window_days):
        strategy = fd.simulate_forensic(data, symbols, ps, pe, regime_mode="btc")
        asset = {s: _asset_metrics(data, s, ps, pe) for s in symbols}
        daily = _cross_asset_daily(data, symbols, ps, pe)
        btc_on = float(strategy["summary"].get("btc_pct_days_above_sma200", np.nan))
        breadth = float(daily["sma_breadth"].mean()) if not daily.empty else np.nan
        mom90 = float(daily["momentum90_breadth"].mean()) if not daily.empty else np.nan
        endpoint = [m.get("return", np.nan) for m in asset.values()]
        best60 = [m.get("best_60d_return", np.nan) for m in asset.values()]
        valid_endpoint = [x for x in endpoint if np.isfinite(x)]
        valid_best60 = [x for x in best60 if np.isfinite(x)]
        row = {
            "period": period,
            "start": ps.date().isoformat(),
            "end": pe.date().isoformat(),
            "old_btc_regime": rd.classify_period(rd._btc_period_return(data, ps, pe)),
            "opportunity_label": classify_opportunity(asset),
            "strategy_return": strategy["summary"]["strategy_return"],
            "trades": strategy["summary"]["trades"],
            "win_rate": (
                strategy["summary"]["winning_trades"] / strategy["summary"]["trades"]
                if strategy["summary"]["trades"] else np.nan
            ),
            "mean_gross_exposure": strategy["summary"]["mean_gross_exposure"],
            "btc_risk_on_fraction": btc_on,
            "mean_asset_sma_breadth": breadth,
            "btc_vs_portfolio_breadth_gap": btc_on - breadth if np.isfinite(btc_on) and np.isfinite(breadth) else np.nan,
            "mean_momentum90_breadth": mom90,
            "positive_endpoint_assets": int(sum(x > 0 for x in valid_endpoint)),
            "endpoint_assets_gt20": int(sum(x >= RALLY_THRESHOLD for x in valid_endpoint)),
            "endpoint_assets_lt_minus20": int(sum(x <= -RALLY_THRESHOLD for x in valid_endpoint)),
            "best_endpoint_return": max(valid_endpoint) if valid_endpoint else np.nan,
            "median_endpoint_return": float(np.median(valid_endpoint)) if valid_endpoint else np.nan,
            "endpoint_dispersion": float(np.std(valid_endpoint, ddof=1)) if len(valid_endpoint) > 1 else np.nan,
            "assets_with_60d_rally_gt20": int(sum(x >= RALLY_THRESHOLD for x in valid_best60)),
            "mean_best_60d_return": float(np.mean(valid_best60)) if valid_best60 else np.nan,
            "min_best_60d_return": min(valid_best60) if valid_best60 else np.nan,
            "max_best_60d_return": max(valid_best60) if valid_best60 else np.nan,
            "max_losing_streak": strategy["summary"]["max_losing_streak"],
            "median_post_exit_return_20d": strategy["summary"]["median_post_exit_return_20d"],
        }
        for symbol, metrics in asset.items():
            for key, value in metrics.items():
                row[f"{symbol}.{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def compare_focus(results: pd.DataFrame, focus_period=10):
    if results.empty or focus_period not in set(results["period"]):
        return {}
    focus = results.loc[results["period"] == focus_period].iloc[0]
    profitable = results[(results["strategy_return"] > 0) & (results["trades"] > 0)]
    fields = [
        "strategy_return", "win_rate", "mean_gross_exposure", "btc_risk_on_fraction",
        "mean_asset_sma_breadth", "btc_vs_portfolio_breadth_gap", "mean_momentum90_breadth",
        "positive_endpoint_assets", "endpoint_assets_gt20", "assets_with_60d_rally_gt20",
        "mean_best_60d_return", "max_losing_streak", "median_post_exit_return_20d",
    ]
    out = {"focus_period": int(focus_period), "focus": {}, "profitable_active_median": {}}
    for field in fields:
        val = focus.get(field, np.nan)
        out["focus"][field] = None if pd.isna(val) else float(val)
        if field in profitable:
            med = profitable[field].median()
            out["profitable_active_median"][field] = None if pd.isna(med) else float(med)
    return out


def main():
    p = argparse.ArgumentParser(description="Krypton cross-asset opportunity diagnostics")
    p.add_argument("--start", default="2020-08-01")
    p.add_argument("--end", default="2026-06-30")
    p.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    symbols = list(TRADING_PAIRS)
    data = wf._prepare_data(symbols, start, end)
    results = run(data, symbols, start, end, args.window_days)
    comparison = compare_focus(results)

    results.to_csv("cross_asset_periods.csv", index=False)
    with open("cross_asset_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, ensure_ascii=False)

    display_cols = [
        "period", "old_btc_regime", "opportunity_label", "strategy_return", "win_rate",
        "mean_gross_exposure", "btc_risk_on_fraction", "mean_asset_sma_breadth",
        "btc_vs_portfolio_breadth_gap", "assets_with_60d_rally_gt20", "mean_best_60d_return",
        "positive_endpoint_assets", "endpoint_assets_gt20", "max_losing_streak",
    ]
    print("\nCROSS-ASSET PERIODS")
    print(results[display_cols].to_string(index=False))
    print("\nFOCUS COMPARISON")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    print("\nArquivos: cross_asset_periods.csv | cross_asset_comparison.json")
    print("Nota: opportunity_label é ex-post e diagnóstico; nunca entra no sinal live.")


if __name__ == "__main__":
    main()
