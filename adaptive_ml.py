"""Research-only ML meta-filter and DQN reward-timing audit.

XGBoost is optional and belongs in ``requirements-research.txt``; the live bot
does not import this module.  Targets are future net returns, training windows
are rolling, and a horizon embargo prevents label overlap with each test fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from adaptive_portfolio import INITIAL_CAPITAL, ROUND_TRIP_COST, as_utc, performance_metrics
from indicators import compute_atr, compute_macd, compute_rsi

FEATURE_COLUMNS = [
    "ret1", "mom5", "mom20", "mom60", "dist_sma20", "dist_sma50",
    "dist_sma200", "rsi14", "macd_hist_pct", "atr_pct", "vol20",
    "btc_mom20", "btc_dist_sma200", "btc_atr_pct",
]


def _asset_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    _, _, hist = compute_macd(close)
    out = pd.DataFrame(index=df.index)
    out["ret1"] = close.pct_change()
    out["mom5"] = close.pct_change(5)
    out["mom20"] = close.pct_change(20)
    out["mom60"] = close.pct_change(60)
    out["dist_sma20"] = close / close.rolling(20).mean() - 1.0
    out["dist_sma50"] = close / close.rolling(50).mean() - 1.0
    out["dist_sma200"] = close / close.rolling(200).mean() - 1.0
    out["rsi14"] = compute_rsi(close, 14) / 100.0
    out["macd_hist_pct"] = hist / close
    out["atr_pct"] = compute_atr(df["high"], df["low"], close) / close
    out["vol20"] = out["ret1"].rolling(20).std()
    return out


def build_supervised_table(data, symbols, horizon: int = 5,
                           target_cost_multiple: float = 2.0) -> pd.DataFrame:
    """Create lagged features and a future-return label for all configured assets."""
    btc_features = _asset_features(data["BTCUSDT"]["df"])
    btc_context = btc_features[["mom20", "dist_sma200", "atr_pct"]].rename(columns={
        "mom20": "btc_mom20", "dist_sma200": "btc_dist_sma200", "atr_pct": "btc_atr_pct"
    })
    frames = []
    for symbol in symbols:
        df = data[symbol]["df"].sort_index()
        features = _asset_features(df).join(btc_context, how="left").ffill()
        future_return = df["close"].shift(-horizon) / df["close"] - 1.0
        features["target"] = (future_return > target_cost_multiple * ROUND_TRIP_COST).astype(float)
        features.loc[future_return.isna(), "target"] = np.nan
        features["future_return"] = future_return
        features["symbol"] = symbol
        features["time"] = features.index
        frames.append(features.reset_index(drop=True))
    table = pd.concat(frames, ignore_index=True)
    return table.dropna(subset=FEATURE_COLUMNS).sort_values(["time", "symbol"]).reset_index(drop=True)


@dataclass
class PredictionResult:
    probabilities: dict[str, pd.Series]
    folds: pd.DataFrame


def walk_forward_xgboost_probabilities(
    data,
    symbols,
    start,
    end,
    *,
    horizon: int = 5,
    train_days: int = 1095,
    test_days: int = 90,
    min_train_rows: int = 800,
) -> PredictionResult:
    """Generate rolling, embargoed probabilities with fixed XGBoost settings."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - exercised in research CI
        raise RuntimeError("xgboost ausente; instale requirements-research.txt") from exc

    start_ts, end_ts = as_utc(start), as_utc(end)
    table = build_supervised_table(data, symbols, horizon=horizon)
    outputs = {s: pd.Series(dtype=float) for s in symbols}
    rows = []
    test_start = start_ts
    while test_start <= end_ts:
        test_end = min(test_start + pd.Timedelta(days=test_days - 1), end_ts)
        train_start = test_start - pd.Timedelta(days=train_days)
        # Labels need horizon fully realized before the fold begins.
        embargo_end = test_start - pd.Timedelta(days=horizon)
        train = table[(table["time"] >= train_start) & (table["time"] < embargo_end)].dropna(subset=["target"])
        test = table[(table["time"] >= test_start) & (table["time"] <= test_end)]
        if len(train) >= min_train_rows and not test.empty and train["target"].nunique() == 2:
            model = XGBClassifier(
                n_estimators=140,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=2,
            )
            model.fit(train[FEATURE_COLUMNS], train["target"].astype(int))
            probs = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            pred = test[["time", "symbol"]].copy()
            pred["probability"] = probs
            for symbol in symbols:
                part = pred[pred["symbol"] == symbol].set_index("time")["probability"]
                outputs[symbol] = pd.concat([outputs[symbol], part])
            rows.append({"test_start": test_start, "test_end": test_end,
                         "train_rows": len(train), "test_rows": len(test),
                         "positive_rate": float(train["target"].mean())})
        else:
            rows.append({"test_start": test_start, "test_end": test_end,
                         "train_rows": len(train), "test_rows": len(test),
                         "positive_rate": float(train["target"].mean()) if len(train) else np.nan})
        test_start += pd.Timedelta(days=test_days)
    return PredictionResult(
        probabilities={s: v[~v.index.duplicated(keep="last")].sort_index() for s, v in outputs.items()},
        folds=pd.DataFrame(rows),
    )


def probability_permission(probabilities: dict[str, pd.Series], threshold: float = 0.55):
    """Build a fail-closed entry permission callable for the tactical simulator."""
    def allowed(symbol: str, ts: pd.Timestamp) -> bool:
        series = probabilities.get(symbol)
        if series is None or ts not in series.index:
            return False
        value = float(series.loc[ts])
        return np.isfinite(value) and value >= threshold
    return allowed


def _paper_strategy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Long-only versions of the five technical actions in the cited DQN paper."""
    close = df["close"].astype(float)
    rsi = compute_rsi(close, 14)
    sma5, sma10 = close.rolling(5).mean(), close.rolling(10).mean()
    mid, std = close.rolling(20).mean(), close.rolling(20).std()
    lower, upper = mid - 2 * std, mid + 2 * std
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap = (typical * df["volume"]).rolling(14).sum() / df["volume"].rolling(14).sum()
    signals = pd.DataFrame(index=df.index)
    signals["rsi"] = (rsi < 30).astype(float)
    signals["sma"] = (sma5 > sma10).astype(float)
    signals["bollinger"] = (close < lower).astype(float)
    signals["momentum20"] = (close.pct_change(20) > 0).astype(float)
    signals["vwap"] = (close < vwap).astype(float)
    return signals.fillna(0.0)


def audit_dqn_reward_timing(daily: pd.DataFrame, lookback: int = 90) -> dict:
    """Quantify the paper environment's same-bar timing versus executable timing.

    This is intentionally an environment audit, not a claim to reproduce the
    paper's unavailable trained network weights.  The same rolling selector is
    evaluated once with ``signal_t * return_t`` and once with the executable
    ``signal_t * return_(t+1)`` convention, both after turnover costs.
    """
    df = daily.sort_index()
    signals = _paper_strategy_signals(df)
    returns = df["close"].pct_change().fillna(0.0)
    same_reward = signals.mul(returns, axis=0)
    next_reward = signals.mul(returns.shift(-1), axis=0)
    same_nav = INITIAL_CAPITAL
    next_nav = INITIAL_CAPITAL
    same_pos = next_pos = 0.0
    same_points, next_points = [], []
    choices = []
    for i in range(lookback, len(df) - 1):
        # Selection uses only rows strictly before i in both cases.
        scores = same_reward.iloc[i-lookback:i].mean()
        action = str(scores.idxmax())
        pos = float(signals.iloc[i][action])
        same_r = float(pos * returns.iloc[i] - ROUND_TRIP_COST / 2.0 * abs(pos - same_pos))
        next_r = float(pos * returns.iloc[i+1] - ROUND_TRIP_COST / 2.0 * abs(pos - next_pos))
        same_nav *= max(1.0 + same_r, 1e-9)
        next_nav *= max(1.0 + next_r, 1e-9)
        same_pos = next_pos = pos
        same_points.append((df.index[i], same_nav))
        next_points.append((df.index[i+1], next_nav))
        choices.append(action)
    same_curve = pd.Series([v for _, v in same_points], index=[t for t, _ in same_points], dtype=float)
    next_curve = pd.Series([v for _, v in next_points], index=[t for t, _ in next_points], dtype=float)
    return {
        "same_bar": performance_metrics(same_curve),
        "lagged_next_bar": performance_metrics(next_curve),
        "same_bar_equity": same_curve,
        "lagged_equity": next_curve,
        "action_counts": pd.Series(choices).value_counts().to_dict(),
    }

