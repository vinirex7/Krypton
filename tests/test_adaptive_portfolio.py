import unittest

import numpy as np
import pandas as pd

import adaptive_portfolio as ap
import range_grid as rg
import walk_forward as wf
from adaptive_ml import audit_dqn_reward_timing


def prepared_market(rows=240, signal_start=205):
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100, 130, rows), index=idx)
    df = pd.DataFrame({
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0,
    }, index=idx)
    signals = pd.Series(0, index=idx, dtype=int)
    signals.iloc[signal_start:] = 1
    atr = pd.Series(2.0, index=idx)
    sma = close.rolling(200, min_periods=200).mean()
    return {"df": df, "signals": signals, "atr": atr, "sma200": sma, "source": "synthetic"}


class AdaptivePortfolioTests(unittest.TestCase):
    def test_combined_sleeves_are_fixed_weight_and_unlevered(self):
        idx = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        a = pd.Series([10000, 11000, 12000], index=idx)
        b = pd.Series([10000, 9000, 8000], index=idx)
        combined = ap.combine_sleeves({"a": a, "b": b}, {"a": 0.4, "b": 0.6})
        self.assertAlmostEqual(combined.iloc[-1], 9600.0)

    def test_cost_gate_requires_edge_above_three_times_all_in_cost(self):
        self.assertFalse(ap.cost_gate(100.0, 0.2, tp_mult=3.0))
        self.assertTrue(ap.cost_gate(100.0, 1.0, tp_mult=3.0))

    def test_adaptive_baseline_matches_frozen_simulator(self):
        one = prepared_market()
        data = {s: {k: v.copy() if hasattr(v, "copy") else v for k, v in one.items()}
                for s in wf.BASE_WEIGHTS}
        start, end = one["df"].index[0], one["df"].index[-1]
        frozen = wf.simulate_portfolio(data, list(wf.BASE_WEIGHTS), start, end, 3.0,
                                       risk_per_trade=0.01, regime_filter=True)
        adaptive = ap.simulate_tactical(data, list(wf.BASE_WEIGHTS), start, end)
        self.assertAlmostEqual(frozen["final_capital"], adaptive["final_capital"], places=8)

    def test_btc_core_uses_close_signal_at_next_open(self):
        one = prepared_market(rows=205, signal_start=204)
        # First valid risk-on close is day 200; entry can only occur day 201 open.
        data = {"BTCUSDT": one}
        result = ap.simulate_btc_core(data, one["df"].index[0], one["df"].index[-1])
        eq = result["equity_curve"]
        first_valid = one["sma200"].first_valid_index()
        self.assertAlmostEqual(eq.loc[first_valid], ap.INITIAL_CAPITAL)
        self.assertGreaterEqual(result["trades"], 1)

    def test_range_state_rejects_monotonic_trend(self):
        idx = pd.date_range("2023-01-01", periods=260, freq="D", tz="UTC")
        close = pd.Series(np.arange(100.0, 360.0), index=idx)
        daily = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                              "close": close, "volume": 1000.0}, index=idx)
        state = rg.daily_range_state(daily)
        self.assertFalse(bool(state["range_on"].iloc[-1]))

    def test_dqn_audit_produces_finite_lagged_curve(self):
        one = prepared_market(rows=260)
        audit = audit_dqn_reward_timing(one["df"], lookback=30)
        self.assertTrue(np.isfinite(audit["lagged_next_bar"]["final_capital"]))
        self.assertGreater(len(audit["lagged_equity"]), 0)


if __name__ == "__main__":
    unittest.main()

