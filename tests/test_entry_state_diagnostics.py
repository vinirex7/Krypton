import unittest

import pandas as pd

import entry_state_diagnostics as esd


class EntryStateDiagnosticsTests(unittest.TestCase):
    def test_return_at_uses_only_prior_bars(self):
        close = pd.Series([100.0, 110.0, 121.0, 133.1])
        self.assertAlmostEqual(esd._return_at(close, 3, 2), 0.21)

    def test_crossings_counts_state_changes_in_window(self):
        close = pd.Series([9.0, 11.0, 9.0, 11.0, 12.0])
        sma = pd.Series([10.0] * 5)
        self.assertEqual(esd._crossings(close, sma, 4, 4), 3.0)

    def test_feature_comparison_computes_spearman_without_scipy(self):
        frame = pd.DataFrame({
            "winner": [0, 0, 1, 1],
            "R": [-2.0, -1.0, 1.0, 2.0],
            "asset_mom20": [40.0, 30.0, 20.0, 10.0],
        })
        result = esd._feature_comparison(frame, "test")
        row = result.loc[result["feature"] == "asset_mom20"].iloc[0]
        self.assertAlmostEqual(row["spearman_with_R"], -1.0)

    def test_tp_sl_comparison_excludes_other_exits(self):
        frame = pd.DataFrame({
            "exit_class": ["SL", "TP", "OTHER"],
            "asset_mom20": [-0.2, 0.4, 99.0],
        })
        result = esd.tp_sl_comparison(frame, "test")
        row = result.loc[result["feature"] == "asset_mom20"].iloc[0]
        self.assertEqual(row["tp_n"], 1)
        self.assertEqual(row["sl_n"], 1)
        self.assertAlmostEqual(row["tp_minus_sl"], 0.6)

    def test_trade_history_flags_same_close_reentry_after_stop(self):
        frame = pd.DataFrame({
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "signal_time": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-02-20"], utc=True),
            "entry_time": pd.to_datetime(["2025-01-02", "2025-01-04", "2025-02-21"], utc=True),
            "exit_time": pd.to_datetime(["2025-01-03", "2025-01-10", "2025-02-25"], utc=True),
            "reason": ["SL", "TP", "SL"],
        })
        result = esd.add_trade_history_features(frame)
        self.assertEqual(result.loc[1, "immediate_reentry_after_stop"], 1.0)
        self.assertEqual(result.loc[1, "stops_last_30d"], 1.0)
        self.assertEqual(result.loc[2, "prior_exit_was_stop"], 0.0)
        self.assertEqual(result.loc[2, "stops_last_30d"], 0.0)

    def test_long_signal_age_and_drawdown_use_only_current_and_prior_bars(self):
        signals = pd.Series([0, 1, 1, 1, 0, 1])
        close = pd.Series([10.0, 12.0, 11.0, 9.0, 100.0, 8.0])
        self.assertEqual(esd._long_signal_age(signals, 3), 3.0)
        self.assertEqual(esd._long_signal_age(signals, 4), 0.0)
        self.assertAlmostEqual(esd._drawdown_from_high(close, 3, 3), -0.25)


if __name__ == "__main__":
    unittest.main()
