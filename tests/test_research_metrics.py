import unittest

import numpy as np
import pandas as pd

from research_metrics import concentration_metrics, deflated_sharpe_ratio, white_reality_check


class ResearchMetricsTests(unittest.TestCase):
    def test_concentration_metrics_remove_best_trade_and_blocks(self):
        idx = pd.date_range("2025-01-01", periods=25, tz="UTC")
        eq = pd.Series(10_000.0 + np.arange(25) * 10.0, index=idx)
        trades = pd.DataFrame({"pnl": [50.0, 100.0, -20.0, 30.0]})
        out = concentration_metrics(trades, eq)
        self.assertAlmostEqual(out["best_trade_pnl"], 100.0)
        self.assertGreaterEqual(out["best_20d_pnl"], out["best_5d_pnl"])
        self.assertIn("top_5pct_trade_pnl_share", out)

    def test_dsr_is_probability(self):
        r = pd.Series([0.001, -0.0002, 0.0015, 0.0004, 0.0008] * 40)
        dsr = deflated_sharpe_ratio(r, n_trials=4)
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)

    def test_white_reality_check_is_deterministic(self):
        idx = pd.date_range("2025-01-01", periods=80, tz="UTC")
        candidates = {
            "a": pd.Series(np.full(80, 0.001), index=idx),
            "b": pd.Series(np.full(80, 0.0005), index=idx),
        }
        out1 = white_reality_check(candidates, bootstrap_samples=100, seed=7)
        out2 = white_reality_check(candidates, bootstrap_samples=100, seed=7)
        self.assertEqual(out1, out2)
        self.assertEqual(out1["winner"], "a")
        self.assertGreaterEqual(out1["p_value"], 0.0)
        self.assertLessEqual(out1["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
