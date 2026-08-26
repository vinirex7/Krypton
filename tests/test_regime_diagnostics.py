import unittest

import pandas as pd

from regime_diagnostics import classify_period, exposure_ratio


class RegimeDiagnosticsTests(unittest.TestCase):
    def test_classify_period(self):
        self.assertEqual(classify_period(0.25), "bull")
        self.assertEqual(classify_period(-0.25), "bear")
        self.assertEqual(classify_period(0.05), "sideways")

    def test_exposure_ratio(self):
        exposure = pd.DataFrame({
            "BTCUSDT": [1, 0, 1],
            "SOLUSDT": [0, 0, 1],
            "BNBUSDT": [0, 0, 0],
        })
        # Simultaneous positions: 1/3, 0/3, 2/3 => mean = 1/3.
        self.assertAlmostEqual(exposure_ratio(exposure, 3), 1.0 / 3.0)

    def test_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            classify_period(0.0, bull_threshold=-0.2, bear_threshold=0.2)


if __name__ == "__main__":
    unittest.main()
