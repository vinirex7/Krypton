import unittest

import pandas as pd

import cross_asset_diagnostics as cad


class CrossAssetDiagnosticsTests(unittest.TestCase):
    def test_trend_efficiency_is_one_for_monotonic_path(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(cad._trend_efficiency(close), 1.0)

    def test_btc_on_confirmation_measures_two_of_three_breadth(self):
        daily = pd.DataFrame({
            "above_BTCUSDT": [1.0, 1.0, 1.0, 0.0],
            "sma_breadth": [1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            "momentum90_breadth": [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
        })
        out = cad._btc_on_confirmation(daily)
        self.assertAlmostEqual(out["btc_on_days_confirmed_2of3"], 2.0 / 3.0)
        self.assertAlmostEqual(out["btc_on_days_weak_breadth"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
