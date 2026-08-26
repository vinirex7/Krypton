import unittest

import cross_asset_diagnostics as cad


class CrossAssetDiagnosticsTests(unittest.TestCase):
    def test_opportunity_label_uses_realized_60d_upside_only_for_diagnostics(self):
        broad = {
            "BTCUSDT": {"best_60d_return": 0.25},
            "SOLUSDT": {"best_60d_return": 0.40},
            "BNBUSDT": {"best_60d_return": 0.10},
        }
        narrow = {
            "BTCUSDT": {"best_60d_return": 0.12},
            "SOLUSDT": {"best_60d_return": 0.30},
            "BNBUSDT": {"best_60d_return": 0.08},
        }
        low = {
            "BTCUSDT": {"best_60d_return": 0.12},
            "SOLUSDT": {"best_60d_return": 0.19},
            "BNBUSDT": {"best_60d_return": 0.08},
        }
        self.assertEqual(cad.classify_opportunity(broad), "broad_long_opportunity")
        self.assertEqual(cad.classify_opportunity(narrow), "narrow_long_opportunity")
        self.assertEqual(cad.classify_opportunity(low), "low_long_opportunity")


if __name__ == "__main__":
    unittest.main()
