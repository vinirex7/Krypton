import unittest

import pandas as pd

import forensic_diagnostics as fd
import research_validation as rv


def prepared_three_symbols():
    idx = pd.date_range("2025-01-01", periods=35, freq="D", tz="UTC")
    data = {}
    for symbol, shift in [("SOLUSDT", 0.0), ("BTCUSDT", 1.0), ("BNBUSDT", -1.0)]:
        close = pd.Series([100.0 + shift + (i * 0.2) for i in range(len(idx))], index=idx)
        df = pd.DataFrame({
            "open": close,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": 1.0,
        }, index=idx)
        signals = pd.Series([1] * 12 + [0] * 5 + [1] * 8 + [0] * 10, index=idx, dtype=int)
        data[symbol] = {
            "df": df,
            "signals": signals,
            "atr": pd.Series(3.0, index=idx),
            "sma200": pd.Series(50.0, index=idx),
        }
    return data


class ForensicDiagnosticsTests(unittest.TestCase):
    def test_forensic_simulation_matches_research_simulation(self):
        data = prepared_three_symbols()
        symbols = ["SOLUSDT", "BTCUSDT", "BNBUSDT"]
        weights = {"SOLUSDT": 0.3125, "BTCUSDT": 0.5, "BNBUSDT": 0.1875}
        start = data["BTCUSDT"]["df"].index[0]
        end = data["BTCUSDT"]["df"].index[-1]

        baseline = rv.simulate(
            data, symbols, start, end,
            regime_mode="btc", drawdown_overlay=False, weights=weights,
        )
        forensic = fd.simulate_forensic(
            data, symbols, start, end,
            regime_mode="btc", weights=weights,
        )

        self.assertAlmostEqual(
            baseline["return"], forensic["summary"]["strategy_return"], places=12
        )
        self.assertEqual(baseline["trades"], forensic["summary"]["trades"])
        self.assertIn("mean_gross_exposure", forensic["summary"])
        self.assertIn("binding_cap_counts", forensic["summary"])


if __name__ == "__main__":
    unittest.main()
