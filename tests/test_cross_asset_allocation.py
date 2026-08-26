import unittest
import numpy as np
import pandas as pd

import cross_asset_allocation as ca


def market(mult=1.0, rows=280):
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100, 180 * mult, rows), index=idx)
    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
    }, index=idx)
    return {
        "df": df,
        "sma200": close.rolling(200, min_periods=200).mean(),
    }


class CrossAssetAllocationTests(unittest.TestCase):
    def setUp(self):
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        self.data = {
            "BTCUSDT": market(1.00),
            "ETHUSDT": market(1.05),
            "SOLUSDT": market(1.20),
            "BNBUSDT": market(0.95),
        }

    def test_weights_are_long_only_and_unlevered(self):
        ts = self.data["BTCUSDT"]["df"].index[-1]
        w = ca.target_weights(self.data, self.symbols, ts, target_vol=0.20, top_n=2)
        self.assertLessEqual(sum(w.values()), 1.0 + 1e-12)
        self.assertTrue(all(v >= 0 for v in w.values()))
        self.assertLessEqual(sum(v > 0 for v in w.values()), 2)

    def test_execution_is_after_signal(self):
        start = self.data["BTCUSDT"]["df"].index[210]
        end = self.data["BTCUSDT"]["df"].index[-1]
        out = ca.simulate_allocator(
            self.data, self.symbols, start, end,
            target_vol=0.20, top_n=2, rebalance_days=7,
        )
        log = out["rebalance_log"]
        self.assertFalse(log.empty)
        sig = pd.to_datetime(log["signal_time"], utc=True)
        exe = pd.to_datetime(log["execution_time"], utc=True)
        self.assertTrue(bool((exe > sig).all()))

    def test_cash_when_no_asset_above_sma(self):
        ts = self.data["BTCUSDT"]["df"].index[-1]
        bad = {}
        for s, item in self.data.items():
            bad[s] = dict(item)
            bad[s]["sma200"] = item["df"]["close"] * 2.0
        w = ca.target_weights(bad, self.symbols, ts, target_vol=0.20, top_n=2)
        self.assertEqual(sum(w.values()), 0.0)


if __name__ == "__main__":
    unittest.main()
