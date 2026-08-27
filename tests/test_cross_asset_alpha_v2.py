import unittest
import numpy as np
import pandas as pd
import cross_asset_alpha_v2 as av2

class AlphaV2Tests(unittest.TestCase):
    def _data(self, n=520):
        idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"); data = {}
        for j, s in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]):
            r = 0.0010 + j * 0.00015 + 0.002 * np.sin(np.arange(n) / (17 + j))
            close = 100 * np.cumprod(1 + r)
            df = pd.DataFrame({"open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
                               "close": close, "volume": 1000.0}, index=idx)
            data[s] = {"df": df, "sma200": df["close"].rolling(200).mean()}
        return data, list(data), idx

    def test_full_v2_is_long_only_and_bounded(self):
        data, symbols, idx = self._data()
        w = av2.target_weights_v2(data, symbols, idx[-1], target_vol=0.15, top_n=2, min_selected=2,
            use_tsmom=True, use_vol_regime=True, use_dispersion=True, use_persistent_regime=True)
        self.assertTrue(all(v >= 0 for v in w.values())); self.assertLessEqual(sum(w.values()), 1.0 + 1e-12)
        self.assertLessEqual(sum(v > 0 for v in w.values()), 2)

    def test_tsmom_rejects_recent_negative_window(self):
        data, symbols, idx = self._data(); s = "SOLUSDT"; df = data[s]["df"].copy()
        df.loc[idx[-29]:, "close"] *= np.linspace(0.99, 0.70, 29); df.loc[idx[-29]:, "open"] = df.loc[idx[-29]:, "close"]
        data[s]["df"] = df; data[s]["sma200"] = df["close"].rolling(200).mean()
        w = av2.target_weights_v2(data, symbols, idx[-1], use_tsmom=True, top_n=4, min_selected=1)
        self.assertEqual(w[s], 0.0)

    def test_simulation_executes_after_signal(self):
        data, symbols, idx = self._data()
        result = av2.simulate_alpha_v2(data, symbols, idx[250], idx[-1], rebalance_days=45,
            use_tsmom=True, use_vol_regime=True, use_dispersion=True, use_persistent_regime=True)
        log = result["rebalance_log"]
        if not log.empty:
            sig = pd.to_datetime(log["signal_time"], utc=True); exe = pd.to_datetime(log["execution_time"], utc=True)
            self.assertTrue(bool((exe > sig).all()))

if __name__ == "__main__": unittest.main()
