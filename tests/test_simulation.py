import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import deep_validation as dv
import walk_forward as wf
from binance_client import BinanceInterface
from risk_manager import RiskManager


def market_frame(rows):
    idx = pd.date_range("2025-01-01", periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close", "volume"])


def prepared(rows, signals, atr=5.0):
    df = market_frame(rows)
    return {
        "BTCUSDT": {
            "df": df,
            "signals": pd.Series(signals, index=df.index, dtype=int),
            "atr": pd.Series(atr, index=df.index, dtype=float),
            "sma200": pd.Series(1.0, index=df.index),
        }
    }


class SimulationTests(unittest.TestCase):
    def test_signal_exit_at_open_precedes_intraday_stop(self):
        data = prepared([
            (100, 101, 99, 100, 1),
            (100, 105, 95, 102, 1),
            (110, 120, 50, 115, 1),
        ] + [(115, 116, 114, 115, 1)] * 18, [1, 0, 0] + [0] * 18)
        m = wf.simulate_portfolio(data, ["BTCUSDT"], data["BTCUSDT"]["df"].index[0],
                                  data["BTCUSDT"]["df"].index[-1], 3.0,
                                  regime_filter=False, weights={"BTCUSDT": 1.0})
        self.assertEqual(m["trade_log"].iloc[0]["reason"], "Sig")
        self.assertGreater(m["trade_log"].iloc[0]["pnl"], 0)

    def test_stop_gap_fills_near_open_not_at_stop(self):
        data = prepared([
            (100, 101, 99, 100, 1),
            (100, 104, 95, 101, 1),
            (80, 85, 70, 82, 1),
        ] + [(82, 83, 81, 82, 1)] * 18, [1, 1, 1] + [0] * 18)
        m = wf.simulate_portfolio(data, ["BTCUSDT"], data["BTCUSDT"]["df"].index[0],
                                  data["BTCUSDT"]["df"].index[-1], 3.0,
                                  regime_filter=False, weights={"BTCUSDT": 1.0})
        self.assertEqual(m["trade_log"].iloc[0]["reason"], "SL_GAP")

    def test_position_sizing_respects_weight_and_cash(self):
        sizing = RiskManager(10_000).calculate_position_size(
            10_000, 100, 5, allocation_pct=0.20, available_cash=1_500
        )
        self.assertLessEqual(sizing["notional"], 1_500)

    def test_portfolio_block_bootstrap_is_finite(self):
        mc = dv.block_bootstrap_monte_carlo(pd.Series([0.01, -0.005, 0.002] * 20), runs=100, seed=7)
        self.assertTrue(np.isfinite(mc["median_return"]))
        self.assertGreaterEqual(mc["p95_adverse_dd"], -1.0)

    def test_open_kline_is_removed(self):
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        closed = [now_ms - 200_000, "1", "2", "0.5", "1.5", "10", now_ms - 100_000,
                  "0", 1, "0", "0", "0"]
        current = [now_ms - 50_000, "1", "2", "0.5", "1.5", "10", now_ms + 50_000,
                   "0", 1, "0", "0", "0"]
        client = BinanceInterface.__new__(BinanceInterface)
        client.client = unittest.mock.Mock()
        client.client.get_klines.return_value = [closed, current]
        df = client.get_ohlcv("BTCUSDT", closed_only=True)
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
