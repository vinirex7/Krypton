import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import aggressive_c_live as live
import live_validation as v


class LiveReplayTests(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2023-01-01", periods=430, tz="UTC")
        self.frames, self.filters = {}, {}
        for i, s in enumerate(v.SYMBOLS):
            t = np.arange(len(idx))
            close = 100 * np.exp(.001 * t + .015 * np.sin(t / 7 + i))
            frame = pd.DataFrame({"open": close * .999, "high": close * 1.01,
                                  "low": close * .99, "close": close, "volume": 1000.}, index=idx)
            frame.attrs["symbol"] = s
            self.frames[s] = frame
            self.filters[s] = {"step_size": .00001, "tick_size": .00001, "min_notional": 5.}
        self.start, self.end = idx[400], idx[402]

    def test_replay_calls_actual_live_daily_cycle_and_never_constructs_client(self):
        original = live.AggressiveCTradeBot.daily_cycle
        calls = []

        def observe(bot):
            calls.append(bot)
            return original(bot)

        with patch.object(live.AggressiveCTradeBot, "daily_cycle", observe), \
                patch.object(v.BinanceInterface, "__init__", side_effect=AssertionError("Network client!")):
            result = v.replay(self.frames, self.filters, self.start, self.end)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0].tactical_risk.risk_per_trade, .02)
        self.assertEqual(calls[0].tactical_risk.max_drawdown_pct, .30)
        self.assertEqual(result["state"]["strategy_fingerprint"], live.strategy_fingerprint())
        self.assertGreater(result["fees"], 0)

    def test_completed_history_is_399_and_excludes_execution_day(self):
        ex = v.HistoricalExchange(self.frames, self.filters, 10000.)
        ex.ts = self.start
        frame = ex.get_ohlcv("BTCUSDT", limit=400)
        self.assertEqual(len(frame), 399)
        self.assertEqual(frame.index[-1], self.start - pd.Timedelta(days=1))

    def test_future_prices_do_not_change_past_equity_or_orders(self):
        baseline = v.replay(self.frames, self.filters, self.start, self.end)
        for frame in self.frames.values():
            frame.loc[frame.index > self.end, ["open", "high", "low", "close"]] *= 10
        changed = v.replay(self.frames, self.filters, self.start, self.end)
        pd.testing.assert_series_equal(baseline["equity_curve"], changed["equity_curve"])
        pd.testing.assert_frame_equal(baseline["orders"], changed["orders"])

    def test_live_signal_price_gate(self):
        ex = v.HistoricalExchange(self.frames, self.filters, 10000.)
        ex.ts = self.start
        ex.prices = {s: 100. for s in v.SYMBOLS}
        self.assertIsNone(ex.place_limit_order("BTCUSDT", "BUY", 1., 101., self.filters["BTCUSDT"], reference_price=100.))
        self.assertEqual(ex.cash, 10000.)

    def test_oco_double_touch_chooses_stop_and_gap_uses_open(self):
        ex = v.HistoricalExchange(self.frames, self.filters, 10000.)
        ex.ts = self.start
        ex.prices = {s: 100. for s in v.SYMBOLS}
        ex.qty["BTCUSDT"] = 1.
        ex.create_oco_order("BTCUSDT", 1., 110., 90., self.filters["BTCUSDT"])
        rows = {"BTCUSDT": pd.Series({"open": 100., "high": 115., "low": 85.})}
        ex.process_ocos(rows)
        self.assertEqual(ex.orders[-1]["kind"], "SL")
        self.assertAlmostEqual(ex.orders[-1]["price"], 90 * (1 - v.config.EXIT_SLIPPAGE_PCT))
        ex.qty["BTCUSDT"] = 1.
        ex.create_oco_order("BTCUSDT", 1., 110., 90., self.filters["BTCUSDT"])
        rows["BTCUSDT"]["open"] = 80.
        ex.process_ocos(rows, gaps_only=True)
        self.assertEqual(ex.orders[-1]["kind"], "SL_GAP")
        self.assertAlmostEqual(ex.orders[-1]["price"], 80 * (1 - v.config.EXIT_SLIPPAGE_PCT))

    def test_walk_forward_includes_initial_cost_and_partial_last_fold(self):
        curve = pd.Series([10000., 9900., 10000., 9800., 9700.],
                          index=pd.date_range("2025-01-01", periods=5, tz="UTC"))
        folds, compounded = v.walk_forward(curve, "2025-01-01", "2025-01-04", test_days=3)
        self.assertEqual(list(folds.test_days), [3, 1])
        self.assertAlmostEqual(compounded, -.03)
        self.assertAlmostEqual(v.metrics(curve)["max_drawdown"], -.03)

    def test_missing_data_is_an_error(self):
        frame = self.frames["BTCUSDT"].loc[self.start:self.end].iloc[1:]
        with self.assertRaisesRegex(ValueError, "missing"):
            v.validate_frame(frame, self.start, self.end, "BTCUSDT")

    def test_calendar_periods(self):
        anchor = v.utc("2026-10-28")
        self.assertEqual(str(v.period_start(anchor, "10m").date()), "2025-12-28")
        self.assertEqual(str(v.period_start(anchor, "4y").date()), "2022-10-28")

    def test_runtime_reconciliation_failure_halts_instead_of_fabricating_trades(self):
        with patch.object(v.ReplayBot, "_reconcile_exchange_state", side_effect=RuntimeError("unmanaged balance")):
            result = v.replay(self.frames, self.filters, self.start, self.end)
        self.assertEqual(result["orders_count"], 0)
        self.assertEqual(result["final_capital"], 10000.)
        self.assertEqual(result["runtime_error_count"], 6)
        self.assertTrue(result["state"]["portfolio_halted"])
        self.assertEqual(result["halt_date"], str(self.start.date()))


if __name__ == "__main__":
    unittest.main()
