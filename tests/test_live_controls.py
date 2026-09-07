import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from live_c import LiveAggressiveCTradeBot
from risk_manager import RiskManager
from tradebot import _queue


class LiveControlTests(unittest.TestCase):
    def test_decision_request_is_file_based_and_single_consumer(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / ".decision"
            _queue(request, "TEST")
            self.assertTrue(request.exists())
            self.assertTrue(LiveAggressiveCTradeBot._consume_request(request))
            self.assertFalse(request.exists())
            self.assertFalse(LiveAggressiveCTradeBot._consume_request(request))

    def test_queue_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / ".decision"
            _queue(request, "TEST")
            _queue(request, "TEST")
            self.assertTrue(request.exists())
            self.assertTrue(LiveAggressiveCTradeBot._consume_request(request))
            self.assertFalse(LiveAggressiveCTradeBot._consume_request(request))

    def test_confirmed_manual_change_rebases_false_halt_without_deleting_state(self):
        bot = LiveAggressiveCTradeBot.__new__(LiveAggressiveCTradeBot)
        bot.state = {
            "portfolio_peak": 2000.0,
            "portfolio_daily_start": 2000.0,
            "portfolio_daily_date": "2026-09-04",
            "portfolio_halted": True,
            "tactical_positions": {"BTCUSDT": {"quantity": 0.25}},
            "alpha_qty": {"BTCUSDT": 0.10},
        }
        bot.tactical_risk = RiskManager(1000.0, risk_per_trade=0.02, max_drawdown_pct=0.30)
        bot.tactical_risk.halted = True
        bot.tactical_risk.circuit_breaker = True
        bot._reconcile_exchange_state = lambda: None
        bot._portfolio_equity = lambda: 1115.6
        bot._tactical_equity = lambda: 613.2
        bot._save_state = lambda: None
        bot._record_snapshot_safe = lambda: None

        self.assertTrue(bot._rebase_after_confirmed_manual_change())
        self.assertAlmostEqual(bot.state["portfolio_peak"], 1115.6)
        self.assertAlmostEqual(bot.state["portfolio_daily_start"], 1115.6)
        self.assertFalse(bot.state["portfolio_halted"])
        self.assertEqual(bot.state["tactical_positions"]["BTCUSDT"]["quantity"], 0.25)
        self.assertEqual(bot.state["alpha_qty"]["BTCUSDT"], 0.10)
        self.assertAlmostEqual(bot.tactical_risk.peak_capital, 613.2)
        self.assertEqual(bot.tactical_risk.daily_date, datetime.now(timezone.utc).date())
        self.assertFalse(bot.tactical_risk.halted)
        self.assertFalse(bot.tactical_risk.circuit_breaker)


if __name__ == "__main__":
    unittest.main()
