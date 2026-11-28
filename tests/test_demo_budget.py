import sqlite3
import unittest
from unittest.mock import Mock, patch

from demo_budget import configure_budget
from krypton import Runner, State
from strategy import D


class DemoBudget(unittest.TestCase):
    @patch("demo_budget.initial_budget", return_value=(D("20"), D("5"), "2026-11-27"))
    def test_allocation_pnl_and_restart(self, quote):
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        api = Mock()
        api.position.return_value = (0, 0)
        api.open_orders.return_value = []
        api.balances.return_value = (D(10000), D(10000))
        state = State(db)
        configure_budget(api, state)
        resumed = State(db)
        configure_budget(api, resumed)
        quote.assert_called_once()
        api.bar_open.return_value = D(100)
        runner = Runner(api, resumed, "KSL2")
        self.assertEqual(runner.record_equity(D(0), D(0), 0), (20, 20, 20))
        api.balances.return_value = (D(9998), D(9998))
        self.assertEqual(runner.record_equity(D(0), D(0), 0), (18, 20, 18))
