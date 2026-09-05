import unittest

import numpy as np
import pandas as pd

import aggressive_c_live as c
from config import (
    AGGRESSIVE_C_ALPHA_SYMBOLS,
    AGGRESSIVE_C_ALPHA_TARGET_VOL,
    AGGRESSIVE_C_ALPHA_WEIGHT,
    AGGRESSIVE_C_MAX_DRAWDOWN_PCT,
    AGGRESSIVE_C_TACTICAL_RISK_PER_TRADE,
    AGGRESSIVE_C_TACTICAL_WEIGHT,
)
from risk_manager import RiskManager


class AggressiveCLiveTests(unittest.TestCase):
    @staticmethod
    def _frame(trend, phase=0.0, periods=400):
        idx = pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC")
        t = np.arange(periods, dtype=float)
        # Deterministic trend with non-zero realized volatility.
        close = 100.0 * np.exp(trend * t + 0.012 * np.sin(t / 5.0 + phase))
        return pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(periods, 1000.0),
            },
            index=idx,
        )

    def test_profile_matches_promoted_c(self):
        p = c.frozen_profile()
        self.assertEqual(p["name"], "AGGRESSIVE_C")
        self.assertAlmostEqual(AGGRESSIVE_C_TACTICAL_WEIGHT, 0.55)
        self.assertAlmostEqual(AGGRESSIVE_C_ALPHA_WEIGHT, 0.45)
        self.assertAlmostEqual(AGGRESSIVE_C_TACTICAL_RISK_PER_TRADE, 0.02)
        self.assertAlmostEqual(AGGRESSIVE_C_ALPHA_TARGET_VOL, 0.30)
        self.assertAlmostEqual(AGGRESSIVE_C_MAX_DRAWDOWN_PCT, 0.30)
        self.assertEqual(tuple(AGGRESSIVE_C_ALPHA_SYMBOLS), ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"))
        self.assertEqual(len(c.strategy_fingerprint()), 64)

    def test_alpha_selects_two_and_never_leverages(self):
        frames = {
            "BTCUSDT": self._frame(0.0030, 0.0),
            "ETHUSDT": self._frame(0.0024, 0.7),
            "SOLUSDT": self._frame(-0.0010, 1.4),
            "BNBUSDT": self._frame(0.0007, 2.1),
        }
        w = c.alpha_target_weights(frames)
        active = [s for s, x in w.items() if x > 0]
        self.assertEqual(len(active), 2)
        self.assertIn("BTCUSDT", active)
        self.assertIn("ETHUSDT", active)
        self.assertLessEqual(sum(w.values()), 1.0 + 1e-12)

    def test_alpha_breadth_gate_goes_to_cash(self):
        frames = {
            "BTCUSDT": self._frame(0.0020, 0.0),
            "ETHUSDT": self._frame(-0.0010, 0.7),
            "SOLUSDT": self._frame(-0.0012, 1.4),
            "BNBUSDT": self._frame(-0.0008, 2.1),
        }
        w = c.alpha_target_weights(frames)
        self.assertTrue(all(abs(x) < 1e-12 for x in w.values()))

    def test_continuity_blocks_stale_deterioration(self):
        idx = pd.date_range("2026-01-01", periods=130, freq="D", tz="UTC")
        # Strong first 100d rise, then 30d pullback: 30d momentum negative while
        # 90d remains positive. All tactical assets share the pattern.
        up = np.linspace(100.0, 170.0, 100)
        down = np.linspace(170.0, 150.0, 30)
        values = np.concatenate([up, down])
        closes = pd.DataFrame({s: values for s in ("SOLUSDT", "BTCUSDT", "BNBUSDT")}, index=idx)
        signals = {s: pd.Series(1, index=idx, dtype=int) for s in closes.columns}
        self.assertFalse(c.continuity_allowed(signals, closes, "BTCUSDT", idx[-1]))

    def test_risk_manager_uses_c_override(self):
        rm = RiskManager(10_000.0, risk_per_trade=0.02, max_drawdown_pct=0.30)
        sizing = rm.calculate_position_size(10_000.0, 100.0, 5.0, allocation_pct=1.0, available_cash=10_000.0)
        self.assertAlmostEqual(sizing["risk_amount_usd"], 200.0)
        self.assertFalse(rm.check_max_drawdown(7_100.0))
        self.assertTrue(rm.check_max_drawdown(6_900.0))

    def test_cost_gate_rejects_tiny_edge(self):
        self.assertFalse(c.cost_gate(100.0, 0.05))
        self.assertTrue(c.cost_gate(100.0, 1.0))

    @staticmethod
    def _reconciliation_bot(balances, *, active_oco=False, order_list_id=None):
        class FakeBinance:
            def __init__(self):
                self.balances = balances
                self.created_oco = []
                self.active_oco = active_oco

            def get_asset_balance(self, asset):
                free, locked = self.balances.get(asset, (0.0, 0.0))
                return {"free": free, "locked": locked, "total": free + locked}

            def get_account_balance(self, asset):
                return self.get_asset_balance(asset)["free"]

            def get_current_price(self, symbol):
                return 100.0

            def has_active_oco(self, symbol, oid):
                return self.active_oco and oid is not None

            def cancel_oco_order(self, symbol, oid):
                self.active_oco = False
                return True

            def create_oco_order(self, **kwargs):
                self.created_oco.append(kwargs)
                self.active_oco = True
                return {"orderListId": 999}

        bot = c.AggressiveCTradeBot.__new__(c.AggressiveCTradeBot)
        bot.symbols = ["BTCUSDT"]
        bot.symbol_infos = {"BTCUSDT": {"step_size": 0.0001, "min_notional": 10.0}}
        bot.binance = FakeBinance()
        bot.state = {
            "tactical_cash": 550.0,
            "alpha_cash": 450.0,
            "tactical_positions": {
                "BTCUSDT": {
                    "quantity": 4.0,
                    "entry_price": 90.0,
                    "stop_loss": 80.0,
                    "take_profit": 120.0,
                    "order_list_id": order_list_id,
                }
            },
            "alpha_qty": {"BTCUSDT": 6.0},
            "portfolio_peak": 2000.0,
            "portfolio_daily_start": 2000.0,
        }
        bot.tactical_risk = RiskManager(1000.0, risk_per_trade=0.02, max_drawdown_pct=0.30)
        bot.protection_blocked = False
        bot._save_state = lambda: None
        return bot

    def test_manual_alpha_sale_with_active_oco_is_reconciled(self):
        bot = self._reconciliation_bot({"BTC": (3.0, 4.0), "USDT": (1300.0, 0.0)}, active_oco=True, order_list_id=123)
        bot._reconcile_exchange_state()
        self.assertAlmostEqual(bot._tactical_qty("BTCUSDT"), 4.0)
        self.assertAlmostEqual(bot._alpha_qty("BTCUSDT"), 3.0)
        self.assertAlmostEqual(bot.state["tactical_cash"], 550.0)
        self.assertAlmostEqual(bot.state["alpha_cash"], 750.0)
        self.assertFalse(bot.protection_blocked)

    def test_unclassified_manual_sale_reduces_both_sleeves_pro_rata(self):
        bot = self._reconciliation_bot({"BTC": (5.0, 0.0), "USDT": (1500.0, 0.0)})
        bot._reconcile_exchange_state()
        self.assertAlmostEqual(bot._tactical_qty("BTCUSDT"), 2.0)
        self.assertAlmostEqual(bot._alpha_qty("BTCUSDT"), 3.0)
        self.assertAlmostEqual(bot.state["tactical_cash"], 750.0)
        self.assertAlmostEqual(bot.state["alpha_cash"], 750.0)
        self.assertEqual(len(bot.binance.created_oco), 1)
        self.assertFalse(bot.protection_blocked)

    def test_manual_usdt_reduction_scales_cash_and_rebases_risk(self):
        bot = self._reconciliation_bot({"BTC": (6.0, 4.0), "USDT": (400.0, 0.0)}, active_oco=True, order_list_id=123)
        bot._reconcile_exchange_state()
        self.assertAlmostEqual(bot.state["tactical_cash"], 220.0)
        self.assertAlmostEqual(bot.state["alpha_cash"], 180.0)
        self.assertAlmostEqual(bot.state["portfolio_peak"], 1400.0)
        self.assertAlmostEqual(bot.tactical_risk.peak_capital, 670.0)
        self.assertFalse(bot.state.get("portfolio_halted", False))


if __name__ == "__main__":
    unittest.main()
