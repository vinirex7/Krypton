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


if __name__ == "__main__":
    unittest.main()
