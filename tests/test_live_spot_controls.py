import tempfile
import unittest
from pathlib import Path

from binance_client import BinanceInterface
from live_c import LiveAggressiveCTradeBot


class FakeSpotClient:
    def __init__(self):
        self.margin_calls = 0

    def get_account(self):
        return {
            "balances": [
                {"asset": "USDT", "free": "123.45", "locked": "6.55"},
                {"asset": "BTC", "free": "0.001", "locked": "0.002"},
            ]
        }

    def get_margin_account(self):
        self.margin_calls += 1
        raise AssertionError("Spot-only Krypton must never query Margin")


class FakeOrderClient:
    def __init__(self):
        self.orders = []
        self.oco_orders = []

    def create_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"orderId": len(self.orders)}

    def create_oco_order(self, **kwargs):
        self.oco_orders.append(kwargs)
        return {"orderListId": len(self.oco_orders)}


class SpotOnlyAndManualControlTests(unittest.TestCase):
    def test_balance_reads_spot_only(self):
        fake = FakeSpotClient()
        b = BinanceInterface.__new__(BinanceInterface)
        b.client = fake
        self.assertEqual(b.get_asset_balance("BTC"), {"free": 0.001, "locked": 0.002, "total": 0.003})
        self.assertAlmostEqual(b.get_account_balance("USDT"), 123.45)
        self.assertAlmostEqual(b.get_asset_total("USDT"), 130.00)
        self.assertEqual(fake.margin_calls, 0)
        self.assertFalse(hasattr(BinanceInterface, "get_margin_capital_snapshot"))
        self.assertFalse(hasattr(BinanceInterface, "_ensure_spot_quote"))

    def test_manual_request_is_one_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "decision-now"
            request.touch()
            self.assertTrue(LiveAggressiveCTradeBot._consume_request(request))
            self.assertFalse(request.exists())
            self.assertFalse(LiveAggressiveCTradeBot._consume_request(request))

    def test_order_quantities_are_fixed_point_strings(self):
        fake = FakeOrderClient()
        b = BinanceInterface.__new__(BinanceInterface)
        b.client = fake
        info = {"step_size": 0.00000001, "tick_size": 0.01, "min_notional": 0.0}

        b.place_limit_order("BTCUSDT", "BUY", 0.000000039, 80_000, info)
        b.place_market_order("BTCUSDT", "SELL", 0.000000039, info)
        b.create_oco_order("BTCUSDT", 0.000000039, 84_874.69, 73_151.09, info)

        quantities = [fake.orders[0]["quantity"], fake.orders[1]["quantity"], fake.oco_orders[0]["quantity"]]
        self.assertEqual(quantities, ["0.00000003"] * 3)
        self.assertTrue(all(isinstance(qty, str) and "e" not in qty.lower() for qty in quantities))


if __name__ == "__main__":
    unittest.main()
