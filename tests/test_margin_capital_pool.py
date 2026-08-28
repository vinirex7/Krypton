import unittest

from binance_client import BinanceInterface


class FakeClient:
    def __init__(
        self,
        *,
        spot_usdt=10.0,
        margin_free=20.0,
        margin_locked=0.0,
        borrowed=0.0,
        interest=0.0,
        max_transferable=20.0,
        total_liability_btc=0.0,
    ):
        self.spot_usdt = float(spot_usdt)
        self.margin_free = float(margin_free)
        self.margin_locked = float(margin_locked)
        self.borrowed = float(borrowed)
        self.interest = float(interest)
        self.max_transferable = float(max_transferable)
        self.total_liability_btc = float(total_liability_btc)
        self.transfers = []

    def get_account(self):
        return {
            "balances": [
                {"asset": "USDT", "free": str(self.spot_usdt), "locked": "0"},
                {"asset": "BTC", "free": "0", "locked": "0"},
            ]
        }

    def get_margin_account(self):
        net = self.margin_free + self.margin_locked - self.borrowed - self.interest
        return {
            "totalLiabilityOfBtc": str(self.total_liability_btc),
            "userAssets": [
                {
                    "asset": "USDT",
                    "free": str(self.margin_free),
                    "locked": str(self.margin_locked),
                    "borrowed": str(self.borrowed),
                    "interest": str(self.interest),
                    "netAsset": str(net),
                }
            ],
        }

    def get_max_margin_transfer(self, **params):
        self.assert_asset(params)
        return {"amount": str(self.max_transferable)}

    @staticmethod
    def assert_asset(params):
        if params.get("asset") != "USDT":
            raise AssertionError(params)

    def make_universal_transfer(self, **params):
        if params.get("type") != "MARGIN_MAIN" or params.get("asset") != "USDT":
            raise AssertionError(params)
        amount = float(params["amount"])
        if amount > self.margin_free + 1e-12:
            raise AssertionError("transfer above margin free")
        self.margin_free -= amount
        self.max_transferable = max(0.0, self.max_transferable - amount)
        self.spot_usdt += amount
        self.transfers.append(amount)
        return {"tranId": 123456}


def interface(fake, enabled=True):
    obj = BinanceInterface.__new__(BinanceInterface)
    obj.client = fake
    obj.margin_pool_enabled = enabled
    return obj


class MarginCapitalPoolTests(unittest.TestCase):
    def test_pool_counts_spot_plus_own_transferable_margin(self):
        b = interface(FakeClient(spot_usdt=10, margin_free=20, max_transferable=12))
        snap = b.get_margin_capital_snapshot("USDT")
        self.assertAlmostEqual(snap["available_own"], 12.0)
        self.assertAlmostEqual(b.get_account_balance("USDT"), 22.0)

    def test_any_cross_margin_liability_disables_margin_pool_capital(self):
        b = interface(
            FakeClient(
                spot_usdt=10,
                margin_free=50,
                borrowed=5,
                max_transferable=40,
                total_liability_btc=0.001,
            )
        )
        snap = b.get_margin_capital_snapshot("USDT")
        self.assertTrue(snap["has_any_liability"])
        self.assertEqual(snap["available_own"], 0.0)
        self.assertAlmostEqual(b.get_account_balance("USDT"), 10.0)

    def test_disabled_pool_preserves_original_spot_semantics(self):
        b = interface(FakeClient(spot_usdt=7, margin_free=100), enabled=False)
        self.assertAlmostEqual(b.get_account_balance("USDT"), 7.0)

    def test_ensure_spot_quote_moves_only_required_own_cash(self):
        fake = FakeClient(spot_usdt=3, margin_free=20, max_transferable=20)
        b = interface(fake)
        self.assertTrue(b._ensure_spot_quote(8.0))
        self.assertEqual(len(fake.transfers), 1)
        self.assertAlmostEqual(fake.transfers[0], 5.0)
        self.assertAlmostEqual(fake.spot_usdt, 8.0)
        self.assertAlmostEqual(fake.margin_free, 15.0)

    def test_ensure_spot_quote_fails_if_own_margin_is_insufficient(self):
        fake = FakeClient(spot_usdt=3, margin_free=2, max_transferable=2)
        b = interface(fake)
        self.assertFalse(b._ensure_spot_quote(8.0))
        self.assertEqual(fake.transfers, [])


if __name__ == "__main__":
    unittest.main()
