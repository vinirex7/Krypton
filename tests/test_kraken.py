import sqlite3
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from exchange_errors import ExchangeError
from kraken_futures import API, CONTRACT, KrakenClient, signature
from krypton import Runner, State
from strategy import BAR, D


class KrakenAPI(unittest.TestCase):
    def setUp(self):
        self.api = KrakenClient("demo")
        self.api.request = Mock()

    def test_encoded_signature_matches_openssl(self):
        query = "orderType=mkt&symbol=PF_XBTUSD"
        expected = ("fuTJYDX8mttxEiqarUqFUXLAp6QLdv0N++GBTMKlYaEFJbQgswbSeNfkLu8Gj"
                    "Yzv3AOANIFtL0NgBQM0XxrzQw==")
        self.assertEqual(signature("dGVzdA==", query, API+"sendorder"), expected)

    def test_market_uses_original_public_reference_and_separate_demo_host(self):
        self.assertEqual(self.api.host, "https://demo-futures.kraken.com")
        self.assertEqual(self.api.reference.host, "https://fapi.binance.com")
        self.assertFalse(self.api.reference.key)
        with patch.object(self.api.reference, "market") as original:
            self.assertIs(self.api.market(BAR), original.return_value)
            original.assert_called_once_with(BAR)

    def test_margin_and_contract_constraints(self):
        spec = {"base": "BTC", "quote": "USD", "symbol": CONTRACT, "contractSize": 1,
                "type": "flexible_futures", "tradeable": True, "tickSize": 1,
                "maxPositionSize": 1200, "contractValueTradePrecision": 4}
        self.api.request.return_value = {"instruments": [spec]}
        self.assertEqual(self.api.filters()["LOT_SIZE"]["stepSize"], D(".0001"))
        spec["contractSize"] = 10
        with self.assertRaises(RuntimeError):
            self.api.filters()
        self.api.request.return_value = {"leveragePreferences": [{"symbol": CONTRACT, "maxLeverage": 5}]}
        with self.assertRaises(RuntimeError):
            self.api.validate_account()

    def test_wallet_equity_excludes_mark_pnl(self):
        self.api.request.return_value = {"accounts": {"flex": {"type": "multiCollateralMarginAccount",
            "balanceValue": "100", "unrealizedFunding": "-1", "availableMargin": "80", "pnl": 10}}}
        self.assertEqual(self.api.balances(), (99, 80))

    def test_partial_ioc_uses_weighted_fill_and_deadline(self):
        fills = [{"type": "EXECUTION", "amount": size, "price": price,
                  "orderPriorExecution": {"orderId": "entry"}} for size, price in [(2, 99), (1, 102)]]
        self.api.request.return_value = {"sendStatus": {"status": "partiallyFilled",
                                         "order_id": "entry", "orderEvents": fills}}
        order = self.api.sell("4", "K1316-0")
        self.assertEqual(order, {"status": "FILLED", "executedQty": "3", "avgPrice": D(100)})
        self.assertEqual(self.api.request.call_args.kwargs["processBefore"], "1970-01-01T00:00:30+00:00")

    def test_reject_vs_uncertain_entry(self):
        for code, rejected in [("insufficientAvailableFunds", True), ("clientOrderIdAlreadyExist", False)]:
            self.api.request.return_value = {"sendStatus": {"status": code}}
            with self.assertRaises(ExchangeError) as ctx:
                self.api.sell("1", "K1316-0")
            self.assertEqual(ctx.exception.rejected, rejected)

    def test_history_recovery_after_status_expires(self):
        row = {"clientId": "K1316-0", "quantity": "2", "filled": ".7",
               "tradeable": CONTRACT, "direction": "Sell"}
        self.api.request.side_effect = [{"orders": []}, {"elements": [], "continuationToken": "abc"},
                                        {"elements": [{"event": {"OrderCancelled": {"order": row}}}]}]
        self.assertEqual(self.api.entry_status("K1316-0")["executedQty"], D(".7"))
        self.assertEqual(self.api.request.call_args.kwargs["continuation_token"], "abc")

    def test_foreign_order_blocks_execution(self):
        self.api.request.return_value = {"openOrders": [{"orderType": "lmt", "symbol": CONTRACT}]}
        with self.assertRaises(RuntimeError):
            self.api.protections()


class KrakenRecovery(unittest.TestCase):
    def test_native_bracket_survives_ambiguous_entry_and_restart(self):
        self.check_restart(False)

    def test_long_survives_ambiguous_entry_and_restart(self):
        self.check_restart(True)

    def check_restart(self, long):
        strategy = "KSL2" if long else "KSO1316"
        filled = ".1" if long else ".6"
        entry_side, exit_side = ("buy", "sell") if long else ("sell", "buy")
        api = KrakenClient("demo")
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        bar = 10000*BAR
        api.now = Mock(return_value=bar+500)
        api.filters = Mock(return_value={"PRICE_FILTER": {"tickSize": 1},
            "LOT_SIZE": {"stepSize": ".001", "minQty": ".001", "maxQty": 100},
            "MARKET_LOT_SIZE": {"stepSize": ".001", "minQty": ".001", "maxQty": 100},
            "MIN_NOTIONAL": {"notional": 0}})
        api.price = Mock(return_value=D(100))
        api.bar_open = Mock(return_value=D(100))
        api.balances = Mock(return_value=(D(100), D(100)))
        position = []
        orders = []
        orders_sent = []
        def venue(method, path, **params):
            if path == API+"openpositions":
                return {"openPositions": position}
            if path == API+"openorders":
                return {"openOrders": orders.copy()}
            if path == API+"orders/status":
                return {"orders": [{"order": {"cliOrdId": (f"KSL2-L-{bar}" if long else f"KSO1316-{bar}"), "symbol": CONTRACT,
                                               "side": entry_side, "filled": filled}, "status": "FULLY_EXECUTED"}]}
            if path == API+"sendorder":
                orders_sent.append(params)
                if params["side"] == entry_side:
                    self.assertFalse(State(db).data["trade"]["confirmed"])
                    position.append({"symbol": CONTRACT, "size": filled, "side": "long" if long else "short", "price": 100})
                    raise TimeoutError()
                orders.append({"cliOrdId": params["cliOrdId"], "symbol": CONTRACT, "side": exit_side,
                               "orderType": params["orderType"], "reduceOnly": True, "triggerSignal": "last",
                               "order_id": params["cliOrdId"], "stopPrice": params["stopPrice"],
                               "unfilledSize": params["size"]})
                return {"sendStatus": {"status": "placed"}}
            self.fail(path)
        api.request = Mock(side_effect=venue)
        runner = Runner(api, State(db), strategy)
        runner.market = Mock(return_value=(None, pd.DataFrame({"atr": [2, 3], "funding": [not long, False], "breakout": [False, long],
                                     "bull": [False, long], "short_regime": [False, False],
                                     "squeeze": [False, False], "candle": [False, False]})))
        with self.assertRaises(TimeoutError):
            runner.tick()
        restarted = Runner(api, State(db), strategy)
        restarted.tick()
        restarted.tick()
        self.assertTrue(State(db).data["trade"]["confirmed"])
        self.assertEqual([o["orderType"] for o in orders_sent], ["mkt", "stp", "take_profit"])
        self.assertEqual([o["stopPrice"] for o in orders_sent[1:]], (["93", "118"] if long else ["105", "94"]))
        self.assertTrue(all(o["reduceOnly"] == "true" and D(o["size"]) == D(filled) and
                            o["triggerSignal"] == "last" for o in orders_sent[1:]))
