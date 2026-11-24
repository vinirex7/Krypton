import sqlite3
import unittest
from unittest.mock import Mock

import pandas as pd

from binance_usdm import BinanceError
from krypton import Runner, State
from strategy import BAR, D, entry, exits, due, quantity


class Rules(unittest.TestCase):
    def test_delayed_priority(self):
        data = pd.DataFrame({"atr":[100,200], "funding":[True,False],
                             "squeeze":[True,False], "candle":[True,True]})
        self.assertEqual(entry(data), ("funding", D(100)))
        data.loc[0, ["funding","squeeze","candle"]] = False
        self.assertIsNone(entry(data))

    def test_risk_brakes(self):
        for balance, unit in [(100,1),(85,.5),(75,.25),(65,0)]:
            size = quantity(D(balance),D(100),D(10),D(1))
            self.assertEqual(size, D(balance)*D(".03")*D(str(unit))/D("2.5"))
        self.assertEqual(quantity(D(100),D(100),D(10),D(".001")),20)

    def test_exits(self):
        self.assertEqual(exits("funding",D(100),D(2)),(105,94))
        self.assertEqual(exits("squeeze",D(100),D(2),D(90),D(2)),(96,None))
        self.assertFalse(due("candle",0,11*BAR))
        self.assertTrue(due("candle",0,12*BAR))
        self.assertTrue(due("funding",0,30*BAR))
        self.assertFalse(due("squeeze",0,1000*BAR))


class Execution(unittest.TestCase):
    def setUp(self):
        self.api = Mock()
        self.api.filters.return_value = {"PRICE_FILTER":{"tickSize":"0.1"},
                "LOT_SIZE":{"minQty":".001","maxQty":"1000","stepSize":".001"},
                "MARKET_LOT_SIZE":{"minQty":".001","maxQty":"1000","stepSize":".001"},
                "MIN_NOTIONAL":{"notional":"5"}}
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.state = State(self.db)
        self.bot = Runner(self.api,self.state)
        self.bar = 10000*BAR
        self.api.now.return_value = self.bar + 1000
        self.api.position.return_value = (D(0),D(0))
        self.api.price.return_value = D(100)
        self.bot.record_equity = Mock(return_value=(D(100),D(100),D(100)))
        self.bot.market = Mock(return_value=(None,pd.DataFrame({"atr":[2,3],"funding":[True,False],
                                       "squeeze":[False,False],"candle":[False,False]})))
        self.calls = []
        self.fail_entry = False
        self.fail_stop = False

        def call(method, path, **params):
            self.calls.append((method,path,params))
            if method == "GET" and path in ("v1/openAlgoOrders","v1/openOrders"):
                return []
            if method == "POST" and params.get("side") == "SELL":
                if self.fail_entry:
                    raise TimeoutError()
                return {"status":"FILLED","avgPrice":"100","executedQty":params["quantity"]}
            if method == "GET" and path == "v1/order":
                raise BinanceError(400,-2013,"Unknown order")
            if path == "v1/algoOrder" and self.fail_stop:
                raise BinanceError(400,-2021,"Would trigger")
            return {}
        self.api.symbol.side_effect = call

    def test_native_bracket_and_no_duplicate(self):
        self.bot.tick()
        self.assertTrue(self.state.data["trade"]["confirmed"])
        bracket = [p for _,route,p in self.calls if route=="v1/algoOrder"]
        self.assertEqual([D(p["triggerPrice"]) for p in bracket],[105,94])
        self.assertTrue(all(p["closePosition"]=="true" and p["side"]=="BUY" for p in bracket))
        self.bot.tick()
        self.assertEqual(sum(p.get("side")=="SELL" for _,_,p in self.calls),1)

    def test_unknown_submission_survives_restart(self):
        self.fail_entry = True
        with self.assertRaises(TimeoutError):
            self.bot.tick()
        restarted = Runner(self.api,State(self.db))
        with self.assertRaises(BinanceError):
            restarted.tick()
        self.assertEqual(sum(p.get("side")=="SELL" for _,_,p in self.calls),1)
        self.assertFalse(restarted.state.data["trade"]["confirmed"])

    def test_failure_to_install_stop_submits_reduce_only_exit(self):
        self.fail_stop = True
        with self.assertRaises(BinanceError):
            self.bot.tick()
        self.assertEqual(self.calls[-1][2]["reduceOnly"],"true")
        self.assertEqual(self.calls[-1][2]["side"],"BUY")

    def test_trailing_stop_replaced_before_cancel(self):
        trade = {"id":"K1316-1","bar":self.bar-BAR,"module":"squeeze","atr":"2"}
        when = pd.to_datetime([self.bar-BAR], unit="ms", utc=True)
        self.bot.market.return_value = (pd.DataFrame({"low":[89.]},index=when),
                                        pd.DataFrame({"atr":[2.]},index=when))
        old = {"algoId":123,"clientAlgoId":"K1316-1-old","orderType":"STOP_MARKET",
               "triggerPrice":"105.0","quantity":"1.2"}
        self.bot.protect(trade,D("-1.2"),D(100),self.bar,[old])
        new = self.calls[0][2]
        self.assertEqual(D(new["triggerPrice"]),95)
        self.assertEqual(new["reduceOnly"],"true")
        self.api.signed.assert_called_once_with("DELETE","v1/algoOrder",algoId=123)

    def test_deadline_uses_reduce_only_market(self):
        trade = {"id":"K1316-2","bar":self.bar-12*BAR,"module":"candle","atr":"2"}
        self.bot.protect(trade,D("-0.2"),D(100),self.bar,[])
        self.assertEqual(self.calls[0][2]["quantity"],"0.2")
        self.assertEqual(self.calls[0][2]["reduceOnly"],"true")

    def test_skips_stale_entry_and_foreign_position(self):
        self.api.now.return_value = self.bar+60_000
        self.bot.tick()
        self.assertEqual(self.state.data["seen"],self.bar)
        self.bot.market.assert_not_called()
        self.api.position.return_value = (D("-0.01"),D(100))
        with self.assertRaises(RuntimeError):
            self.bot.tick()


if __name__ == "__main__":
    unittest.main()
