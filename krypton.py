"""Run with python krypton.py signal|demo|live (Linux or macOS)."""
import argparse
import fcntl
from http.client import IncompleteRead
import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from urllib.error import URLError

import pandas as pd

from binance_usdm import BinanceError, Client, SYMBOL
from strategy import BAR, D, quantity, due, entry, exits, floor_step, indicators

LOG = logging.getLogger("K1316")
PREFIX = "K1316-"


def num(x):
    return format(x, "f")


class State:
    def __init__(self, db):
        self.db = db
        db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, data TEXT NOT NULL)")
        row = db.execute("SELECT data FROM state WHERE id=1").fetchone()
        self.data = json.loads(row[0]) if row else {"peak": "0", "seen": 0, "trade": None}

    def save(self):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (1, ?)", (json.dumps(self.data),))


class Runner:
    def __init__(self, api, state):
        self.api = api
        self.state = state
        self.rules = api.filters()
        self.cached_bar = None

    def market(self, bar):
        if self.cached_bar != bar:
            bars, funding = self.api.market(bar)
            self.stats = indicators(bars, funding)
            self.bars = bars
            self.cached_bar = bar
        return self.bars, self.stats

    def algos(self):
        return self.api.symbol("GET", "v1/openAlgoOrders")

    def cancel(self, order):
        self.api.signed("DELETE", "v1/algoOrder", algoId=order["algoId"])

    def flatten(self, amount, reason):
        LOG.warning("Closing existing short: %s", reason)
        self.api.symbol("POST", "v1/order", side="BUY", type="MARKET", reduceOnly="true",
                        quantity=num(abs(amount)), newOrderRespType="RESULT")

    def protect(self, trade, amount, sale, bar, orders):
        atr = D(trade["atr"])
        opened = trade["bar"]
        if due(trade["module"], opened, bar):
            return self.flatten(amount, "time exit")
        stop, target = exits(trade["module"], sale, atr)
        # Install the initial exchange stop before fetching indicator history on restart.
        owned = [o for o in orders if o["clientAlgoId"].startswith(trade["id"]+"-")]
        stops = [o for o in owned if o["orderType"] == "STOP_MARKET"]
        if trade["module"] == "squeeze" and bar > opened and stops:
            bars, stats = self.market(bar)
            since = pd.to_datetime(opened, unit="ms", utc=True)
            if since < bars.index[0]:
                raise RuntimeError("Position predates available candles; inspect the resting stop")
            low = D(str(bars.low.loc[since:].min()))
            stop, _ = exits("squeeze", sale, atr, low, D(str(stats.atr.iloc[-1])))
        tick = D(self.rules["PRICE_FILTER"]["tickSize"])
        # Round short stop toward entry and take-profit down to the next valid tick.
        stop = floor_step(stop, tick)
        if target is not None:
            target = floor_step(target, tick)
        desired = [("STOP_MARKET", stop, "s")]
        if target is not None:
            desired.append(("TAKE_PROFIT_MARKET", target, "t"))
        for kind, level, suffix in desired:
            existing = [o for o in owned if o["orderType"] == kind]
            matched = [o for o in existing if D(o["triggerPrice"]) == level
                       and (trade["module"] != "squeeze" or D(o["quantity"]) == abs(amount))]
            if not matched:
                # Keep the old protection until Binance accepts its replacement.
                try:
                    close_params = {"closePosition": "true"}
                    if trade["module"] == "squeeze":
                        close_params = {"quantity": num(abs(amount)), "reduceOnly": "true"}
                    token = hashlib.sha256(f"{bar}:{kind}:{level}:{amount}".encode()).hexdigest()[:10]
                    self.api.symbol("POST", "v1/algoOrder", algoType="CONDITIONAL", side="BUY",
                                    type=kind, triggerPrice=num(level), workingType="CONTRACT_PRICE",
                                    clientAlgoId=f"{trade['id']}-{suffix}{token}", **close_params)
                except (BinanceError, URLError, TimeoutError) as exc:
                    if (isinstance(exc, BinanceError) and exc.code == -2021) or (kind == "STOP_MARKET" and not existing):
                        self.flatten(amount, "protective trigger crossed or initial stop unconfirmed")
                    raise
            for old in existing:
                if not matched or old["algoId"] != matched[0]["algoId"]:
                    self.cancel(old)

    def record_equity(self, amount, sale, bar):
        account = self.api.signed("GET", "v3/account")
        usdt = next(asset for asset in account["assets"] if asset["asset"] == "USDT")
        wallet = D(usdt["walletBalance"])
        # Contract opening price, matching the convention used by the simulator.
        last = self.api.request("GET", "/fapi/v1/klines", symbol=SYMBOL, interval="4h", limit=1)
        if int(last[-1][0]) != bar:
            raise ValueError("Candle boundary changed, retry")
        equity = wallet + abs(amount) * (sale-D(last[-1][1]))
        peak = max(D(self.state.data["peak"]), equity)
        if equity <= 0:
            raise RuntimeError("Positive USDT futures equity required")
        self.state.data["peak"] = num(peak)
        return equity, peak, D(usdt["availableBalance"])

    def tick(self):
        now = self.api.now()
        bar = now // BAR * BAR
        state = self.state.data
        trade = state["trade"]
        amount, sale = self.api.position()
        if amount > 0 or (amount and not trade):
            raise RuntimeError("Unexpected BTC position; reconcile the dedicated account")
        orders = self.algos()
        if any(not o["clientAlgoId"].startswith(PREFIX) for o in orders):
            raise RuntimeError("Unexpected BTC conditional order")
        if trade and not trade["confirmed"]:
            # Query the original order after crashes or ambiguous network responses.
            order = self.api.symbol("GET", "v1/order", origClientOrderId=trade["id"])
            if order["status"] in ("NEW", "PARTIALLY_FILLED"):
                if amount < 0:
                    self.protect(trade, amount, sale, bar, orders)
                return
            filled = D(order["executedQty"])
            if filled and not amount:
                # If a native protection closed the short after a partial fill, verify
                # via the order history before manually clearing a pending intent.
                raise RuntimeError("Entry filled but position absent during reconciliation")
            if filled:
                trade["qty"] = num(filled)
            trade["confirmed"] = bool(filled)
            self.state.save()
        if amount:
            if abs(amount) != D(trade["qty"]) and trade["confirmed"]:
                raise RuntimeError("Unexpected BTC quantity; reconcile the dedicated account")
            self.protect(trade, amount, sale, bar, orders)
            if state["seen"] != bar:
                self.record_equity(amount, sale, bar)
                state["seen"] = bar
                self.state.save()
            return
        if trade and trade["confirmed"] and now-trade["bar"] < 60_000 and not amount:
            # Binance snapshots may lag a successful fill. Keep the native protection in place.
            return
        if trade or orders:
            for order in orders:
                self.cancel(order)
            state["trade"] = None
            self.state.save()
        if state["seen"] == bar:
            return
        equity, peak, available = self.record_equity(amount, sale, bar)
        if now-bar > 30_000:
            state["seen"] = bar
            self.state.save()
            return
        _, stats = self.market(bar)
        choice = entry(stats)
        if not choice:
            state["seen"] = bar
            self.state.save()
            return
        module, atr = choice
        if self.api.symbol("GET", "v1/openOrders"):
            raise RuntimeError("Unexpected BTC open order")
        price = self.api.price()
        size = quantity(equity, peak, price, atr)
        # Cap by available margin and reserve execution fees.
        market = self.rules["MARKET_LOT_SIZE"]
        lot = self.rules["LOT_SIZE"]
        step = max(D(market["stepSize"]), D(lot["stepSize"]))
        size = floor_step(min(size, available*D("1.99")/price, D(market["maxQty"]), D(lot["maxQty"])), step)
        state["seen"] = bar
        if (size < max(D(market["minQty"]),D(lot["minQty"])) or
                size*price < D(self.rules["MIN_NOTIONAL"]["notional"]) or
                self.api.now() - bar > 30_000):
            self.state.save()
            return
        trade = {"id": PREFIX+str(bar), "module": module, "atr": num(atr), "bar": bar,
                 "qty": num(size), "confirmed": False}
        state["trade"] = trade
        self.state.save()
        try:
            order = self.api.symbol("POST", "v1/order", side="SELL", type="MARKET", quantity=num(size),
                                    newClientOrderId=trade["id"], newOrderRespType="RESULT")
        except BinanceError as exc:
            if exc.code in {-2019, -1111, -4004, -4164, -2027}:
                state["trade"] = None  # Explicit rejection. Keep this bar consumed.
                self.state.save()
            raise
        if order["status"] == "FILLED":
            trade["qty"] = order["executedQty"]
            trade["confirmed"] = True
            self.state.save()
            self.protect(trade, -D(order["executedQty"]), D(order["avgPrice"]), bar, [])
        LOG.info("Entry %s %s %s", trade["id"], module, order["status"])


def startup(api, state):
    api.now()
    if str(api.signed("GET", "v1/positionSide/dual")["dualSidePosition"]).lower() != "false":
        raise RuntimeError("Select One-way Mode for this account")
    if str(api.signed("GET", "v1/multiAssetsMargin")["multiAssetsMargin"]).lower() != "false":
        raise RuntimeError("Select Single-Asset Mode")
    config = api.symbol("GET", "v1/symbolConfig")[0]
    if int(config["leverage"]) != 2:
        raise RuntimeError("Set BTCUSDT leverage to 2x")
    return Runner(api, state)


NETWORK_ERRORS = (URLError, TimeoutError, ConnectionError, IncompleteRead)
CYCLE_ERRORS = NETWORK_ERRORS + (BinanceError, ValueError)


def reconnect(api, state):
    while True:
        try:
            return startup(api, state)
        except NETWORK_ERRORS as exc:
            LOG.warning("Waiting for Binance: %s", exc)
        except BinanceError as exc:
            if exc.status < 500 and exc.status not in (418, 429):
                raise
            LOG.warning("Binance unavailable: %s", exc)
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["signal","demo","live"], default="signal", nargs="?")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.mode == "signal":
        api = Client("signal")
        bar = api.now() // BAR * BAR
        bars, funding = api.market(bar)
        print("BTCUSDT", pd.to_datetime(bar, unit="ms", utc=True), entry(indicators(bars, funding)))
        return
    prefix = "KRYPTON_" + args.mode.upper()
    key, secret = os.environ[prefix+"_KEY"], os.environ[prefix+"_SECRET"]
    api = Client(args.mode, key, secret)
    root = Path(os.environ.get("KRYPTON_STATE_DIR", "var")).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    # One persistent state per dedicated account and environment, preserved on key rotation.
    with (root/(args.mode+".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with closing(sqlite3.connect(root/(args.mode+".sqlite"))) as db:
            runner = reconnect(api, State(db))
            while True:
                try:
                    runner.tick()
                except CYCLE_ERRORS as exc:
                    LOG.error("Cycle deferred: %s", exc)
                time.sleep(3)


if __name__ == "__main__":
    main()
