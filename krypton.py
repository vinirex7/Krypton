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

from binance_usdm import Client, SYMBOL
from demo_budget import configure_budget
from exchange_errors import ExchangeError
from kraken_futures import KrakenClient, CONTRACT
from strategy import BAR, D, quantity, due, select_entry, exits, floor_step, indicators

LOG = logging.getLogger("Krypton")
SHORT_PREFIXES = ("K1316-", "KSO1316-")


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
    def __init__(self, api, state, strategy="KSO1316"):
        self.strategy = strategy
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

    def flatten(self, amount, reason):
        LOG.warning("Closing existing position: %s", reason)
        self.api.close_position(num(abs(amount)), long=amount > 0)

    def protect(self, trade, amount, sale, bar, orders):
        atr = D(trade["atr"])
        opened = trade["bar"]
        if due(trade["module"], opened, bar):
            # A reduce-only exit can fill even when its response never arrives.
            # Persist the consumed opening before sending it, including across restarts.
            if self.state.data.get("timed_exit_bar") != bar:
                self.state.data["timed_exit_bar"] = bar
                self.state.save()
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
        stop = -floor_step(-stop, tick) if amount > 0 else floor_step(stop, tick)
        if target is not None:
            target = floor_step(target, tick)
        desired = [("STOP_MARKET", stop, "s")]
        if target is not None:
            desired.append(("TAKE_PROFIT_MARKET", target, "t"))
        for kind, level, suffix in desired:
            existing = [o for o in owned if o["orderType"] == kind]
            matched = [o for o in existing if D(o["triggerPrice"]) == level
                       and (o.get("closePosition") in ("true", True) or D(o["quantity"]) == abs(amount))]
            if not matched:
                # Keep the old protection until the exchange accepts its replacement.
                try:
                    token = hashlib.sha256(f"{bar}:{kind}:{level}:{amount}".encode()).hexdigest()[:10]
                    self.api.protect_position(kind, num(level), f"{trade['id']}-{suffix}{token}",
                                           num(abs(amount)), close_all=trade["module"] != "squeeze", long=amount > 0)
                except PROTECTION_ERRORS as exc:
                    if (isinstance(exc, ExchangeError) and exc.crossed) or (kind == "STOP_MARKET" and not existing):
                        self.flatten(amount, "protective trigger crossed or initial stop unconfirmed")
                    raise
            for old in existing:
                if not matched or old["algoId"] != matched[0]["algoId"]:
                    self.api.cancel(old)

    def record_equity(self, amount, sale, bar):
        wallet, available = self.api.balances()
        if "demo_100_brl" in self.state.data:
            # Use only the allocated cash plus this dedicated wallet's P&L.
            wallet -= D(self.state.data["demo_100_brl"]["reserve_usd"])
            available = min(wallet, available)
        equity = wallet + amount * (self.api.bar_open(bar)-sale)
        peak = max(D(self.state.data["peak"]), equity)
        if equity <= 0:
            raise RuntimeError("Positive futures equity required")
        self.state.data["peak"] = num(peak)
        return equity, peak, available

    def tick(self):
        now = self.api.now()
        bar = now // BAR * BAR
        state = self.state.data
        trade = state["trade"]
        amount, sale = self.api.position()
        prefixes = SHORT_PREFIXES if self.strategy == "KSO1316" else (self.strategy + "-",)
        if trade and not trade["id"].startswith(prefixes):
            raise RuntimeError("Resume active trade with its original strategy")
        if amount and (not trade or (amount > 0) != (trade["module"] == "long_breakout")):
            raise RuntimeError("Unexpected BTC position; reconcile the dedicated account")
        orders = self.api.protections()
        if any(not o["clientAlgoId"].startswith(prefixes) or
               (trade and not o["clientAlgoId"].startswith(trade["id"]+"-")) or
               (amount and o.get("side", "SELL" if amount > 0 else "BUY") != ("SELL" if amount > 0 else "BUY"))
               for o in orders):
            raise RuntimeError("Unexpected BTC conditional order")
        if trade and not trade["confirmed"]:
            # Query the original order after crashes or ambiguous network responses.
            if amount and not orders:
                if abs(amount) > D(trade["qty"]):
                    raise RuntimeError("Unexpected quantity while reconciling entry")
                # Protect the observed fill even if the order-history service is lagging.
                self.protect(trade, amount, sale, bar, orders)
                return
            order = self.api.entry_status(trade["id"])
            if order["status"] in ("NEW", "PARTIALLY_FILLED"):
                if amount:
                    self.protect(trade, amount, sale, bar, orders)
                return
            filled = D(order["executedQty"])
            if filled and not amount:
                # If a native protection closed the short after a partial fill, verify
                # via the order history before manually clearing a pending intent.
                if now-trade["bar"] < 60_000:
                    return
                raise RuntimeError("Entry filled but position absent during reconciliation")
            if filled:
                trade["qty"] = num(filled)
            trade["confirmed"] = bool(filled)
            self.state.save()
        if amount:
            if abs(amount) != D(trade["qty"]) and trade["confirmed"]:
                if 0 < abs(amount) < D(trade["qty"]) and isinstance(self.api, KrakenClient):
                    # A market exit is IOC on Kraken. Finish reducing a partially closed short.
                    self.flatten(amount, "remaining quantity after a reduction")
                    return
                raise RuntimeError("Unexpected BTC quantity; reconcile the dedicated account")
            self.protect(trade, amount, sale, bar, orders)
            if state["seen"] != bar:
                self.record_equity(amount, sale, bar)
                state["seen"] = bar
                self.state.save()
            return
        if trade and trade["confirmed"] and now-trade["bar"] < 60_000 and not amount:
            # Position snapshots may lag a successful fill. Keep the native protection in place.
            return
        if trade or orders:
            for order in orders:
                self.api.cancel(order)
            state["trade"] = None
            self.state.save()
        if state["seen"] == bar:
            return
        equity, peak, available = self.record_equity(amount, sale, bar)
        if state.get("timed_exit_bar") == bar or now-bar > 30_000:
            state["seen"] = bar
            self.state.save()
            return
        _, stats = self.market(bar)
        choice = select_entry(stats, self.strategy)
        if not choice:
            state["seen"] = bar
            self.state.save()
            return
        module, atr = choice
        if self.api.open_orders():
            raise RuntimeError("Unexpected BTC open order")
        price = self.api.price()
        long = module == "long_breakout"
        size = quantity(equity, peak, price, atr, D(".01") if long else D(".03"))
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
        trade = {"id": (self.strategy + ("-L-" if long else "-") + str(bar)), "module": module, "atr": num(atr), "bar": bar,
                 "qty": num(size), "confirmed": False}
        state["trade"] = trade
        self.state.save()
        try:
            order = self.api.open_position(num(size), trade["id"], long=long)
        except ExchangeError as exc:
            if exc.rejected:
                state["trade"] = None  # Explicit rejection. Keep this bar consumed.
                self.state.save()
            raise
        if order["status"] == "FILLED":
            trade["qty"] = order["executedQty"]
            trade["confirmed"] = True
            self.state.save()
            self.protect(trade, (1 if long else -1)*D(order["executedQty"]), D(order["avgPrice"]), bar, [])
        LOG.info("Entry %s %s %s", trade["id"], module, order["status"])


def startup(api, state, strategy="KSO1316", demo_100_brl=False):
    api.now()
    api.validate_account()
    if demo_100_brl:
        configure_budget(api, state)
    elif "demo_100_brl" in state.data:
        raise RuntimeError("Estado reservado ao teste KSL2 de R$ 100")
    return Runner(api, state, strategy)


NETWORK_ERRORS = (URLError, TimeoutError, ConnectionError, IncompleteRead)
PROTECTION_ERRORS = NETWORK_ERRORS + (ExchangeError,)
CYCLE_ERRORS = PROTECTION_ERRORS + (ValueError,)


def reconnect(api, state, strategy="KSO1316", demo_100_brl=False):
    while True:
        try:
            return startup(api, state, strategy, demo_100_brl)
        except NETWORK_ERRORS as exc:
            LOG.warning("Waiting for exchange: %s", exc)
        except ExchangeError as exc:
            if exc.status < 500 and exc.status not in (418, 429):
                raise
            LOG.warning("Exchange unavailable: %s", exc)
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["signal","demo","live"], default="signal", nargs="?")
    parser.add_argument("--exchange", choices=["binance", "kraken"], default="binance")
    parser.add_argument("--strategy", choices=["KSO1316", "KSL1", "KSL2"], default="KSO1316")
    args = parser.parse_args()
    if args.strategy != "KSO1316" and args.mode == "live":
        parser.error(f"{args.strategy} em validação: use signal ou demo")
    kraken = args.exchange == "kraken"
    client = KrakenClient if kraken else Client
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.mode == "signal":
        api = client("signal")
        if kraken:
            api.filters()
        bar = api.now() // BAR * BAR
        bars, funding = api.market(bar)
        print(CONTRACT if kraken else SYMBOL, pd.to_datetime(bar, unit="ms", utc=True), select_entry(indicators(bars, funding), args.strategy))
        return
    prefix = ("KRYPTON_KRAKEN_" if kraken else "KRYPTON_") + args.mode.upper()
    key, secret = os.environ.get(prefix+"_KEY"), os.environ.get(prefix+"_SECRET")
    if not key or not secret:
        parser.error(f"Configure {prefix}_KEY e {prefix}_SECRET no ambiente local")
    api = client(args.mode, key, secret)
    root = Path(os.environ.get("KRYPTON_STATE_DIR", "var")).resolve()
    if kraken:
        root = root / "kraken"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    # One persistent state per dedicated account and environment, preserved on key rotation.
    with (root/(args.mode+".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with closing(sqlite3.connect(root/(args.mode+".sqlite"))) as db:
            runner = reconnect(api, State(db), args.strategy,
                               kraken and args.mode == "demo" and args.strategy == "KSL2")
            while True:
                try:
                    runner.tick()
                except CYCLE_ERRORS as exc:
                    LOG.error("Cycle deferred: %s", exc)
                time.sleep(3)


if __name__ == "__main__":
    main()
