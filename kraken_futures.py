"""Kraken linear BTC perpetual execution with KSO1316's original public signal feed."""
import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from binance_usdm import Client as BinanceMarket
from exchange_errors import ExchangeError
from strategy import BAR, D

CONTRACT = "PF_XBTUSD"
API = "/derivatives/api/v3/"
HOSTS = {"live": "https://futures.kraken.com", "signal": "https://futures.kraken.com",
         "demo": "https://demo-futures.kraken.com"}
KINDS = {"STOP_MARKET": "stp", "TAKE_PROFIT_MARKET": "take_profit"}


def signature(secret, query, path):
    path = path.removeprefix("/derivatives")
    digest = hashlib.sha256((query + path).encode()).digest()
    return base64.b64encode(hmac.digest(base64.b64decode(secret, validate=True), digest, "sha512")).decode()


def error(code, status=400, **kwargs):
    if code in ("apiLimitExceeded", "Unavailable", "Server Error"):
        status = 503
    return ExchangeError(f"Kraken HTTP {status} / {code}", status, code, **kwargs)


class KrakenClient:
    def __init__(self, mode, key="", secret=""):
        self.host, self.key, self.secret = HOSTS[mode], key, secret
        self.reference = BinanceMarket("signal")
        self.clock_sample = None

    def request(self, method, path, *, private=False, **params):
        query = urlencode(params)
        headers = {"Accept": "application/json"}
        if private:
            headers.update(APIKey=self.key, Authent=signature(self.secret, query, path))
        url = self.host + path
        body = None
        if method == "POST":
            body = query.encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif query:
            url += "?" + query
        try:
            with urlopen(Request(url, method=method, data=body, headers=headers), timeout=8) as res:
                data = json.load(res, parse_float=D)
        except HTTPError as exc:
            raise error(exc.reason, exc.code) from None
        if "error" in data or data.get("result", "success") != "success":
            raise error(data.get("error", "unexpected response"))
        return data

    def ticker(self):
        return self.request("GET", API + "tickers/" + CONTRACT)

    def now(self):
        local = time.monotonic()
        if self.clock_sample is None or local - self.clock_sample[0] >= 60:
            stamp = self.ticker()["serverTime"]
            server_ms = int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()*1000)
            self.clock_sample = (time.monotonic(), server_ms)
        sampled, server_ms = self.clock_sample
        return server_ms + int((time.monotonic()-sampled)*1000)

    def price(self):
        tick = self.ticker()["ticker"]
        if tick["symbol"] != CONTRACT or tick["suspended"] or tick["postOnly"] or tick["tag"] != "perpetual":
            raise ValueError("BTC perpetual unavailable for market orders")
        return D(tick["last"])

    def market(self, bar):
        return self.reference.market(bar)

    def filters(self):
        specs = self.request("GET", API + "instruments")["instruments"]
        spec = next((row for row in specs if row["symbol"] == CONTRACT), None)
        if (not spec or spec["base"] not in ("XBT", "BTC") or spec["quote"] != "USD"
                or spec["type"] != "flexible_futures" or D(spec["contractSize"]) != 1
                or not spec["tradeable"] or spec.get("isExpired")):
            raise RuntimeError("Unexpected Kraken BTC perpetual specification")
        step = D(1).scaleb(-int(spec["contractValueTradePrecision"]))
        lot = {"stepSize": step, "minQty": step, "maxQty": D(spec["maxPositionSize"])}
        return {"PRICE_FILTER": {"tickSize": D(spec["tickSize"])}, "LOT_SIZE": lot,
                "MARKET_LOT_SIZE": lot, "MIN_NOTIONAL": {"notional": 0}}

    def validate_account(self):
        prefs = self.request("GET", API + "leveragepreferences", private=True)["leveragePreferences"]
        if not any(p["symbol"].upper() == CONTRACT and D(str(p.get("maxLeverage"))) == 2
                   for p in prefs if p.get("maxLeverage") is not None):
            raise RuntimeError("Set the Kraken BTC perpetual fixed leverage to 2x")
        self.balances()

    def balances(self):
        wallets = self.request("GET", API + "accounts", private=True)["accounts"]
        flex = wallets["flex"]
        if flex["type"] != "multiCollateralMarginAccount":
            raise RuntimeError("Kraken derivatives wallet required")
        return D(flex["balanceValue"]) + D(flex["unrealizedFunding"]), D(flex["availableMargin"])

    def bar_open(self, bar):
        path = "/api/charts/v1/trade/" + CONTRACT + "/4h"
        candles = self.request("GET", path, **{"from": bar//1000, "to": (bar+BAR-1)//1000})["candles"]
        for candle in candles:
            if int(candle["time"]) == bar:
                return D(candle["open"])
        raise ValueError("Kraken 4h opening price not yet available")

    def position(self):
        rows = self.request("GET", API + "openpositions", private=True)["openPositions"]
        active = [p for p in rows if D(p["size"])]
        if not active:
            return D(0), D(0)
        if len(active) != 1 or active[0]["symbol"].upper() != CONTRACT:
            raise RuntimeError("Unexpected position in dedicated Kraken wallet")
        row = active[0]
        if row["side"] not in ("short", "long") or D(row["size"]) < 0:
            raise RuntimeError("Unexpected Kraken position side or size")
        return D(row["size"]) * (-1 if row["side"] == "short" else 1), D(row["price"])

    def open_orders(self):
        return self.request("GET", API + "openorders", private=True)["openOrders"]

    def protections(self):
        result = []
        for row in self.open_orders():
            kind = {"stop": "STOP_MARKET", "stp": "STOP_MARKET",
                    "take_profit": "TAKE_PROFIT_MARKET"}.get(row["orderType"])
            if (row["symbol"].upper() != CONTRACT or not kind or row["side"] not in ("buy", "sell")
                    or row["reduceOnly"] is not True or row.get("triggerSignal") != "last"):
                raise RuntimeError("Unexpected order in dedicated Kraken wallet")
            result.append({"clientAlgoId": row.get("cliOrdId") or "", "algoId": row["order_id"],
                           "orderType": kind, "side": row["side"].upper(), "triggerPrice": row["stopPrice"], "quantity": row["unfilledSize"]})
        return result

    def cancel(self, order):
        res = self.request("POST", API + "cancelorder", private=True, order_id=order["algoId"])
        status = res["cancelStatus"]["status"]
        if status not in ("cancelled", "notFound"):
            raise error(status)

    def send(self, kind, side, size, **params):
        res = self.request("POST", API + "sendorder", private=True, orderType=kind,
                           side=side, size=size, symbol=CONTRACT, **params)
        order = res["sendStatus"]
        status = order["status"]
        if status not in ("filled", "partiallyFilled", "placed", "cancelled"):
            rejected = status in {"invalidSize", "insufficientAvailableFunds", "wouldCauseLiquidation",
                                  "maxPositionViolation", "iocWouldNotExecute", "wouldProcessAfterSpecifiedTime"}
            raise error(status, rejected=rejected)
        return order

    def close_short(self, size):
        return self.close_position(size, long=False)

    def close_position(self, size, *, long):
        self.send("mkt", "sell" if long else "buy", size, reduceOnly="true")

    def protect_short(self, kind, level, client_id, size, close_all):
        return self.protect_position(kind, level, client_id, size, close_all, long=False)

    def protect_position(self, kind, level, client_id, size, close_all, *, long):
        order = self.send(KINDS[kind], "sell" if long else "buy", size, stopPrice=level, cliOrdId=client_id,
                          triggerSignal="last", reduceOnly="true")
        if order["status"] != "placed":
            raise error("protective order not resting: " + order["status"], crossed=True)

    def sell(self, size, client_id):
        return self.open_position(size, client_id, long=False)

    def open_position(self, size, client_id, *, long):
        bar = int(client_id.rsplit("-", 1)[1])
        deadline = datetime.fromtimestamp((bar+30_000)/1000, timezone.utc).isoformat()
        order = self.send("mkt", "buy" if long else "sell", size, cliOrdId=client_id, processBefore=deadline)
        # Market orders are IOC. An execution event records this fill's price and amount.
        fills = [e for e in order.get("orderEvents", []) if e["type"] == "EXECUTION"
                 and e["orderPriorExecution"]["orderId"] == order["order_id"]]
        total = sum((D(e["amount"]) for e in fills), D(0))
        if total and (total == D(size) or order["status"] in ("partiallyFilled", "cancelled")):
            avg = sum(D(e["amount"])*D(e["price"]) for e in fills)/total
            return {"status": "FILLED", "executedQty": format(total, "f"), "avgPrice": avg}
        return {"status": "NEW"}

    def entry_status(self, client_id):
        res = self.request("POST", API + "orders/status", private=True, cliOrdIds=client_id)
        for item in res["orders"]:
            order = item["order"]
            if order["cliOrdId"] == client_id and order["symbol"].upper() == CONTRACT:
                status = {"FULLY_EXECUTED": "FILLED", "CANCELLED": "CANCELED",
                          "REJECTED": "REJECTED", "ENTERED_BOOK": "NEW"}.get(item["status"])
                if status is None or order["side"].lower() != ("buy" if "-L-" in client_id else "sell"):
                    raise RuntimeError("Unexpected Kraken entry status")
                return {"status": status, "executedQty": order["filled"]}
        # orders/status retains closed orders for only five seconds.
        params = {"since": int(client_id.rsplit("-", 1)[1]), "tradeable": CONTRACT,
                  "closed": "true", "sort": "desc"}
        tokens = set()
        while True:
            page = self.request("GET", "/api/history/v3/orders", private=True, **params)
            for row in page["elements"]:
                for event, details in row["event"].items():
                    order = details.get("newOrder") or details.get("order") or {}
                    if (order.get("clientId") == client_id and order["tradeable"].upper() == CONTRACT
                            and order["direction"] == ("Buy" if "-L-" in client_id else "Sell")):
                        filled = D(order["filled"])
                        if filled == D(order["quantity"]) or event in ("OrderCancelled", "OrderRejected"):
                            return {"status": "CLOSED", "executedQty": filled}
            token = page.get("continuationToken")
            if not token or token in tokens:
                break
            tokens.add(token)
            params["continuation_token"] = token
        raise ValueError(f"Awaiting confirmation of Kraken entry {client_id}; inspect order history")
