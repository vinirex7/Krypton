"""Small REST client for Binance USD-M. No automatic retry of mutations."""
import hashlib
import hmac
import json
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from strategy import BAR, D

SYMBOL = "BTCUSDT"
HOSTS = {"signal": "https://fapi.binance.com", "demo": "https://demo-fapi.binance.com",
         "live": "https://fapi.binance.com"}


class BinanceError(RuntimeError):
    def __init__(self, status, code, message):
        super().__init__(f"Binance HTTP {status} / {code}: {message}")
        self.code = code
        self.status = status


class Client:
    def __init__(self, mode, key="", secret=""):
        self.host, self.key, self.secret = HOSTS[mode], key, secret
        self.clock_offset = 0

    def request(self, method, route, signed=False, **params):
        headers = {}
        if signed:
            params.update(timestamp=int(time.time()*1000) + self.clock_offset, recvWindow=5000)
            headers["X-MBX-APIKEY"] = self.key
        query = urlencode(params)
        if signed:
            signature = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            query += "&signature=" + signature
        url = self.host + route
        if query:
            url += "?" + query
        req = Request(url, method=method, headers=headers)
        try:
            with urlopen(req, timeout=8) as response:
                return json.load(response)
        except HTTPError as error:
            body = error.read()
            try:
                info = json.loads(body)
            except ValueError:
                info = {}
            raise BinanceError(error.code, info.get("code"), info.get("msg", "Request failed")) from None

    def now(self):
        server = int(self.request("GET", "/fapi/v1/time")["serverTime"])
        self.clock_offset = server - int(time.time()*1000)
        return server

    def market(self, bar):
        raw = self.request("GET", "/fapi/v1/klines", symbol=SYMBOL, interval="4h",
                           endTime=bar-1, limit=1500)
        times = [int(x[0]) for x in raw]
        if len(times) < 1000 or times[-1] != bar-BAR or any(b-a != BAR for a,b in zip(times,times[1:])):
            raise ValueError("Insufficient or discontinuous BTC 4h candles")
        bars = pd.DataFrame([[float(x[i]) for i in (1,2,3,4)] for x in raw],
                            index=pd.to_datetime(times, unit="ms", utc=True),
                            columns=["open","high","low","close"])
        # Supplying endTime alone may select the earliest records. Request an explicit range.
        start = bar - 120*86400000
        rates = []
        while start < bar:
            page = self.request("GET", "/fapi/v1/fundingRate", symbol=SYMBOL, startTime=start,
                                endTime=bar, limit=1000)
            if not page:
                break
            last = int(page[-1]["fundingTime"])
            if last < start:
                raise ValueError("Funding pagination stalled")
            rates.extend(page)
            start = last + 1
            if len(page) < 1000:
                break
        idx = pd.to_datetime([r["fundingTime"] for r in rates], unit="ms", utc=True)
        if len(idx) < 273 or not idx.is_unique or not idx.is_monotonic_increasing:
            raise ValueError("Insufficient funding history")
        if bar - idx[-1].value//1000000 > 9*3600000:
            raise ValueError("Stale funding")
        return bars, pd.Series([float(r["fundingRate"]) for r in rates], index=idx)

    def filters(self):
        symbols = self.request("GET", "/fapi/v1/exchangeInfo")["symbols"]
        spec = next(s for s in symbols if s["symbol"] == SYMBOL)
        if (spec["status"], spec["contractType"], spec["marginAsset"]) != ("TRADING","PERPETUAL","USDT"):
            raise ValueError("Unexpected contract specification")
        return {f["filterType"]: f for f in spec["filters"]}

    def signed(self, method, endpoint, **params):
        return self.request(method, "/fapi/" + endpoint, True, **params)

    def symbol(self, method, endpoint, **params):
        return self.signed(method, endpoint, symbol=SYMBOL, **params)

    def position(self):
        rows = self.symbol("GET", "v3/positionRisk")
        if any(r["positionSide"] != "BOTH" for r in rows):
            raise ValueError("One-way position mode required")
        return (D(rows[0]["positionAmt"]), D(rows[0]["entryPrice"])) if rows else (D(0),D(0))

    def price(self):
        return D(self.request("GET", "/fapi/v2/ticker/price", symbol=SYMBOL)["price"])
