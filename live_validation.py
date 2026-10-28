"""Historical replay of the actual Aggressive C engine, without a Binance account.

Only the exchange, clock and persistence are replaced. daily_cycle(), sizing,
signals, alpha targets, sleeve transfers and reconciliation remain live code.
Daily OHLC cannot reproduce intraday peaks, order queues or 00:05 prices; see
ASSUMPTIONS in every run's manifest. Never use this adapter for live trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

import aggressive_c_live as live
import config
from binance_client import BinanceInterface
from live_c import LiveAggressiveCTradeBot
from risk_manager import RiskManager

MARKET_URL = "https://data-api.binance.vision/api/v3"
SYMBOLS = list(dict.fromkeys([*config.TRADING_PAIRS, *config.AGGRESSIVE_C_ALPHA_SYMBOLS]))
INITIAL_CAPITAL = 10_000.0
ASSUMPTIONS = [
    "Uses AggressiveCTradeBot.daily_cycle and inherited execution/reconciliation methods unchanged.",
    "Daily decisions use preceding completed candles; exchange limit=400 includes the current candle, leaving 399 completed bars.",
    "00:05 UTC mid-price is approximated by daily open; accepted LIMIT orders are assumed fully filled immediately.",
    "Fee is FEE_RATE charged in quote USDT; actual BNB discounts/base-asset commissions are unknown.",
    "LIMIT offset +/-0.05% and tactical signal-price deviation gate are the live implementation; MARKET/stop slippage uses config.",
    "OCO gap fills precede the daily decision; intraday double touches resolve stop first. TP fills at limit price.",
    "Risk is sampled at open, fills and close; 30-second peaks/halts and intraday ordering cannot be recovered from daily bars.",
    "Current exchange LOT_SIZE/PRICE_FILTER/NOTIONAL filters are used, not unavailable historical filter snapshots.",
    "Starts in cash, no manual decisions/deposits/withdrawals, no final forced liquidation; final equity is mark-to-market.",
    "Replays current live reconciliation even for OCO fills; its marked-value external-flow adjustment is not silently corrected.",
    "Runtime exceptions are caught as in live_c.run; reconciliation exceptions set portfolio_halted and are logged, never silently fixed.",
    "No artificial 0.3% transfer charge: current Spot-only live only re-earmarks cash and charges executed trades.",
    "Walk-forward uses fixed live parameters, 399 completed daily candles of causal history, contiguous 90-day test slices, state carried forward.",
    "No fitting or TP selection. Historical windows before strategy promotion are retrospective, not untouched out-of-sample evidence.",
]


def utc(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def request_json(path, params):
    for attempt in range(3):
        try:
            response = requests.get(f"{MARKET_URL}/{path}", params=params, timeout=30)
            response.raise_for_status()
            value = response.json()
            if isinstance(value, dict) and "code" in value:
                raise RuntimeError(str(value))
            return value
        except (requests.RequestException, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Official Binance data unavailable: {path}") from exc


def validate_frame(df, start, end, symbol):
    expected = pd.date_range(utc(start), utc(end), freq="D")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: duplicate or unordered dates")
    if not df.index.equals(expected):
        raise ValueError(f"{symbol}: missing/extra daily candles ({len(df)} vs {len(expected)})")
    cols = ["open", "high", "low", "close", "volume"]
    if not np.isfinite(df[cols].to_numpy()).all():
        raise ValueError(f"{symbol}: nonfinite data")
    if (df[cols[:4]] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError(f"{symbol}: invalid price/volume")
    if ((df.high < df[["open", "close", "low"]].max(axis=1)) |
            (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError(f"{symbol}: invalid OHLC")


def load_market(start, end, cache_dir, *, allow_short_warmup=False):
    """Strict exact-market downloader: no proxy and no stale/partial cache fallback."""
    start, end = utc(start), utc(end)
    if end >= pd.Timestamp.now(tz="UTC").normalize():
        raise ValueError("End must be a completed UTC day")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames, metadata = {}, {}
    for symbol in SYMBOLS:
        effective_start = start
        if allow_short_warmup:
            first = request_json("klines", {"symbol": symbol, "interval": "1d", "startTime": 0, "limit": 1})
            if not first:
                raise RuntimeError(f"Cannot verify listing history for {symbol}")
            effective_start = max(start, pd.to_datetime(first[0][0], unit="ms", utc=True))
        path = cache_dir / f"{symbol}_{start.date()}_{end.date()}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            rows, cursor = [], effective_start
            while cursor <= end:
                page = request_json("klines", {"symbol": symbol, "interval": "1d", "limit": 1000,
                                    "startTime": int(cursor.timestamp() * 1000),
                                    "endTime": int((end + pd.Timedelta(days=1)).timestamp() * 1000) - 1})
                if not page:
                    raise RuntimeError(f"Incomplete Binance history: {symbol} at {cursor}")
                rows.extend(page)
                next_cursor = pd.to_datetime(page[-1][0], unit="ms", utc=True) + pd.Timedelta(days=1)
                if next_cursor <= cursor:
                    raise RuntimeError("Nonadvancing Binance pagination")
                cursor = next_cursor
            df = pd.DataFrame(rows)
            df.index = pd.to_datetime(df[0], unit="ms", utc=True)
            df = df[[1, 2, 3, 4, 5]].astype(float)
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index.name = "open_time"
            validate_frame(df, effective_start, end, symbol)
            df.to_csv(path, float_format="%.12g")
            # Use exactly the serialized snapshot, including in the first run.
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
        validate_frame(df, effective_start, end, symbol)
        df.attrs["symbol"] = symbol
        frames[symbol] = df
        metadata[symbol] = {"rows": len(df), "start": str(df.index[0]), "end": str(df.index[-1]),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"Data verified: {symbol}: {len(df)} candles", flush=True)
    filter_path = cache_dir / "exchange_filters.json"
    if not filter_path.exists():
        filters = {}
        for symbol in SYMBOLS:
            raw = request_json("exchangeInfo", {"symbol": symbol})["symbols"][0]
            fs = {f["filterType"]: f for f in raw["filters"]}
            filters[symbol] = {"step_size": float(fs["LOT_SIZE"]["stepSize"]),
                               "tick_size": float(fs["PRICE_FILTER"]["tickSize"]),
                               "min_notional": float((fs.get("NOTIONAL") or fs["MIN_NOTIONAL"])["minNotional"])}
        filter_path.write_text(json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(),
                                           "symbols": filters}, indent=2))
    filters_snapshot = json.loads(filter_path.read_text())
    return frames, filters_snapshot["symbols"], {"market": MARKET_URL, "data": metadata,
                                                "filters_snapshot": filters_snapshot}


class HistoricalExchange:
    """In-memory Spot exchange. No client, API keys, network orders or state DB."""
    _step_precision = staticmethod(BinanceInterface._step_precision)

    def __init__(self, frames, filters, capital, *, rounding=True):
        self.rounding = rounding
        self.frames, self.filters, self.cash = frames, filters, capital
        self.qty = {s: 0.0 for s in SYMBOLS}
        self.prices = {}
        self.ocos = {}
        self.orders = []
        self.ts = None
        self.sleeve = ""
        self.next_id = 1
        self.rejections = []

    def _round_step(self, quantity, step):
        return BinanceInterface._round_step(self, quantity, step) if self.rounding else quantity

    def _round_tick(self, price, tick):
        return BinanceInterface._round_tick(self, price, tick) if self.rounding else price

    def get_current_price(self, symbol):
        return self.prices[symbol]

    def get_ohlcv(self, symbol, interval="1d", limit=400, closed_only=True):
        assert interval == "1d" and closed_only
        out = self.frames[symbol].loc[self.frames[symbol].index < self.ts].tail(limit - 1)
        out.attrs["symbol"] = symbol
        return out

    def get_asset_balance(self, asset):
        if asset == "USDT":
            return {"free": self.cash, "locked": 0.0, "total": self.cash}
        symbol = asset + "USDT"
        total = self.qty.get(symbol, 0.0)
        locked = sum(o["quantity"] for o in self.ocos.values() if o["symbol"] == symbol)
        return {"free": max(0.0, total - locked), "locked": locked, "total": total}

    def get_account_balance(self, asset):
        return self.get_asset_balance(asset)["free"]

    def get_asset_total(self, asset):
        return self.get_asset_balance(asset)["total"]

    def has_active_oco(self, symbol, oid):
        return oid in self.ocos and self.ocos[oid]["symbol"] == symbol

    def cancel_oco_order(self, symbol, oid):
        self.ocos.pop(oid, None)
        return True

    def create_oco_order(self, symbol, quantity, take_profit_price, stop_price, symbol_info):
        qty = self._round_step(quantity, symbol_info["step_size"])
        tp = self._round_tick(take_profit_price, symbol_info["tick_size"])
        sl = self._round_tick(stop_price, symbol_info["tick_size"])
        mid = self.prices[symbol]
        if qty <= 0 or not sl < mid < tp or qty > self.get_asset_balance(symbol[:-4])["free"] + 1e-10:
            return None
        oid = self.next_id
        self.next_id += 1
        self.ocos[oid] = {"symbol": symbol, "quantity": qty, "tp": tp, "sl": sl}
        return {"orderListId": oid}

    def _fill(self, symbol, side, quantity, price, kind, *, min_notional=True):
        info = self.filters[symbol]
        qty = self._round_step(quantity, info["step_size"])
        notional = qty * price
        fee = notional * config.FEE_RATE
        invalid = qty <= 0 or (min_notional and notional < info["min_notional"])
        invalid |= side == "BUY" and notional + fee > self.cash + 1e-8
        invalid |= side == "SELL" and qty > self.get_asset_balance(symbol[:-4])["free"] + 1e-10
        if invalid:
            self.rejections.append({"time": str(self.ts), "symbol": symbol, "side": side, "kind": kind})
            return None
        self.cash += -(notional + fee) if side == "BUY" else notional - fee
        self.qty[symbol] += qty if side == "BUY" else -qty
        oid = self.next_id
        self.next_id += 1
        self.orders.append({"time": self.ts, "symbol": symbol, "sleeve": self.sleeve,
                            "side": side, "quantity": qty, "price": price, "fee": fee, "kind": kind})
        return {"status": "FILLED", "orderId": oid, "executedQty": str(qty), "cummulativeQuoteQty": str(notional)}

    def place_limit_order(self, symbol, side, quantity, price, symbol_info, reference_price=None):
        if reference_price and abs(price - reference_price) / reference_price > config.SLIPPAGE_LIMIT_PCT:
            self.rejections.append({"time": str(self.ts), "symbol": symbol, "side": side, "kind": "SIGNAL_PRICE_GATE"})
            return None
        price = self._round_tick(price, symbol_info["tick_size"])
        return self._fill(symbol, side, quantity, price, "LIMIT")

    def place_market_order(self, symbol, side, quantity, symbol_info):
        slip = config.ENTRY_SLIPPAGE_PCT if side == "BUY" else -config.EXIT_SLIPPAGE_PCT
        return self._fill(symbol, side, quantity, self.prices[symbol] * (1 + slip), "MARKET", min_notional=False)

    def process_ocos(self, row_by_symbol, *, gaps_only=False):
        for oid, oco in list(self.ocos.items()):
            row = row_by_symbol[oco["symbol"]]
            op = float(row.open)
            if op <= oco["sl"]:
                px, kind = op * (1 - config.EXIT_SLIPPAGE_PCT), "SL_GAP"
            elif op >= oco["tp"]:
                px, kind = oco["tp"], "TP_GAP"
            elif not gaps_only and float(row.low) <= oco["sl"]:
                px, kind = oco["sl"] * (1 - config.EXIT_SLIPPAGE_PCT), "SL"
            elif not gaps_only and float(row.high) >= oco["tp"]:
                px, kind = oco["tp"], "TP"
            else:
                continue
            self.ocos.pop(oid)
            self.sleeve = "tactical"
            self._fill(oco["symbol"], "SELL", oco["quantity"], px, kind, min_notional=False)


class ReplayBot(LiveAggressiveCTradeBot):
    def __init__(self, exchange, start, capital):
        self.binance = exchange
        self.symbols = SYMBOLS
        self.symbol_infos = exchange.filters
        self.protection_blocked = False
        self.state = self._initial_state(capital)
        self.state["initialized_at"] = str(utc(start))
        self.state["last_sleeve_rebalance"] = utc(start).date().isoformat()
        self.tactical_risk = RiskManager(capital * config.AGGRESSIVE_C_TACTICAL_WEIGHT,
                                        risk_per_trade=config.AGGRESSIVE_C_TACTICAL_RISK_PER_TRADE,
                                        max_drawdown_pct=config.AGGRESSIVE_C_MAX_DRAWDOWN_PCT,
                                        circuit_breaker_pct=config.CIRCUIT_BREAKER_PCT)

    def _save_state(self):
        self.state["tactical_risk_state"] = self.tactical_risk.snapshot()

    def _execute_order(self, symbol, side, quantity, sleeve, **kwargs):
        self.binance.sleeve = sleeve
        # Telemetry is observational; reuse the execution engine without DB writes.
        return live.AggressiveCTradeBot._execute_order(self, symbol, side, quantity, sleeve, **kwargs)

    def daily_cycle(self):
        return live.AggressiveCTradeBot.daily_cycle(self)


class ReplayClock:
    current = None

    @classmethod
    def now(cls, tz=None):
        value = cls.current.to_pydatetime()
        return value if tz is None else value.astimezone(tz)


def metrics(curve):
    rets = curve.pct_change().dropna()
    duration = (curve.index[-1] - curve.index[0]).total_seconds() / 86400
    ret = float(curve.iloc[-1] / curve.iloc[0] - 1)
    dd = float((curve / curve.cummax() - 1).min())
    sd = float(rets.std())
    return {"return": ret, "cagr": float((1 + ret) ** (365.25 / duration) - 1) if duration else 0.0,
            "max_drawdown": dd, "sharpe": float(rets.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0,
            "final_capital": float(curve.iloc[-1])}


def replay(frames, filters, start, end, *, capital=INITIAL_CAPITAL, signal_cache=None, rounding=True, allow_short_warmup=False):
    start, end = utc(start), utc(end)
    expected = pd.date_range(start, end, freq="D")
    for symbol in SYMBOLS:
        validate_frame(frames[symbol].loc[start:end], start, end, symbol)
        if not allow_short_warmup and len(frames[symbol].loc[frames[symbol].index < start]) < live.HISTORY_LIMIT - 1:
            raise ValueError(f"{symbol}: less than 399 completed warm-up candles")
    exchange = HistoricalExchange(frames, filters, capital, rounding=rounding)
    ReplayClock.current = start + pd.Timedelta(minutes=5)
    signal_cache = {} if signal_cache is None else signal_cache
    original_signals = live.compute_signals

    def cached_signals(frame, **kwargs):
        key = (frame.attrs["symbol"], frame.index[0], frame.index[-1], tuple(sorted(kwargs.items())))
        if key not in signal_cache:
            signal_cache[key] = original_signals(frame, **kwargs)
        return signal_cache[key]

    points = [(start, capital)]
    runtime_errors = []
    halt_date = None
    with patch.object(live, "datetime", ReplayClock), patch.object(live, "compute_signals", cached_signals):
        bot = ReplayBot(exchange, start, capital)
        for i, ts in enumerate(expected):
            exchange.ts = ts
            ReplayClock.current = ts + pd.Timedelta(minutes=5)
            rows = {s: frames[s].loc[ts] for s in SYMBOLS}
            exchange.prices = {s: float(rows[s].open) for s in SYMBOLS}
            exchange.process_ocos(rows, gaps_only=True)
            # This is the production method, not a copied research strategy.
            try:
                bot.daily_cycle()
            except Exception as exc:
                runtime_errors.append({"time": str(ts), "phase": "daily_cycle", "error": str(exc)})
            exchange.process_ocos(rows)
            exchange.prices = {s: float(rows[s].close) for s in SYMBOLS}
            ReplayClock.current = ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
            try:
                bot._reconcile_exchange_state()
                eq, _ = bot._portfolio_risk_update()
            except Exception as exc:
                # Same containment as the running process, which remains alive
                # while blocking new exposure. Mark actual exchange equity.
                bot.state["portfolio_halted"] = True
                runtime_errors.append({"time": str(ts), "phase": "runtime_reconciliation", "error": str(exc)})
                eq = bot._portfolio_equity()
            if bot.state["portfolio_halted"] and halt_date is None:
                halt_date = str(ts.date())
            if exchange.cash < -1e-7 or any(q < -1e-8 for q in exchange.qty.values()):
                raise AssertionError("Spot cash/quantity became negative")
            virtual = bot._tactical_equity() + bot._alpha_equity()
            dust_tolerance = sum(filters[s]["min_notional"] +
                                 1.5 * filters[s]["step_size"] * exchange.prices[s] for s in SYMBOLS)
            if not runtime_errors and not np.isclose(virtual, eq, atol=dust_tolerance, rtol=0):
                raise AssertionError(f"Sleeve equity mismatch: {virtual} vs {eq}")
            points.append((ts + pd.Timedelta(days=1), eq))
            if i % 90 == 0:
                print(f"Replay {start.date()}: {ts.date()} equity={eq:.2f}", flush=True)
    curve = pd.Series([v for _, v in points], index=[t for t, _ in points], name="equity")
    return {**metrics(curve), "equity_curve": curve, "orders": pd.DataFrame(exchange.orders, columns=["time", "symbol", "sleeve", "side", "quantity", "price", "fee", "kind"]),
            "rejections": pd.DataFrame(exchange.rejections, columns=["time", "symbol", "side", "kind"]), "state": bot.state,
            "runtime_errors": pd.DataFrame(runtime_errors, columns=["time", "phase", "error"]), "runtime_error_count": len(runtime_errors),
            "first_runtime_error": runtime_errors[0] if runtime_errors else None,
            "halt_date": halt_date, "orders_count": len(exchange.orders),
            "fees": sum(o["fee"] for o in exchange.orders)}


def walk_forward(curve, start, end, test_days=90):
    """Stateful fixed-policy temporal validation, including a final partial fold.

    Pre-test history supplies only causal indicators, never a selected parameter.
    Stitching contiguous test slices must equal the original continuous replay.
    """
    if test_days <= 0:
        raise ValueError("test_days must be positive")
    start, end_exclusive = utc(start), utc(end) + pd.Timedelta(days=1)
    rows = []
    cursor = start
    while cursor < end_exclusive:
        stop = min(cursor + pd.Timedelta(days=test_days), end_exclusive)
        fold_curve = curve.loc[cursor:stop]
        rows.append({"fold": len(rows) + 1, "history_start": str((cursor - pd.Timedelta(days=live.HISTORY_LIMIT - 1)).date()),
                     "history_end": str((cursor - pd.Timedelta(days=1)).date()),
                     "test_start": str(cursor.date()), "test_end": str((stop - pd.Timedelta(days=1)).date()),
                     "test_days": (stop - cursor).days, "parameters": "fixed live; no selection",
                     **metrics(fold_curve)})
        cursor = stop
    out = pd.DataFrame(rows)
    compounded = float(np.prod(1 + out["return"]) - 1)
    if not np.isclose(compounded, metrics(curve)["return"], atol=1e-12):
        raise AssertionError("Walk-forward stitching lost/overlapped dates")
    return out, compounded


def period_start(as_of, label):
    if label == "10m":
        return as_of - pd.DateOffset(months=10)
    if label in {"2y", "3y", "4y"}:
        return as_of - pd.DateOffset(years=int(label[0]))
    raise ValueError(f"Unknown period: {label}")


def main(mode="both"):
    parser = argparse.ArgumentParser(description="Replay current Aggressive C live configuration")
    parser.add_argument("--as-of", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"), help="Exclusive UTC end")
    parser.add_argument("--periods", nargs="+", default=["10m", "2y", "3y", "4y"], choices=["10m", "2y", "3y", "4y"])
    parser.add_argument("--start", help="Single custom start; overrides --periods")
    parser.add_argument("--end", help="Single custom inclusive end; must be a completed day")
    parser.add_argument("--output", default="results/live-parity")
    parser.add_argument("--cache-dir", default=".market-cache")
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--no-rounding", action="store_true",
                        help="Research counterfactual: bypass exchange quantity-step and price-tick quantization only")
    parser.add_argument("--allow-short-warmup", action="store_true",
                        help="Allow pre-test history to start at exchange-verified listing date; no gaps permitted within test period")
    args = parser.parse_args()
    as_of = utc(args.as_of)
    end = utc(args.end) if args.end else as_of - pd.Timedelta(days=1)
    periods = {"custom": utc(args.start)} if args.start else {p: period_start(as_of, p) for p in args.periods}
    if min(periods.values()) > end:
        parser.error("start must be <= end")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logging.getLogger().setLevel(logging.ERROR)
    frames, filters, provenance = load_market(min(periods.values()) - pd.Timedelta(days=400), end, args.cache_dir, allow_short_warmup=args.allow_short_warmup)
    manifest = {"as_of": str(as_of.date()), "end_inclusive": str(end.date()),
                "profile": live.frozen_profile(), "strategy_fingerprint": live.strategy_fingerprint(),
                "assumptions": list(ASSUMPTIONS), "provenance": provenance,
                "exchange_rounding": not args.no_rounding, "allow_short_warmup": args.allow_short_warmup}
    if args.allow_short_warmup:
        manifest["assumptions"].append(
            "Pre-test history begins no earlier than the exchange-verified listing date. "
            "No synthetic candles: live indicators/alpha use only available history. Test-period coverage remains strict.")
        manifest["initial_completed_candles"] = {symbol: min(live.HISTORY_LIMIT - 1, int((df.index < min(periods.values())).sum()))
                                                  for symbol, df in frames.items()}
    if args.no_rounding:
        manifest["assumptions"].append(
            "COUNTERFACTUAL: step-size and tick-size quantization disabled for all simulated fills and OCO levels. "
            "Minimum notional, balance checks, fees, signals, risk, dust reconciliation thresholds and halts unchanged. "
            "Float64 arithmetic remains; this is not a numerical arbitrary-precision experiment or executable Binance order model.")
    manifest["execution_config"] = {name: getattr(config, name) for name in (
        "SUPERTREND_PERIOD", "SUPERTREND_MULTIPLIER", "RSI_PERIOD", "RSI_LOW", "RSI_HIGH",
        "MACD_FAST", "MACD_SLOW", "MACD_SIGNAL", "FEE_RATE", "ENTRY_SLIPPAGE_PCT",
        "EXIT_SLIPPAGE_PCT", "SLIPPAGE_LIMIT_PCT", "CIRCUIT_BREAKER_PCT", "MAX_SIMULTANEOUS_POS",
        "STOP_LOSS_ATR_MULT", "TAKE_PROFIT_ATR_MULT", "REGIME_SMA_PERIOD", "TIMEFRAME")}
    manifest["source_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ("aggressive_c_live.py", "live_c.py", "config.py", "risk_manager.py",
                                              "binance_client.py", "indicators.py", "live_validation.py")}
    manifest["versions"] = {"numpy": np.__version__, "pandas": pd.__version__}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    cache, summaries = {}, []
    for label, start in periods.items():
        result = replay(frames, filters, start, end, signal_cache=cache, rounding=not args.no_rounding, allow_short_warmup=args.allow_short_warmup)
        summary = {"period": label, "start": str(start.date()), "end": str(end.date()),
                   **{k: v for k, v in result.items() if k not in {"equity_curve", "orders", "rejections", "runtime_errors", "state"}}}
        result["equity_curve"].to_csv(output / f"{label}_equity.csv")
        result["orders"].to_csv(output / f"{label}_orders.csv", index=False)
        result["rejections"].to_csv(output / f"{label}_rejections.csv", index=False)
        result["runtime_errors"].to_csv(output / f"{label}_runtime_errors.csv", index=False)
        (output / f"{label}_state.json").write_text(json.dumps(result["state"], indent=2, default=str))
        if mode != "backtest":
            folds, compounded = walk_forward(result["equity_curve"], start, end, args.test_days)
            folds.to_csv(output / f"{label}_walk_forward.csv", index=False)
            summary.update({"wf_folds": len(folds), "wf_positive_folds": int((folds["return"] > 0).sum()),
                            "wf_compounded_return": compounded})
        summaries.append(summary)
        # Persist progress after every completed period.
        pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
        (output / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
