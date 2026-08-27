"""Research-only Binance Spot + USD-M perpetual funding carry.

Delta-neutral sleeve: long Spot and short the same base quantity in USD-M
perpetuals. Signals use only funding settlements known by the prior daily close;
portfolio changes execute at the next daily open. No live order capability.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

import adaptive_portfolio as ap
from config import ENTRY_SLIPPAGE_PCT, EXIT_SLIPPAGE_PCT, FEE_RATE

BASE_URL = "https://data.binance.vision/data/futures/um"
FUTURES_TAKER_FEE = 0.0005
FUTURES_SLIPPAGE = 0.0005
INITIAL_CAPITAL = ap.INITIAL_CAPITAL
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def _utc(x):
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _period_months(start, end):
    cur = pd.Timestamp(_utc(start).date()).replace(day=1)
    last = pd.Timestamp(_utc(end).date()).replace(day=1)
    while cur <= last:
        yield cur.year, cur.month
        cur = cur + pd.offsets.MonthBegin(1)


def _read_zip_csv(content: bytes, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        raw = pd.read_csv(zf.open(zf.namelist()[0]), header=None)
    if raw.empty:
        return pd.DataFrame(columns=columns)
    first = str(raw.iloc[0, 0]).strip().lower()
    if not first.replace(".", "", 1).isdigit():
        raw = raw.iloc[1:].reset_index(drop=True)
    raw = raw.iloc[:, :len(columns)].copy()
    raw.columns = columns[:raw.shape[1]]
    return raw


def _get_archive(session, url):
    r = session.get(url, timeout=45)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def _download_series(symbol, start, end, kind):
    if kind not in {"klines", "fundingRate"}:
        raise ValueError("kind invalido")
    s, e = _utc(start), _utc(end)
    session = requests.Session()
    frames = []
    columns = KLINE_COLUMNS if kind == "klines" else FUNDING_COLUMNS
    for year, month in _period_months(s, e):
        ym = f"{year:04d}-{month:02d}"
        if kind == "klines":
            monthly = f"{BASE_URL}/monthly/klines/{symbol}/1d/{symbol}-1d-{ym}.zip"
        else:
            monthly = f"{BASE_URL}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
        content = _get_archive(session, monthly)
        if content is not None:
            frames.append(_read_zip_csv(content, columns))
            continue
        # Historical monthly 404 normally means the contract was not listed yet.
        # Daily fallback is only needed for the still-incomplete end month.
        if (year, month) != (e.year, e.month):
            continue
        month_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        month_end = month_start + pd.offsets.MonthEnd(0)
        day_start = max(s.normalize(), month_start)
        day_end = min(e.normalize(), month_end)
        for day in pd.date_range(day_start, day_end, freq="D", tz="UTC"):
            ds = day.strftime("%Y-%m-%d")
            if kind == "klines":
                url = f"{BASE_URL}/daily/klines/{symbol}/1d/{symbol}-1d-{ds}.zip"
            else:
                url = f"{BASE_URL}/daily/fundingRate/{symbol}/{symbol}-fundingRate-{ds}.zip"
            content = _get_archive(session, url)
            if content is not None:
                frames.append(_read_zip_csv(content, columns))
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _epoch_to_utc(values):
    x = pd.to_numeric(values, errors="coerce")
    finite = x[np.isfinite(x)]
    unit = "us" if (not finite.empty and float(finite.abs().median()) > 1e14) else "ms"
    return pd.to_datetime(x, unit=unit, utc=True, errors="coerce")


def download_perp_and_funding(symbol, start, end):
    k = _download_series(symbol, start, end, "klines")
    if k.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    k["open_time"] = _epoch_to_utc(k["open_time"])
    for col in ["open", "high", "low", "close"]:
        k[col] = pd.to_numeric(k[col], errors="coerce")
    k = k.dropna(subset=["open_time", "open", "close"]).drop_duplicates("open_time")
    k = k.set_index("open_time").sort_index()
    k = k.loc[(_utc(start) <= k.index) & (k.index <= _utc(end))]

    f = _download_series(symbol, start, end, "fundingRate")
    if f.empty:
        return k, pd.Series(dtype=float)
    f["calc_time"] = _epoch_to_utc(f["calc_time"])
    f["last_funding_rate"] = pd.to_numeric(f["last_funding_rate"], errors="coerce")
    f = f.dropna(subset=["calc_time", "last_funding_rate"]).drop_duplicates("calc_time")
    f = f.loc[(_utc(start) <= f["calc_time"]) & (f["calc_time"] < _utc(end) + timedelta(days=1))]
    daily = f.set_index("calc_time")["last_funding_rate"].resample("1D").sum().fillna(0.0)
    return k, daily


def trailing_funding_apr(funding_daily: pd.Series, ts, lookback_days=7):
    ts = _utc(ts).normalize()
    hist = funding_daily.loc[:ts].tail(lookback_days)
    if len(hist) < lookback_days:
        return np.nan
    return float(hist.mean() * 365.0)


@dataclass
class Position:
    spot_qty: float
    fut_qty: float
    prev_fut_mark: float


def _spot_mark(positions, spot_data, ts, field="close"):
    return sum(
        p.spot_qty * float(spot_data[s]["df"].loc[ts, field])
        for s, p in positions.items()
    )


def simulate_funding_carry(spot_data, futures_data, symbols, start, end, *,
                           entry_apr=0.12, exit_apr=0.00, lookback_days=7,
                           top_n=2, rebalance_days=7, notional_fraction=0.50):
    """Simulate long-spot/short-perp funding carry with segregated margin stress.

    `notional_fraction` is total spot notional as a fraction of sleeve equity;
    the same base quantity is shorted in perpetuals. At 0.50, half of equity
    remains as futures/cash collateral, i.e. roughly 1x leverage on the short leg.
    """
    if not (0 < notional_fraction <= 0.60):
        raise ValueError("notional_fraction deve estar em (0, 0.60]")
    s, e = _utc(start), _utc(end)
    base_calendar = spot_data["BTCUSDT"]["df"].loc[s:e].index
    calendar = [ts for ts in base_calendar if any(
        ts in futures_data.get(sym, {}).get("perp", pd.DataFrame()).index and
        ts in spot_data[sym]["df"].index for sym in symbols
    )]
    if not calendar:
        empty = pd.Series(dtype=float)
        return {**ap.performance_metrics(empty), "equity_curve": empty,
                "trade_log": pd.DataFrame(), "funding_pnl": 0.0,
                "liquidation_events": 0}

    cash = INITIAL_CAPITAL
    positions: dict[str, Position] = {}
    pending = None
    pending_signal_time = None
    last_signal_i = None
    points, logs = [], []
    total_funding = 0.0
    liquidation_events = 0

    def equity_at(ts, field="close"):
        return cash + _spot_mark(positions, spot_data, ts, field)

    def close_all(ts):
        nonlocal cash
        for sym, p in list(positions.items()):
            spot_open = float(spot_data[sym]["df"].loc[ts, "open"])
            fut_open = float(futures_data[sym]["perp"].loc[ts, "open"])
            cash += p.fut_qty * (p.prev_fut_mark - fut_open)
            cash -= p.fut_qty * fut_open * (FUTURES_SLIPPAGE + FUTURES_TAKER_FEE)
            spot_px = spot_open * (1.0 - EXIT_SLIPPAGE_PCT)
            cash += p.spot_qty * spot_px * (1.0 - FEE_RATE)
            logs.append({"event": "close", "symbol": sym, "execution_time": ts})
            del positions[sym]

    def open_selected(ts, selected):
        nonlocal cash
        if not selected:
            return
        equity = cash
        per_notional = equity * notional_fraction / len(selected)
        for sym in selected:
            spot_open = float(spot_data[sym]["df"].loc[ts, "open"])
            fut_open = float(futures_data[sym]["perp"].loc[ts, "open"])
            spot_px = spot_open * (1.0 + ENTRY_SLIPPAGE_PCT)
            qty = per_notional / spot_px
            spot_cost = qty * spot_px * (1.0 + FEE_RATE)
            fut_fee_slip = qty * fut_open * (FUTURES_TAKER_FEE + FUTURES_SLIPPAGE)
            if spot_cost + fut_fee_slip > cash:
                continue
            cash -= spot_cost + fut_fee_slip
            positions[sym] = Position(qty, qty, fut_open)
            logs.append({"event": "open", "symbol": sym, "execution_time": ts,
                         "notional": per_notional})

    for i, ts in enumerate(calendar):
        if pending is not None:
            close_all(ts)
            open_selected(ts, pending)
            for row in logs[-(len(pending) + 2 * top_n):]:
                if row.get("execution_time") == ts:
                    row["signal_time"] = pending_signal_time
            pending = None

        for sym, p in list(positions.items()):
            fut = futures_data[sym]["perp"]
            if ts not in fut.index:
                continue
            fut_close = float(fut.loc[ts, "close"])
            cash += p.fut_qty * (p.prev_fut_mark - fut_close)
            p.prev_fut_mark = fut_close
            rate = float(futures_data[sym]["funding"].get(ts.normalize(), 0.0))
            funding_cash = p.fut_qty * fut_close * rate
            cash += funding_cash
            total_funding += funding_cash

        points.append((ts, equity_at(ts, "close")))

        if positions:
            worst_cash = cash
            maintenance = 0.0
            for sym, p in positions.items():
                fut = futures_data[sym]["perp"]
                if ts not in fut.index:
                    continue
                high = float(fut.loc[ts, "high"])
                close = float(fut.loc[ts, "close"])
                worst_cash += p.fut_qty * (close - high)
                maintenance += p.fut_qty * high * 0.01
            if worst_cash <= maintenance:
                liquidation_events += 1
                raise AssertionError(
                    f"liquidation proxy breached at {ts}: worst_cash={worst_cash}, "
                    f"maintenance={maintenance}"
                )

        if last_signal_i is None or i - last_signal_i >= rebalance_days:
            scored = []
            held = set(positions)
            for sym in symbols:
                perp = futures_data.get(sym, {}).get("perp", pd.DataFrame())
                funding = futures_data.get(sym, {}).get("funding", pd.Series(dtype=float))
                if ts not in perp.index or ts not in spot_data[sym]["df"].index:
                    continue
                apr = trailing_funding_apr(funding, ts, lookback_days)
                threshold = exit_apr if sym in held else entry_apr
                if np.isfinite(apr) and apr >= threshold:
                    scored.append((apr, sym))
            scored.sort(reverse=True)
            selected = [sym for _, sym in scored[:top_n]]
            # A review cadence is not a forced turnover cadence. If the desired
            # hedge set is unchanged, keep both legs open and avoid needless fees.
            pending = None if set(selected) == held else selected
            pending_signal_time = ts if pending is not None else None
            last_signal_i = i

    ts = calendar[-1]
    for sym, p in list(positions.items()):
        spot_close = float(spot_data[sym]["df"].loc[ts, "close"])
        fut_close = float(futures_data[sym]["perp"].loc[ts, "close"])
        cash -= p.fut_qty * fut_close * (FUTURES_SLIPPAGE + FUTURES_TAKER_FEE)
        spot_px = spot_close * (1.0 - EXIT_SLIPPAGE_PCT)
        cash += p.spot_qty * spot_px * (1.0 - FEE_RATE)
        del positions[sym]
    points[-1] = (ts, cash)

    equity = pd.Series([v for _, v in points], index=[t for t, _ in points], dtype=float)
    trade_log = pd.DataFrame(logs)
    if not trade_log.empty and "signal_time" in trade_log:
        opens = trade_log[trade_log["event"] == "open"].dropna(subset=["signal_time"])
        if not opens.empty and not bool((pd.to_datetime(opens["execution_time"], utc=True) >
                                         pd.to_datetime(opens["signal_time"], utc=True)).all()):
            raise AssertionError("look-ahead detectado no carry")
    return {**ap.performance_metrics(equity), "equity_curve": equity,
            "trade_log": trade_log, "funding_pnl": total_funding,
            "liquidation_events": liquidation_events}
