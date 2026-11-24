"""Replay the frozen K1316 benchmark using production signal and risk functions."""
import argparse
from pathlib import Path

import pandas as pd

from strategy import BAR, D, due, entry, exits, indicators, quantity


def replay(bars, funding, start, cost):
    stats = indicators(bars, funding.set_index("time").fundingRate)
    carry = funding.set_index(funding.time.dt.round("h")).reindex(bars.index)
    carry = carry.fundingRate.fillna(0) * carry.markPrice.fillna(bars.open)
    first = bars.index.searchsorted(pd.Timestamp(start, tz="UTC"))
    end = bars.index.searchsorted(pd.Timestamp("2026-11-23", tz="UTC"))
    fee, slip = D(".0005")*cost, D(".0002")*cost
    cash = peak = intrabar_peak = D(1)
    dd = D(0)
    module = None
    count = 0
    for i in range(first, end):
        bar_ms = bars.index[i].value // 1000000
        op, high, low, close = map(lambda v: D(str(v)), bars.iloc[i][["open","high","low","close"]])
        if module:
            cash += qty * D(str(carry.iloc[i]))
        equity = cash + (qty*(sale-op) if module else 0)
        peak = max(peak, equity)
        if not module:
            decision = entry(stats.iloc[i-2:i])
            if decision:
                candidate, atr = decision
                sale = op*(1-slip)
                qty = quantity(cash, peak, sale, atr)
                if qty:
                    module, opened, trough = candidate, bar_ms, sale
                    cash -= qty*sale*fee
        adverse = favorable = closing = cash
        if module:
            stop, target = exits(module, sale, atr, trough if bar_ms>opened else None,
                                 D(str(stats.atr.iloc[i-1])))
            fill = None
            barrier = True
            if op >= stop or (target is not None and op <= target):
                fill = op
            elif high >= stop:
                fill = stop
            elif target is not None and low <= target:
                fill = target
            elif due(module, opened, bar_ms+BAR) or i == end-1:
                fill, barrier = close, False
            adverse = cash + qty*(sale - (max(op,fill) if fill is not None and barrier else high))
            favorable = cash + qty*(sale - (fill if fill is not None else low))
            if fill is None:
                closing = cash + qty*(sale-close)
                trough = min(trough, low)
            else:
                buy = fill*(1+slip)
                cash += qty*(sale-buy) - qty*buy*fee
                closing = cash
                count += 1
                module = None
        intrabar_peak = max(intrabar_peak, closing, favorable)
        dd = max(dd, 1-min(closing,adverse)/intrabar_peak)
    return count, float(cash-1), float(dd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    bars = pd.read_parquet(next(args.data.glob("BTCUSDT_PERP*.parquet"))).set_index("datetime")
    bars.index = pd.to_datetime(bars.index, utc=True)
    rates = pd.read_csv(next(args.data.glob("BTCUSDT_PERP*funding.csv")))
    rates["time"] = pd.to_datetime(rates.fundingTime, unit="ms", utc=True)
    for start, cost, expected in [
        ("2019-12-01", 1, (238, 1.4900101808146404, .17297598395117042)),
        ("2024-01-01", 1, (115, .5201354010126806, .17254651151664613)),
        ("2024-01-01", 2, (115, .2336185185741817, .19722832916139055))
    ]:
        actual = replay(bars, rates, start, cost)
        if any(abs(a-b)>1e-10 for a,b in zip(actual,expected)):
            raise ValueError((actual,expected))
        print(start, cost, actual)
