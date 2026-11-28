"""Replay the frozen KSO1316 benchmark using production signal and risk functions."""
import argparse
from hashlib import sha256
from pathlib import Path

import pandas as pd

from strategy import BAR, D, due, select_entry, exits, indicators, quantity


def replay(bars, funding, start, cost, strategy="KSO1316", end_date="2026-11-23", ledger=None):
    stats = indicators(bars, funding.set_index("time").fundingRate)
    carry = funding.set_index(funding.time.dt.round("h")).reindex(bars.index)
    carry = carry.fundingRate.fillna(0) * carry.markPrice.fillna(bars.open)
    first = bars.index.searchsorted(pd.Timestamp(start, tz="UTC"))
    end = bars.index.searchsorted(pd.Timestamp(end_date, tz="UTC"))
    fee, slip = D(".0005")*cost, D(".0002")*cost
    cash = peak = intrabar_peak = D(1)
    dd = D(0)
    module = None
    count = 0
    total_net = D(0)
    for i in range(first, end):
        bar_ms = bars.index[i].value // 1000000
        op, high, low, close = map(lambda v: D(str(v)), bars.iloc[i][["open","high","low","close"]])
        if module:
            cash -= side * qty * D(str(carry.iloc[i]))
        equity = cash + (side*qty*(op-sale) if module else 0)
        peak = max(peak, equity)
        if not module:
            decision = select_entry(stats.iloc[i-2:i], strategy)
            if decision:
                candidate, atr = decision
                side = 1 if candidate == "long_breakout" else -1
                sale = op*(1+side*slip)
                qty = quantity(cash, peak, sale, atr, D(".01") if side == 1 else D(".03"))
                if qty:
                    initial = cash
                    module, opened, trough = candidate, bar_ms, sale
                    cash -= qty*sale*fee
        adverse = favorable = closing = cash
        if module:
            stop, target = exits(module, sale, atr, trough if bar_ms>opened else None,
                                 D(str(stats.atr.iloc[i-1])))
            fill = None
            barrier = True
            worst, best = (low, high) if side == 1 else (high, low)
            if (op-stop)*side <= 0 or (target is not None and (op-target)*side >= 0):
                fill = op
            elif (worst-stop)*side <= 0:
                fill = stop
            elif target is not None and (best-target)*side >= 0:
                fill = target
            elif due(module, opened, bar_ms+BAR) or i == end-1:
                fill, barrier = close, False
            adverse_px = worst
            if fill is not None and barrier:
                adverse_px = min(op, fill) if side == 1 else max(op, fill)
            adverse = cash + qty*side*(adverse_px-sale)
            favorable = cash + qty*side*((fill if fill is not None else best)-sale)
            if fill is None:
                closing = cash + qty*side*(close-sale)
                trough = min(trough, low)
            else:
                exit_price = fill*(1-side*slip)
                cash += qty*side*(exit_price-sale) - qty*exit_price*fee
                net = cash-initial
                total_net += net
                if ledger is not None:
                    ledger.append((opened, bar_ms, module, float(net)))
                closing = cash
                count += 1
                module = None
        intrabar_peak = max(intrabar_peak, closing, favorable)
        dd = max(dd, 1-min(closing,adverse)/intrabar_peak)
    assert abs(cash - 1 - total_net) < D("1e-18")
    return count, float(cash-1), float(dd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    price_file = next(args.data.glob("BTCUSDT_PERP*.parquet"))
    funding_file = next(args.data.glob("BTCUSDT_PERP*funding.csv"))
    expected_hashes = ("9bb865ead938f46c827201adc4590cb3a837b4064f7bd617913dcdce8b847689",
                       "bf6bed8045743b49874eac4fd0738b9aead357ab7d1f74b4494f7653b7956995")
    if tuple(sha256(path.read_bytes()).hexdigest() for path in (price_file, funding_file)) != expected_hashes:
        raise ValueError("Benchmark requires the original BTC perpetual and funding snapshots")
    bars = pd.read_parquet(price_file).set_index("datetime")
    bars.index = pd.to_datetime(bars.index, utc=True)
    rates = pd.read_csv(funding_file)
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

    if args.compare:
        assert bars.index.is_unique and (bars.index.to_series().diff().dropna() == pd.Timedelta(hours=4)).all()
        assert rates.time.is_unique and rates.time.is_monotonic_increasing
        for start, end, cost in [
            ("2019-12-01","2024-01-01",1), ("2024-01-01","2025-01-01",1),
            ("2025-01-01","2026-11-23",1), ("2019-12-01","2026-11-23",1),
            ("2019-12-01","2026-11-23",2), ("2025-01-01","2026-11-23",2)]:
            for strategy in ("KSO1316","KSL1","KSL2"):
                ledger = []
                count, ret, dd = replay(bars, rates, start, cost, strategy, end, ledger)
                longs = [r for r in ledger if r[2] == "long_breakout"]
                assert count == len(ledger)
                assert all(a[1] < b[0] for a, b in zip(ledger, ledger[1:]))
                print(start, end, cost, strategy, count, len(longs), ret, dd, sum(r[3] for r in longs), flush=True)
