"""KSO1316 and KSL1 on Binance BTCUSDT perpetual, using 4-hour contract candles."""
from decimal import Decimal, ROUND_FLOOR

import numpy as np
import pandas as pd

BAR = 4 * 60 * 60 * 1000
D = Decimal


def indicators(bars, funding):
    """Index = UTC bar opening. Funding index = actual settlement timestamp."""
    close = bars.close
    prior = close.shift().fillna(close.iloc[0])
    tr = np.maximum(bars.high, prior) - np.minimum(bars.low, prior)
    atr = tr.iloc[13:].ewm(alpha=1/14, adjust=False).mean().reindex(close.index)
    ema21, ema50, ema200 = [close.ewm(span=n, adjust=False).mean() for n in (21, 50, 200)]
    mid = close.rolling(20).mean()
    band = 2 * close.rolling(20).std(ddof=0)
    narrow = (2 * band / mid).rolling(270).rank(pct=True) <= .20
    smooth = funding.rolling(3).mean()
    z = (smooth - smooth.rolling(270).mean()) / smooth.rolling(270).std(ddof=1)
    at_close = z.reindex(close.index + pd.Timedelta(hours=4), method="ffill").to_numpy()
    bear = (close < ema200) & (ema50 < ema200)
    crowd = (bear & (ema200 < ema200.shift(30)) & (close < ema21)
             & (close < close.shift(6)) & (close < bars.open) & (at_close > .5))
    squeeze = bear & narrow & (close < mid - band)
    candle = ((bars.open - close > atr) & (close - bars.low < .15 * (bars.high - bars.low))
              & (close > ema200))
    bull = (close > ema200) & (ema50 > ema200) & (ema200 > ema200.shift(30))
    # A sharp reversal above EMA200 is also eligible for the original candle short.
    reversal = (close < ema21) & (ema21 < ema21.shift())
    breakout = close > bars.high.shift().rolling(55).max()
    return pd.DataFrame({"bull": bull, "short_regime": bear | reversal,
                         "breakout": breakout, "atr": atr, "funding": crowd, "squeeze": squeeze, "candle": candle})


def entry(closed):
    # The most recent closed candle is the waiting candle.
    signal = closed.iloc[-2]
    for module in ("funding", "squeeze", "candle"):
        if signal[module]:
            return module, D(str(signal.atr))
    return None


def select_entry(closed, strategy):
    if strategy == "KSO1316":
        return entry(closed)
    if strategy not in ("KSL1", "KSL2"):
        raise ValueError("Unknown strategy")
    latest = closed.iloc[-1]
    if strategy == "KSL2" or latest.short_regime:
        short = entry(closed)
        if short:
            return short
    if latest.bull and latest.breakout:
        return "long_breakout", D(str(latest.atr))
    return None


def quantity(equity, peak, price, atr, risk=D(".03")):
    if equity <= 0 or price <= 0 or atr <= 0 or peak < equity:
        raise ValueError("Invalid equity, peak, entry price or ATR")
    dd = 1 - equity / peak
    fraction = D("1") if dd < D(".15") else D(".5") if dd < D(".25") else D(".25") if dd < D(".35") else D("0")
    return min(2 * equity / price, equity * risk * fraction / (D("2.5") * atr))


def exits(module, sale, atr, trough=None, prior_atr=None):
    if module == "long_breakout":
        return sale - D("2.5") * atr, sale + 6 * atr
    stop = sale + D("2.5") * atr
    target = sale - 3 * atr if module == "funding" else None
    if module == "squeeze" and trough is not None:
        stop = min(stop, min(sale, trough) + 3 * prior_atr)
    return stop, target


def due(module, opened, bar):
    lifetime = {"funding": 30, "candle": 12, "long_breakout": 60}.get(module)
    return lifetime is not None and bar >= opened + lifetime * BAR


def floor_step(value, step):
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step
