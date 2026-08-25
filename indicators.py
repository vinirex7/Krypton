# indicators.py — Indicadores técnicos do Krypton

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 7,
    multiplier: float = 3.0,
) -> tuple:
    """Supertrend sem propagar direction=0 após o warm-up do ATR."""
    atr = compute_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    st = pd.Series(np.nan, index=close.index, dtype=float)
    direction = pd.Series(np.nan, index=close.index, dtype=float)

    valid = np.flatnonzero(atr.notna().to_numpy())
    if len(valid) == 0:
        return st, direction

    first = int(valid[0])
    direction.iloc[first] = 1 if close.iloc[first] >= hl2.iloc[first] else -1
    st.iloc[first] = lower.iloc[first] if direction.iloc[first] == 1 else upper.iloc[first]

    for i in range(first + 1, len(close)):
        prev_st = st.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        ub = upper.iloc[i]
        lb = lower.iloc[i]

        if pd.notna(prev_st):
            if close.iloc[i - 1] <= prev_st:
                ub = min(ub, upper.iloc[i - 1])
            if close.iloc[i - 1] >= prev_st:
                lb = max(lb, lower.iloc[i - 1])

        if close.iloc[i] > ub:
            direction.iloc[i] = 1
            st.iloc[i] = lb
        elif close.iloc[i] < lb:
            direction.iloc[i] = -1
            st.iloc[i] = ub
        else:
            direction.iloc[i] = prev_dir
            st.iloc[i] = lb if prev_dir == 1 else ub

    return st, direction


def compute_signals(
    df: pd.DataFrame,
    st_period: int = 7,
    st_mult: float = 3.0,
    rsi_period: int = 14,
    rsi_low: float = 40,
    rsi_high: float = 70,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_sig: int = 9,
) -> pd.Series:
    """Retorna +1 LONG, -1 SHORT (apenas sinal), 0 FLAT.

    O live Spot trata -1 como EXIT e nunca abre short.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    rsi = compute_rsi(close, rsi_period)
    macd_line, sig_line, _ = compute_macd(close, macd_fast, macd_slow, macd_sig)
    _, st_dir = compute_supertrend(high, low, close, st_period, st_mult)

    signals = pd.Series(0, index=df.index, dtype=int)
    current = 0

    for i in range(len(df)):
        d = st_dir.iloc[i]
        r = rsi.iloc[i]
        ml = macd_line.iloc[i]
        sl = sig_line.iloc[i]

        if any(pd.isna(x) for x in [d, r, ml, sl]):
            signals.iloc[i] = current
            continue

        long_ok = d == 1 and rsi_low <= r <= rsi_high and ml > sl
        short_ok = d == -1 and (100 - rsi_high) <= r <= (100 - rsi_low) and ml < sl

        if long_ok:
            current = 1
        elif short_ok:
            current = -1
        elif (d == 1 and current == -1) or (d == -1 and current == 1):
            current = 0

        signals.iloc[i] = current

    return signals
