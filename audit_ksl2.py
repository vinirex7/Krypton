"""Conferência independente das operações e aproximação das regras temporais do executor."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategy import indicators, select_entry


def load(data):
    bars = pd.read_parquet(next(data.glob("BTCUSDT_PERP*.parquet"))).set_index("datetime")
    bars.index = pd.to_datetime(bars.index, utc=True)
    funds = pd.read_csv(data / "BTCUSDT_PERP_funding.csv")
    funds.index = pd.to_datetime(funds.fundingTime, unit="ms", utc=True)
    assert bars.index.is_unique and (np.diff(bars.open_time) == 14400000).all()
    assert np.isfinite(bars[["open","high","low","close"]]).all().all()
    assert (bars.low > 0).all() and (bars.low <= bars[["close","open"]].min(axis=1)).all()
    assert (bars.high >= bars[["close","open"]].max(axis=1)).all()
    assert funds.index.is_unique and funds.index.is_monotonic_increasing and funds.fundingRate.notna().all()
    return bars, funds


def decisions(bars, rates, runtime):
    # Each row describes the choice at its opening. No open bar is passed to the selector.
    sig = indicators(bars, rates)
    choices = []
    for i, time in enumerate(bars.index):
        if i < 2:
            choices.append((None, np.nan))
            continue
        if runtime:
            if i < 1000:
                choices.append((None, np.nan))
                continue
            history = rates.loc[time-pd.Timedelta(days=120):time]
            if len(history) < 273 or time-history.index[-1] > pd.Timedelta(hours=9):
                choices.append((None, np.nan))
                continue
            frame = indicators(bars.iloc[max(0,i-1500):i], history)
        else:
            frame = sig.iloc[i-2:i]
        choice = select_entry(frame, "KSL2")
        choices.append((choice[0], float(choice[1])) if choice else (None, np.nan))
    return choices, sig.atr.to_numpy()


def verify_decisions(bars, rates, choices, atr):
    """Build the signals at their decision timestamps independently of the selector."""
    close = bars.close
    fast, medium, slow = (close.ewm(span=n, adjust=False).mean() for n in (21,50,200))
    tr = pd.concat([bars.high-bars.low, (bars.high-close.shift()).abs(),
                    (bars.low-close.shift()).abs()], axis=1).max(axis=1)
    expected_atr = tr.iloc[13:].ewm(alpha=1/14, adjust=False).mean().reindex(close.index)
    np.testing.assert_allclose(atr, expected_atr, equal_nan=True)
    center = close.rolling(20).mean()
    deviation = close.rolling(20).std(ddof=0)
    width = 4*deviation/center
    squeezed = width.rolling(270).apply(
        lambda x: (np.count_nonzero(x<x[-1])+(np.count_nonzero(x==x[-1])+1)/2)/len(x),
        raw=True) <= .2
    smooth = rates.rolling(3).mean()
    z = (smooth-smooth.rolling(270).mean())/smooth.rolling(270).std()
    # Most recent known settlement at the close of the signal candle.
    known = z.asof(close.index + pd.Timedelta(hours=4)).to_numpy()
    bear = (close<slow) & (medium<slow)
    funding = bear & (slow<slow.shift(30)) & (close<fast) & (close<close.shift(6)) & (close<bars.open) & (known>.5)
    squeeze = bear & squeezed & (close<center-2*deviation)
    candle = (bars.open-close>expected_atr) & ((close-bars.low)<.15*(bars.high-bars.low)) & (close>slow)
    long = (medium>slow) & (slow>slow.shift(30)) & (close>slow) & (close>bars.high.rolling(55).max().shift(1))
    expected = np.select([x.shift(n,fill_value=False) for x,n in
                          ((funding,2),(squeeze,2),(candle,2),(long,1))],
                         ["funding","squeeze","candle","long_breakout"], default="")
    actual = np.array([choice or "" for choice,_ in choices])
    assert np.array_equal(expected,actual)
    ix = np.flatnonzero(actual!="")
    signature_at = ix-np.where(actual[ix]=="long_breakout",1,2)
    np.testing.assert_allclose([choices[i][1] for i in ix],expected_atr.iloc[signature_at])


@dataclass
class Trade:
    opened: int
    module: str
    direction: int
    size: float
    entry: float
    atr: float
    trough: float
    starting_cash: float


def simulate(bars, funds, choices, atr, first, end, cost, runtime, short_only):
    """Signed units and a single cash account. Funding is applied to carried positions."""
    settlements = funds.set_axis(funds.index.round("4h")).reindex(bars.index)
    carry = (settlements.fundingRate.fillna(0)*settlements.markPrice.fillna(bars.open)).to_numpy()
    candles = bars[["open","high","low","close"]].to_numpy()
    start, stop = bars.index.searchsorted([pd.Timestamp(first,tz="UTC"),pd.Timestamp(end,tz="UTC")])
    cash = open_peak = close_peak = old_peak = 1.
    close_dd = old_dd = 0.
    ledger = []
    pending = None
    fee, slip = .0005*cost, .0002*cost
    for i in range(start, stop):
        op, hi, lo, close = candles[i]
        if pending:
            cash -= pending.size*carry[i]
        equity = cash + (pending.size*(op-pending.entry) if pending else 0)
        open_peak = max(equity, open_peak)
        if pending is None:
            module, risk_atr = choices[i]
            long = module == "long_breakout"
            if module and not (short_only and long):
                side = 1 if long else -1
                entry = op*(1+side*slip)
                drawdown = 1-equity/open_peak
                brake = 1 if drawdown < .15 else .5 if drawdown < .25 else .25 if drawdown < .35 else 0
                ref = op if runtime else entry
                size = min((1.99 if runtime else 2)*cash/ref,
                           cash*(.01 if long else .03)*brake/(2.5*risk_atr))
                if size:
                    pending = Trade(i,module,side,side*size,entry,risk_atr,entry,cash)
                    cash -= size*entry*fee
        favorable = adverse = marked = cash
        if pending:
            p = pending
            long = p.direction == 1
            duration = {"funding":30,"candle":12,"long_breakout":60}.get(p.module)
            barrier = p.entry - p.direction*2.5*p.atr
            if p.module == "squeeze" and i > p.opened:
                barrier = min(barrier, p.trough+3*atr[i-1])
            target = (p.entry+6*p.atr if long else p.entry-3*p.atr if p.module=="funding" else None)
            best, worst = (hi,lo) if long else (lo,hi)
            fill = None
            timed_open = runtime and duration is not None and i-p.opened >= duration
            if timed_open:
                fill, reason = op, "prazo_abertura"
            elif (op-barrier)*p.direction <= 0 or (target is not None and (op-target)*p.direction >= 0):
                fill, reason = op, "gap"
            elif (worst-barrier)*p.direction <= 0:
                fill, reason = barrier, "stop"
            elif target is not None and (best-target)*p.direction >= 0:
                fill, reason = target, "alvo"
            elif not runtime and duration is not None and i+1-p.opened >= duration:
                fill, reason = close, "prazo_fechamento"
            elif i+1 == stop:
                fill, reason = close, "fim"
            worst_seen = worst
            if fill is not None and reason not in ("prazo_fechamento","fim"):
                worst_seen = min(op,fill) if long else max(op,fill)
            adverse = cash+p.size*(worst_seen-p.entry)
            favorable = cash+p.size*((best if fill is None else fill)-p.entry)
            if fill is None:
                marked = cash+p.size*(close-p.entry)
                p.trough = min(p.trough,lo)
            else:
                px = fill*(1-p.direction*slip)
                cash += p.size*(px-p.entry)-abs(p.size)*px*fee
                ledger.append((p.opened,i,p.module,reason,cash-p.starting_cash))
                pending = None
                marked = cash
        close_peak = max(close_peak, marked)
        close_dd = max(close_dd,1-marked/close_peak)
        old_peak = max(old_peak,favorable,marked)
        old_dd = max(old_dd,1-min(marked,adverse)/old_peak)
    trades = pd.DataFrame(ledger,columns=["entry","exit","module","reason","net"])
    assert np.isclose(trades.net.sum(),cash-1)
    assert all(trades.exit.iloc[:-1].to_numpy()<trades.entry.iloc[1:].to_numpy())
    return trades, cash-1, old_dd, close_dd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    bars, funds=load(args.data)
    expected=pd.read_csv(Path(__file__).with_name("backtests_ksl.csv"))
    records=[]
    for runtime in (False, True):
        choices, atr=decisions(bars,funds.fundingRate,runtime)
        if not runtime:
            verify_decisions(bars, funds.fundingRate, choices, atr)
        for row in expected.itertuples():
            if row.estrategia == "KSL1":
                continue
            tr,ret,old_dd,close_dd=simulate(bars,funds,choices,atr,row.inicio_UTC,row.fim_exclusivo_UTC,
                                            row.multiplicador_custos,runtime,row.estrategia=="KSO1316")
            longs=tr[tr.module=="long_breakout"]
            if not runtime:
                assert len(tr)==row.operacoes and len(longs)==row.long
                np.testing.assert_allclose([ret,old_dd,longs.net.sum()],
                    [row.retorno_liquido,row.drawdown_modelado,row.pnl_long_sobre_capital_inicial],atol=1e-10,rtol=0)
            records.append([runtime,row.inicio_UTC,row.fim_exclusivo_UTC,row.multiplicador_custos,
                            row.estrategia,len(tr),len(longs),ret,old_dd,close_dd,longs.net.sum()])
            if row.inicio_UTC=="2019-12-01" and row.fim_exclusivo_UTC=="2026-11-23" and row.multiplicador_custos==1:
                if row.estrategia=="KSO1316":
                    short_entries=set(tr.entry)
                else:
                    combined=set(tr.loc[tr.module!="long_breakout","entry"])
                    blocked={k for k in short_entries-combined if ((longs.entry<k)&(longs.exit>=k)).any()}
                    print("executor",runtime,"shorts ausentes",len(short_entries-combined),
                          "durante LONG",len(blocked),"shorts exclusivos KSL2",len(combined-short_entries),flush=True)
    pd.DataFrame(records,columns=["executor_aproximado","inicio","fim_exclusivo","custos","estrategia",
        "operacoes","long","retorno","dd_convencao_original","dd_fechamentos","contribuicao_long"]).to_csv(args.output,index=False)


if __name__=="__main__":
    main()
