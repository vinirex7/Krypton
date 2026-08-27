"""Cross-Asset Alpha v2.1: soft quality ranking without AND-gating exposure.

Structural correction after v2 showed that stacking hard filters cut active states
roughly in half. v2.1 preserves v1 eligibility and 15% volatility targeting, but
ranks eligible assets by momentum quality and trend persistence. No new threshold.
"""
from __future__ import annotations
from math import sqrt
import numpy as np
import pandas as pd
import adaptive_portfolio as ap
import cross_asset_allocation as ca
from config import ENTRY_SLIPPAGE_PCT, EXIT_SLIPPAGE_PCT, FEE_RATE
MOMENTUM_WINDOWS=(30,90,180); VOL_WINDOW=20; COV_WINDOW=60; PERSISTENCE_WINDOW=20

def target_weights_v21(data, symbols, ts, *, target_vol=0.15, top_n=2, min_selected=2):
    close=ca._aligned_close(data,symbols).loc[:ts]
    if len(close)<221: return {s:0.0 for s in symbols}
    scores={}; vols={}
    for s in symbols:
        series=close[s].dropna()
        if ts not in series.index or len(series.loc[:ts])<221: continue
        hist=series.loc[:ts]; sma=data[s]['sma200'].reindex(hist.index)
        if pd.isna(sma.loc[ts]) or float(hist.loc[ts])<=float(sma.loc[ts]): continue
        moms=[float(hist.loc[ts]/hist.shift(w).loc[ts]-1.0) for w in MOMENTUM_WINDOWS]
        if not all(np.isfinite(moms)): continue
        base=float(np.mean(moms))
        if base<=0: continue
        quality=sum(m>0 for m in moms)/len(moms)
        recent=pd.DataFrame({'p':hist,'sma':sma}).tail(PERSISTENCE_WINDOW).dropna()
        if recent.empty: continue
        persistence=float((recent.p>recent.sma).mean())
        rv=float(hist.pct_change().rolling(VOL_WINDOW).std().loc[ts]*sqrt(365.0))
        if not np.isfinite(rv) or rv<=0: continue
        scores[s]=base*quality*persistence; vols[s]=rv
    selected=sorted(scores,key=scores.get,reverse=True)[:top_n]
    if len(selected)<min_selected: return {s:0.0 for s in symbols}
    inv=np.array([1/vols[s] for s in selected],dtype=float); base_w=inv/inv.sum()
    ret=close[selected].pct_change().dropna().tail(COV_WINDOW)
    if len(ret)<30: return {s:0.0 for s in symbols}
    cov=ret.cov().to_numpy(dtype=float)*365.0
    pv=sqrt(max(float(base_w@cov@base_w),0.0)); scale=min(1.0,target_vol/pv) if pv>0 else 0.0
    out={s:0.0 for s in symbols}
    for s,w in zip(selected,base_w*scale): out[s]=float(w)
    return out

def simulate_alpha_v21(data,symbols,start,end,*,rebalance_days=45,target_vol=0.15,top_n=2,min_selected=2):
    start_ts,end_ts=ap.as_utc(start),ap.as_utc(end)
    cal=sorted(set.intersection(*[set(data[s]['df'].loc[start_ts:end_ts].index) for s in symbols]))
    if not cal:
        e=pd.Series(dtype=float); return {**ap.performance_metrics(e),'equity_curve':e,'rebalance_log':pd.DataFrame()}
    cash=ap.INITIAL_CAPITAL; qty={s:0.0 for s in symbols}; pending=None; sig=None; last=None; pts=[]; logs=[]
    for i,ts in enumerate(cal):
        if pending is not None:
            pre=ca._mark(cash,qty,data,ts,'open'); opens={s:float(data[s]['df'].loc[ts,'open']) for s in symbols}
            for s in symbols:
                tv=pre*float(pending.get(s,0)); cv=qty[s]*opens[s]
                if cv>tv and qty[s]>0:
                    q=min(qty[s],(cv-tv)/opens[s]); px=opens[s]*(1-EXIT_SLIPPAGE_PCT); gross=q*px
                    cash+=gross*(1-FEE_RATE); qty[s]-=q
            eq=ca._mark(cash,qty,data,ts,'open')
            for s in symbols:
                tv=eq*float(pending.get(s,0)); cv=qty[s]*opens[s]; deficit=max(0.0,tv-cv)
                if deficit<=0 or cash<=0: continue
                px=opens[s]*(1+ENTRY_SLIPPAGE_PCT); spend=min(deficit,cash/(1+FEE_RATE)); q=spend/px
                cash-=q*px*(1+FEE_RATE); qty[s]+=q
            logs.append({'signal_time':sig,'execution_time':ts,'target_gross':sum(pending.values()),
                         'selected':sum(float(v)>0 for v in pending.values()),**{f'target_{s}':pending.get(s,0) for s in symbols}})
            pending=None
        pts.append((ts,ca._mark(cash,qty,data,ts,'close')))
        if last is None or i-last>=rebalance_days:
            pending=target_weights_v21(data,symbols,ts,target_vol=target_vol,top_n=top_n,min_selected=min_selected); sig=ts; last=i
    ts=cal[-1]
    for s in symbols:
        if qty[s]<=0: continue
        px=float(data[s]['df'].loc[ts,'close'])*(1-EXIT_SLIPPAGE_PCT); cash+=qty[s]*px*(1-FEE_RATE); qty[s]=0
    pts[-1]=(ts,cash); equity=pd.Series([v for _,v in pts],index=[t for t,_ in pts],dtype=float); log=pd.DataFrame(logs)
    if not log.empty and not bool((pd.to_datetime(log.execution_time,utc=True)>pd.to_datetime(log.signal_time,utc=True)).all()):
        raise AssertionError('look-ahead detectado no Alpha v2.1')
    return {**ap.performance_metrics(equity),'equity_curve':equity,'rebalance_log':log}
