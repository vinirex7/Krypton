"""Frozen stage-2 validation for soft-ranking Cross-Asset Alpha v2.1."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
import numpy as np,pandas as pd
import adaptive_portfolio as ap, backtest, cross_asset_alpha_v21 as a21, cross_asset_hybrid_v2 as hv2, deep_validation as dv, walk_forward as wf
LIVE=list(wf.BASE_WEIGHTS); ASSETS=['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT']; WEIGHTS=(.10,.15,.20,.25)
CADENCE=45; TARGET_VOL=.15; SLEEVE_DAYS=90; TRANSFER=.003; HURDLE=.15
backtest.BINANCE_GLOBAL_URL='https://data-api.binance.vision/api/v3/klines'

def periods(curves,s,e):
    wins={'2022':('2022-01-01','2022-12-31'),'2023':('2023-01-01','2023-12-31'),'2024':('2024-01-01','2024-12-31'),
          '2025_h1':('2025-01-01','2025-06-30'),'2025':('2025-01-01','2025-12-31'),'2025_plus':('2025-01-01',e),
          '2026':('2026-01-01',e),'full':(s,e)}
    return pd.DataFrame([{'variant':n,'period':p,**ap.performance_metrics(ap.slice_and_rebase(c,a,b))}
                         for n,c in curves.items() for p,(a,b) in wins.items()])

def run(start,end,mc_runs=5000):
    s=datetime.strptime(start,'%Y-%m-%d').replace(tzinfo=timezone.utc); e=datetime.strptime(end,'%Y-%m-%d').replace(tzinfo=timezone.utc)
    data=wf._prepare_data(ASSETS,s,e); frozen=wf.simulate_portfolio(data,LIVE,s,e,3.0,risk_per_trade=.01,regime_filter=True)
    base=ap.simulate_tactical(data,LIVE,s,e)
    if not np.isclose(frozen['final_capital'],base['final_capital'],rtol=0,atol=1e-7): raise AssertionError('baseline divergiu')
    cont=ap.simulate_tactical(data,LIVE,s,e,cost_aware=True,entry_permission=ap.persistent_state_permission(data,LIVE))
    v1=hv2.simulate_breadth_allocator(data,ASSETS,s,e,target_vol=TARGET_VOL,top_n=2,min_selected=2,rebalance_days=CADENCE)
    v21=a21.simulate_alpha_v21(data,ASSETS,s,e,rebalance_days=CADENCE,target_vol=TARGET_VOL,top_n=2,min_selected=2)
    curves={'baseline':base['equity_curve'],'challenger_v1':hv2.combine_rebalanced_sleeves(cont['equity_curve'],v1['equity_curve'],.10,rebalance_days=SLEEVE_DAYS,transfer_cost=TRANSFER)}
    for w in WEIGHTS: curves[f'alpha_v21_a{int(w*100)}']=hv2.combine_rebalanced_sleeves(cont['equity_curve'],v21['equity_curve'],w,rebalance_days=SLEEVE_DAYS,transfer_cost=TRANSFER)
    p=periods(curves,s,e); ix=p.set_index(['variant','period']); v1f=ix.loc[('challenger_v1','full')]; v1v=ix.loc[('challenger_v1','2025_plus')]; v1h=ix.loc[('challenger_v1','2025_h1')]
    a10=ix.loc[('alpha_v21_a10','full')]; a10v=ix.loc[('alpha_v21_a10','2025_plus')]
    ar={'cagr_better':bool(a10.cagr>v1f.cagr),'sharpe_better':bool(a10.sharpe>v1f.sharpe),'calmar_better':bool(a10.calmar>v1f.calmar),
        'dd_under_15':bool(a10.max_drawdown>=-.15),'2025_plus_not_worse':bool(a10v['return']>=v1v['return']),'2026_positive':bool(ix.loc[('alpha_v21_a10','2026'),'return']>0)}
    arch={'passed':all(ar.values()),**ar}; gates={}
    for w in WEIGHTS[1:]:
        n=f'alpha_v21_a{int(w*100)}'; f=ix.loc[(n,'full')]; val=ix.loc[(n,'2025_plus')]; h1=ix.loc[(n,'2025_h1')]
        r={'architecture_passed':arch['passed'],'cagr_ge_15':bool(f.cagr>=HURDLE),'sharpe_ge_v1':bool(f.sharpe>=v1f.sharpe),
           'calmar_ge_v1':bool(f.calmar>=v1f.calmar),'dd_under_15':bool(f.max_drawdown>=-.15),'2025_plus_ge_v1':bool(val['return']>=v1v['return']),
           '2025_h1_not_2pp_worse':bool(h1['return']>=v1h['return']-.02),'2022_gt_minus5':bool(ix.loc[(n,'2022'),'return']>=-.05),'2026_positive':bool(ix.loc[(n,'2026'),'return']>0)}
        gates[n]={'passed':all(r.values()),**r}
    selected=next((f'alpha_v21_a{int(w*100)}' for w in WEIGHTS[1:] if gates[f'alpha_v21_a{int(w*100)}']['passed']),None)
    full=pd.DataFrame([{'variant':n,**ap.performance_metrics(c)} for n,c in curves.items()]); mc_names=['baseline','challenger_v1','alpha_v21_a10']+([selected] if selected else [])
    mc={n:dv.block_bootstrap_monte_carlo(curves[n].pct_change().dropna(),runs=mc_runs,seed=42) for n in mc_names}
    full.to_csv('cross_asset_alpha_v21_full.csv',index=False); p.to_csv('cross_asset_alpha_v21_periods.csv',index=False); pd.DataFrame(curves).to_csv('cross_asset_alpha_v21_curves.csv'); v21['rebalance_log'].to_csv('cross_asset_alpha_v21_rebalances.csv',index=False)
    report={'start':start,'end':end,'architecture_gate':arch,'weight_gates':gates,'selected_candidate':selected,'monte_carlo':mc,
            'guardrails':{'live_changed':False,'soft_ranking_frozen_before_results':True,'spot_only':True,'leverage':1.0,'promotion_requires_future_paper':True}}
    with open('cross_asset_alpha_v21_report.json','w') as f: json.dump(report,f,indent=2,default=str)
    print(full.to_string(index=False)); print(p.pivot(index='variant',columns='period',values='return').to_string()); print(json.dumps(report,indent=2)); return report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--start',default='2020-08-01'); p.add_argument('--end',default='2026-08-25'); p.add_argument('--mc-runs',type=int,default=5000); a=p.parse_args(); run(a.start,a.end,a.mc_runs)
if __name__=='__main__': main()
