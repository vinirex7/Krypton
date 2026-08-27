"""Diagnose whether Aggressive C depends on excessive single-asset target weights."""
from __future__ import annotations
import argparse, json
from collections import Counter
from datetime import datetime, timezone
import numpy as np, pandas as pd
import adaptive_portfolio as ap, backtest, cross_asset_hybrid_v2 as hv2, walk_forward as wf

ASSETS=["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT"]
ALPHA_WEIGHT=.45
TARGET_VOL=.30
CADENCE=45
backtest.BINANCE_GLOBAL_URL="https://data-api.binance.vision/api/v3/klines"

def run(start="2020-08-01",end="2026-08-25"):
    s=datetime.strptime(start,"%Y-%m-%d").replace(tzinfo=timezone.utc)
    e=datetime.strptime(end,"%Y-%m-%d").replace(tzinfo=timezone.utc)
    data=wf._prepare_data(ASSETS,s,e)
    alpha=hv2.simulate_breadth_allocator(data,ASSETS,s,e,target_vol=TARGET_VOL,top_n=2,min_selected=2,rebalance_days=CADENCE)
    log=alpha["rebalance_log"].copy()
    cols=[f"target_{x}" for x in ASSETS]
    if log.empty: raise AssertionError("alpha rebalance log vazio")
    log["max_single_target"]=log[cols].max(axis=1)
    log["max_single_portfolio_weight"]=log["max_single_target"]*ALPHA_WEIGHT
    log["target_gross_check"]=log[cols].sum(axis=1)
    pairs=[]
    stats={}
    for sym,col in zip(ASSETS,cols):
        selected=log[col]>0
        vals=log.loc[selected,col]
        stats[sym]={
            "selected_rebalances":int(selected.sum()),
            "selection_rate":float(selected.mean()),
            "mean_target_when_selected":float(vals.mean()) if len(vals) else 0.0,
            "median_target_when_selected":float(vals.median()) if len(vals) else 0.0,
            "max_target":float(vals.max()) if len(vals) else 0.0,
            "max_total_portfolio_weight":float(vals.max()*ALPHA_WEIGHT) if len(vals) else 0.0,
        }
    for _,row in log.iterrows():
        sel=tuple(sorted(sym.replace("USDT","") for sym,col in zip(ASSETS,cols) if float(row[col])>0))
        pairs.append("+".join(sel) if sel else "CASH")
    pair_counts=Counter(pairs)
    report={
        "start":start,"end":end,"rebalance_count":int(len(log)),
        "alpha_weight_in_C":ALPHA_WEIGHT,"target_vol":TARGET_VOL,
        "asset_stats":stats,"pair_counts":dict(pair_counts),
        "max_single_alpha_target":float(log.max_single_target.max()),
        "p95_single_alpha_target":float(log.max_single_target.quantile(.95)),
        "max_single_total_portfolio_weight":float(log.max_single_portfolio_weight.max()),
        "fraction_rebalances_single_alpha_target_gt_60pct":float((log.max_single_target>.60).mean()),
        "median_target_gross":float(log.target_gross_check.median()),
        "mean_target_gross":float(log.target_gross_check.mean()),
        "cash_rebalance_rate":float((log.target_gross_check==0).mean()),
        "interpretation_guardrail":"leave-one-out dependence is not automatically weight concentration; inspect target weights before changing strategy",
    }
    log["selected_pair"]=pairs
    log.to_csv("c_asset_exposure_rebalances.csv",index=False)
    with open("c_asset_exposure_report.json","w") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2)); return report

def main():
    p=argparse.ArgumentParser(); p.add_argument("--start",default="2020-08-01"); p.add_argument("--end",default="2026-08-25"); a=p.parse_args(); run(a.start,a.end)
if __name__=="__main__": main()
