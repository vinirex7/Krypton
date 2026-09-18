"""Validate ATR-hysteresis BTC core v3 without changing Krypton live."""
from __future__ import annotations

import argparse, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

import adaptive_core_v3 as corev3
import adaptive_portfolio as ap
import backtest
import deep_validation as dv
import walk_forward as wf

SYMBOLS = list(wf.BASE_WEIGHTS)
WEIGHTS = (0.10, 0.20, 0.30)
BUFFERS = (0.5, 1.0)
backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"


def periods_for(curves, start, end):
    windows = {
        "development": (start, "2024-12-31"),
        "2022_bear": ("2022-01-01", "2022-12-31"),
        "2025_h1": ("2025-01-01", "2025-06-30"),
        "2025_h2": ("2025-07-01", "2025-12-31"),
        "validation_2025_plus": ("2025-01-01", end),
        "2026_diagnostic": ("2026-01-01", end),
        "full": (start, end),
    }
    rows=[]
    for name, curve in curves.items():
        for period,(a,b) in windows.items():
            rows.append({"variant":name,"period":period,
                         **ap.performance_metrics(ap.slice_and_rebase(curve,a,b))})
    return pd.DataFrame(rows)


def checks_for(periods):
    p=periods.set_index(["variant","period"])
    base_full=p.loc[("baseline","full")]
    base_val=p.loc[("baseline","validation_2025_plus")]
    base_h1=p.loc[("baseline","2025_h1")]
    out={}
    for name in periods.variant.unique():
        if name in {"baseline","continuity_tactical"} or name.startswith("core_only"):
            continue
        full=p.loc[(name,"full")]; dev=p.loc[(name,"development")]
        val=p.loc[(name,"validation_2025_plus")]; bear=p.loc[(name,"2022_bear")]
        h1=p.loc[(name,"2025_h1")]; d26=p.loc[(name,"2026_diagnostic")]
        rules={
            "dev_profitable": bool(dev["return"]>0),
            "full_cagr_15pct_better": bool(full["cagr"]>base_full["cagr"]*1.15),
            "full_calmar_better": bool(full["calmar"]>base_full["calmar"]),
            "validation_beats_baseline": bool(val["return"]>base_val["return"]),
            "max_dd_under_15pct": bool(full["max_drawdown"]>=-0.15),
            "bear_2022_above_minus_5pct": bool(bear["return"]>=-0.05),
            "h1_2025_not_over_2pp_worse": bool(h1["return"]>=base_h1["return"]-0.02),
            "diagnostic_2026_profitable": bool(d26["return"]>0),
        }
        out[name]={"passed":all(rules.values()),**rules}
    return out


def run(start,end,mc_runs=5000):
    s=datetime.strptime(start,"%Y-%m-%d").replace(tzinfo=timezone.utc)
    e=datetime.strptime(end,"%Y-%m-%d").replace(tzinfo=timezone.utc)
    data=wf._prepare_data(SYMBOLS,s,e)
    frozen=wf.simulate_portfolio(data,SYMBOLS,s,e,3.0,risk_per_trade=0.01,regime_filter=True)
    baseline=ap.simulate_tactical(data,SYMBOLS,s,e)
    if not np.isclose(frozen["final_capital"],baseline["final_capital"],rtol=0,atol=1e-7):
        raise AssertionError("baseline divergiu")
    continuity=ap.simulate_tactical(data,SYMBOLS,s,e,cost_aware=True,
                                    entry_permission=ap.persistent_state_permission(data,SYMBOLS))
    curves={"baseline":baseline["equity_curve"],"continuity_tactical":continuity["equity_curve"]}
    core_logs=[]
    for buf in BUFFERS:
        core=corev3.simulate_btc_trend_core(data,s,e,entry_atr_mult=buf,slope_lookback=20)
        curves[f"core_only_b{buf}"]=core["equity_curve"]
        if not core["trade_log"].empty:
            x=core["trade_log"].copy(); x["buffer_atr"]=buf; core_logs.append(x)
        for w in WEIGHTS:
            name=f"b{buf}_core{int(w*100)}_continuity{100-int(w*100)}"
            curves[name]=ap.combine_sleeves({"core":core["equity_curve"],"sat":continuity["equity_curve"]},
                                            {"core":w,"sat":1-w})
    periods=periods_for(curves,s,e)
    checks=checks_for(periods)
    robust=[]
    for w in WEIGHTS:
        names=[f"b{b}_core{int(w*100)}_continuity{100-int(w*100)}" for b in BUFFERS]
        if all(checks[n]["passed"] for n in names): robust.append(w)
    selected_weight=min(robust) if robust else None
    selected=None if selected_weight is None else f"b1.0_core{int(selected_weight*100)}_continuity{100-int(selected_weight*100)}"
    mc={}
    for name in ["baseline","continuity_tactical"] + [n for n in checks if checks[n]["passed"]]:
        mc[name]=dv.block_bootstrap_monte_carlo(curves[name].pct_change().dropna(),runs=mc_runs,seed=42)
    full=pd.DataFrame([{"variant":n,**ap.performance_metrics(c)} for n,c in curves.items()])
    full.to_csv("adaptive_core_v3_full_results.csv",index=False)
    periods.to_csv("adaptive_core_v3_period_results.csv",index=False)
    pd.DataFrame(curves).sort_index().ffill().to_csv("adaptive_core_v3_equity_curves.csv")
    pd.concat(core_logs,ignore_index=True).to_csv("adaptive_core_v3_trades.csv",index=False) if core_logs else pd.DataFrame().to_csv("adaptive_core_v3_trades.csv",index=False)
    report={"start":start,"end":end,"buffers_tested":list(BUFFERS),"weights_tested":list(WEIGHTS),
            "selected_candidate":selected,"robust_weights":robust,"promotion_checks":checks,
            "monte_carlo":mc,"guardrails":{"live_changed":False,"spot_only":True,"leverage":1.0,
            "2026_is_pristine_holdout":False,"promotion_requires_future_paper":True}}
    with open("adaptive_core_v3_report.json","w") as f: json.dump(report,f,indent=2,default=str)
    print("\nCORE V3 FULL RESULTS\n",full.to_string(index=False))
    print("\nRETURNS\n",periods.pivot(index="variant",columns="period",values="return").to_string())
    print("\nCHECKS\n",json.dumps(checks,indent=2))
    print("\nSELECTED",selected or "NONE")
    return report


def main():
    p=argparse.ArgumentParser(); p.add_argument("--start",default="2020-08-01"); p.add_argument("--end",default="2026-08-25"); p.add_argument("--mc-runs",type=int,default=5000)
    a=p.parse_args(); run(a.start,a.end,a.mc_runs)

if __name__=="__main__": main()
