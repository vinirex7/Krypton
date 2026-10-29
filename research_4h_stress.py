"""Stress temporal e de custos da candidata 4H. Só pesquisa; não negocia."""
import json
from pathlib import Path
import pandas as pd
import research_4h_atr_reversal as r

OUT=Path("results/4h_study")
OUT.mkdir(parents=True,exist_ok=True)
COSTS=[
    ("baseline",0.0010,0.0005),
    ("moderate_cost",0.0015,0.0010),
    ("severe_cost",0.0020,0.0015),
]
RISKS=[0.0075,0.0100]
SMA_PERIODS=[800,1200,1600]


def years(equity):
    out=[]
    for year in range(2021,2027):
        start=f"{year}-01-01"
        end=f"{year+1}-01-01" if year<2026 else r.END_EXCLUSIVE
        m=r.measure(equity,start,end)
        out.append({"year":year,"return_pct":m.get("return_pct"),"maxdd_pct":m.get("maxdd_pct")})
    return out


def folds(equity):
    out=[]
    start=pd.Timestamp(r.SPLIT,tz="UTC")
    end=pd.Timestamp(r.END_EXCLUSIVE,tz="UTC")
    while start<end:
        stop=min(start+pd.Timedelta(days=90),end)
        m=r.measure(equity,str(start.date()),str(stop.date()))
        out.append({"start":str(start.date()),"end_exclusive":str(stop.date()),
                    "return_pct":m.get("return_pct"),"maxdd_pct":m.get("maxdd_pct")})
        start=stop
    return out


def main():
    idx,pack,sma1200=r.prepare()
    btc=pd.Series(pack["BTCUSDT"]["c"],index=idx)
    rows=[]
    for sma_period in SMA_PERIODS:
        sma=btc.rolling(sma_period,min_periods=sma_period).mean().to_numpy()
        for name,fee,slip in COSTS:
            for risk in RISKS:
                r.FEE_RATE=fee
                r.ENTRY_SLIPPAGE_PCT=slip
                r.EXIT_SLIPPAGE_PCT=slip
                key=f"HALF5_risk{risk*100:.2f}_sma{sma_period}_{name}"
                variant=(key,5.0,"signal","half")
                m,eq,trades=r.run_variant(variant,idx,pack,sma,risk_per_entry=risk)
                record={"scenario":name,"risk_pct":risk*100,"sma_4h_bars":sma_period,
                        "roundtrip_nominal_cost_pct":200*(fee+slip),
                        "metrics":m,"annual":years(eq),"holdout_90d_folds":folds(eq)}
                rows.append(record)
                print("STRESS_RESULT "+json.dumps(record,allow_nan=False),flush=True)
    # Precommit the robustness judgment: not a new parameter-selection exercise.
    baseline=[x for x in rows if x["scenario"]=="baseline" and x["sma_4h_bars"]==1200]
    print("STRESS_BASELINE "+json.dumps(baseline,allow_nan=False),flush=True)
    (OUT/"robustness.json").write_text(json.dumps({"rows":rows,"selected_before_stress":"HALF5_risk1.00_sma1200_baseline","alternate_conservative":"HALF5_risk0.75_sma1200_baseline"},indent=2,allow_nan=False),encoding="utf-8")
    print("STRESS_COMPLETED",len(rows),flush=True)


if __name__=="__main__":
    main()
