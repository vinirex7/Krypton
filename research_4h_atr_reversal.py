"""Pesquisa isolada: saídas ATR versus reversão no tático 4H do Krypton.
Não altera/aciona o runtime live. Dados oficiais Binance Spot, UTC.
Usa compute_signals/compute_atr/compute_supertrend do próprio repositório.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import (TRADING_PAIRS, FEE_RATE, ENTRY_SLIPPAGE_PCT,
                    EXIT_SLIPPAGE_PCT, RSI_LOW, RSI_HIGH,
                    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
                    SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER,
                    RSI_PERIOD)
from indicators import (compute_atr, compute_signals, compute_supertrend,
                        compute_rsi, compute_macd)

SOURCE = "https://data-api.binance.vision/api/v3/klines"
START_DATA = "2020-09-01"
START_TEST = "2021-01-01"
SPLIT = "2025-01-01"
END_EXCLUSIVE = "2026-10-28"
INITIAL = 10000.0
SYMBOLS = list(TRADING_PAIRS)
RISK = 0.02
HARD_STOP_ATR = 2.0
MAX_DD_HALT = 0.30
CB_DAY = 0.04
CACHE = Path("results/4h_study/data")
OUT = Path("results/4h_study")
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def millis(ts):
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def fetch_symbol(symbol):
    path = CACHE / f"{symbol}_{START_DATA}_{END_EXCLUSIVE}_4h.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        begin, stop = millis(START_DATA), millis(END_EXCLUSIVE)
        rows, cursor = [], begin
        with requests.Session() as session:
            while cursor < stop:
                params = {"symbol": symbol, "interval": "4h",
                          "startTime": cursor, "endTime": stop - 1, "limit": 1000}
                last = None
                for attempt in range(4):
                    try:
                        res = session.get(SOURCE, params=params, timeout=35)
                        res.raise_for_status()
                        page = res.json()
                        if not isinstance(page, list):
                            raise ValueError(f"Unexpected Binance payload {str(page)[:300]}")
                        break
                    except (requests.RequestException, ValueError) as exc:
                        last = exc
                        time.sleep(1.0 + attempt)
                else:
                    raise RuntimeError(f"{symbol}: Binance unavailable at {cursor}: {last}")
                if not page:
                    raise RuntimeError(f"{symbol}: incomplete data after {pd.to_datetime(cursor, unit='ms', utc=True)}")
                rows.extend(page)
                cursor = int(page[-1][0]) + 4 * 3600000
                print(f"DATA {symbol}: {pd.to_datetime(cursor, unit='ms', utc=True)}", flush=True)
                if len(page) < 1000 and cursor < stop:
                    raise RuntimeError(f"{symbol}: partial page/coverage")
        arr = np.asarray(rows, dtype=object)
        df = pd.DataFrame({
            "open": arr[:, 1].astype(float), "high": arr[:, 2].astype(float),
            "low": arr[:, 3].astype(float), "close": arr[:, 4].astype(float),
            "volume": arr[:, 5].astype(float),
        }, index=pd.to_datetime(arr[:, 0].astype("int64"), unit="ms", utc=True))
        df.to_csv(path, float_format="%.12f")
    expected = pd.date_range(start=pd.Timestamp(START_DATA, tz="UTC"), end=millis_to_ts(millis(END_EXCLUSIVE) - 4*3600000), freq="4h")
    if not df.index.equals(expected):
        raise ValueError(f"{symbol} noncontiguous 4H coverage: got {len(df)}, expected {len(expected)}; first={df.index.min()}, last={df.index.max()}")
    ohlc = df[["open", "high", "low", "close"]]
    if (ohlc <= 0).any().any() or not np.isfinite(ohlc.to_numpy()).all():
        raise ValueError(f"{symbol} nonfinite or nonpositive OHLC")
    if (df.high < df[["open", "low", "close"]].max(axis=1)).any() or (df.low > df[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError(f"{symbol} inconsistent OHLC")
    print(f"VALIDATED {symbol} {len(df)} exact 4H bars; {df.index[0]} -> {df.index[-1]}", flush=True)
    return df


def millis_to_ts(v):
    return pd.Timestamp(v, unit="ms", tz="UTC")


def prepare():
    frames = {sym: fetch_symbol(sym) for sym in SYMBOLS}
    common = frames[SYMBOLS[0]].index
    for sym in SYMBOLS[1:]:
        if not frames[sym].index.equals(common):
            raise ValueError("Misaligned markets")
    packs = {}
    for sym, frame in frames.items():
        atr = compute_atr(frame.high, frame.low, frame.close, period=14)
        sig = compute_signals(frame, st_period=SUPERTREND_PERIOD,
                              st_mult=SUPERTREND_MULTIPLIER,
                              rsi_period=RSI_PERIOD, rsi_low=RSI_LOW,
                              rsi_high=RSI_HIGH,
                              macd_fast=MACD_FAST, macd_slow=MACD_SLOW,
                              macd_sig=MACD_SIGNAL).to_numpy(dtype="int8")
        _, st = compute_supertrend(frame.high, frame.low, frame.close,
                                   SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
        rsi = compute_rsi(frame.close, RSI_PERIOD)
        ml, ms, _ = compute_macd(frame.close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        packs[sym] = {"o": frame.open.to_numpy(), "h": frame.high.to_numpy(),
                      "l": frame.low.to_numpy(), "c": frame.close.to_numpy(),
                      "atr": atr.to_numpy(), "sig": sig,
                      "st": st.to_numpy(), "rsi": rsi.to_numpy(),
                      "macd": ml.to_numpy(), "macdsig": ms.to_numpy()}
    btc_sma = frames["BTCUSDT"].close.rolling(1200, min_periods=1200).mean().to_numpy()
    return common, packs, btc_sma


VARIANTS = [
    ("TP2_ATR_and_signal", 2.0, "signal", None),
    ("TP3_ATR_and_signal_LIVE_ANALOG", 3.0, "signal", None),
    ("TP4_ATR_and_signal", 4.0, "signal", None),
    ("TP5_ATR_and_signal", 5.0, "signal", None),
    ("TP6_ATR_and_signal", 6.0, "signal", None),
    ("TP3_ATR_no_reversal", 3.0, "none", None),
    ("REV_signal_with_2ATR_hardstop", None, "signal", None),
    ("REV_supertrend_with_2ATR_hardstop", None, "supertrend", None),
    ("REV_MACD_RSI_with_2ATR_hardstop", None, "macdrsi", None),
    ("REV_signal_NO_HARD_STOP_UNSAFE", None, "signal", "nohard"),
    ("HYBRID_half_TP3_then_signal", 3.0, "signal", "half"),
    ("TRAIL3_ATR_plus_signal", None, "signal", "trail3"),
]


def should_reverse(p, kind, prev):
    if kind == "none":
        return False
    if kind == "signal":
        return int(p["sig"][prev]) != 1
    if kind == "supertrend":
        return np.isfinite(p["st"][prev]) and p["st"][prev] == -1
    if kind == "macdrsi":
        return p["macd"][prev] < p["macdsig"][prev] and p["rsi"][prev] < 48.0
    raise ValueError(kind)


def measure(series, start, end_excl):
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end_excl, tz="UTC")
    within = series.loc[(series.index >= start) & (series.index < end)]
    if len(within) < 2:
        return {}
    first_prior = series.loc[series.index < start]
    baseline = float(first_prior.iloc[-1]) if len(first_prior) else INITIAL
    ending = float(within.iloc[-1])
    returns = within.pct_change().dropna()
    maxdd = (within / np.maximum.accumulate(np.r_[baseline, within.to_numpy()])[1:] - 1.0).min()
    years = (within.index[-1] - start).total_seconds() / (365.0 * 86400.0)
    cagr = (ending / baseline) ** (1.0 / years) - 1.0 if years > 0 and ending > 0 else np.nan
    return {"return_pct": (ending/baseline-1)*100,
            "cagr_pct": float(cagr)*100,
            "maxdd_pct": float(maxdd)*100,
            "sharpe_4h": float(np.sqrt(365*6)*returns.mean()/returns.std()) if len(returns)>2 and returns.std()>0 else None,
            "start_equity": baseline, "end_equity": ending, "bars": len(within)}


def run_variant(params, idx, p, btc_sma, risk_per_entry=RISK):
    name, tp_atr, rev_mode, special = params
    cash = INITIAL
    openpos, trades, eqrows = {}, [], []
    peak, halted, daily_start, last_day = INITIAL, False, INITIAL, None
    fee_total = 0.0
    trade_n = 0

    def portfolio_price(i, f):
        return cash + sum(pos["qty"] * p[sym][f][i] for sym, pos in openpos.items())

    def sell(sym, qty, px, stamp, reason):
        nonlocal cash, fee_total, trade_n
        pos = openpos[sym]
        qty = min(float(qty), pos["qty"])
        if qty <= 1e-14:
            return
        fee = qty * px * FEE_RATE
        fee_total += fee
        cash += qty * px - fee
        pos["pnl"] += qty*(px-pos["entry"]) - fee
        pos["qty"] -= qty
        if pos["qty"] < 1e-11:
            trades.append({"sym": sym, "entry_t": str(pos["entered"]),
                           "exit_t": str(stamp), "pnl": pos["pnl"],
                           "reason": reason, "hold_bars": pos["hold"]})
            trade_n += 1
            openpos.pop(sym)

    start_i = int(idx.searchsorted(pd.Timestamp(START_TEST, tz="UTC")))
    for i in range(start_i, len(idx)):
        stamp, prev = idx[i], i-1
        open_equity = portfolio_price(i, "o")
        day = stamp.date()
        if day != last_day:
            last_day, daily_start = day, open_equity
        # Decision at prior completed 4H close -> fill at current OPEN.
        for sym in list(openpos):
            if should_reverse(p[sym], rev_mode, prev):
                sell(sym, openpos[sym]["qty"], p[sym]["o"][i]*(1-EXIT_SLIPPAGE_PCT), stamp, "REV_"+rev_mode)

        eq_open = portfolio_price(i, "o")
        btc = p["BTCUSDT"]["c"][prev]
        riskon = bool(np.isfinite(btc_sma[prev]) and btc > btc_sma[prev])
        if not halted and riskon and (daily_start-eq_open) / daily_start < CB_DAY:
            for sym in SYMBOLS:
                if sym in openpos or len(openpos)>=3:
                    continue
                sp = p[sym]
                av = float(sp["atr"][prev])
                signal_close = float(sp["c"][prev])
                if int(sp["sig"][prev])!=1 or not np.isfinite(av) or av<=0:
                    continue
                # Live fixed 3*ATR cost gate used for all exit candidates.
                round_trip = 2*FEE_RATE + ENTRY_SLIPPAGE_PCT + EXIT_SLIPPAGE_PCT
                if 3*av/signal_close <= 3*round_trip:
                    continue
                entry = float(sp["o"][i])*(1+ENTRY_SLIPPAGE_PCT)
                if entry <= 0 or abs(entry / signal_close-1)>0.005:
                    continue
                current_eq = portfolio_price(i, "o")
                qty = min(risk_per_entry*current_eq/(2*av),
                          current_eq*float(TRADING_PAIRS[sym])/entry,
                          cash/((1+FEE_RATE)*entry))
                if qty*entry < 5.0:
                    continue
                cost = qty*entry
                fee = cost*FEE_RATE
                cash -= cost+fee
                fee_total += fee
                openpos[sym] = {"qty":qty, "entry":entry, "atr0":av, "stop":entry-2*av,
                                "tp":entry+tp_atr*av if tp_atr is not None else None,
                                "pnl":-fee, "entered":stamp, "hold":0,
                                "highest":entry, "half_done":False}

        # OCO intrabar: pessimistic ordering if both touched.
        for sym in list(openpos):
            pos = openpos[sym]
            pos["hold"] += 1
            sp = p[sym]
            oo, hi, lo = float(sp["o"][i]), float(sp["h"][i]), float(sp["l"][i])
            stop = pos["stop"] if special!="nohard" else -math.inf
            tp = pos["tp"]
            if oo <= stop:
                sell(sym, pos["qty"], oo*(1-EXIT_SLIPPAGE_PCT), stamp, "STOP_GAP")
            elif lo <= stop:
                sell(sym, pos["qty"], stop*(1-EXIT_SLIPPAGE_PCT), stamp, "STOP")
            elif tp is not None and (oo>=tp or hi>=tp):
                if special=="half":
                    if not pos["half_done"]:
                        sell(sym, pos["qty"]*0.5, tp, stamp, "PARTIAL_TP")
                        if sym in openpos:
                            pos["half_done"]=True
                            pos["stop"]=max(pos["stop"],pos["entry"])
                            pos["tp"]=None
                else:
                    sell(sym, pos["qty"], tp, stamp, "TAKE_PROFIT")
            # Only trailing state from closed candle used on subsequent candle.
            if sym in openpos and special=="trail3":
                pos["highest"] = max(pos["highest"], hi)
                av_cur = float(sp["atr"][i])
                if np.isfinite(av_cur):
                    pos["stop"] = max(pos["stop"], pos["highest"]-3*av_cur)

        eq = portfolio_price(i, "c")
        peak = max(peak,eq)
        if peak>0 and (peak-eq)/peak >= MAX_DD_HALT:
            halted = True
        eqrows.append((stamp,eq))

    equity = pd.Series({a:b for a,b in eqrows}, dtype=float).sort_index()
    m = {"variant": name, "risk_pct": risk_per_entry*100, "train":measure(equity, START_TEST, SPLIT),
         "holdout":measure(equity, SPLIT, END_EXCLUSIVE),
         "overall":measure(equity, START_TEST, END_EXCLUSIVE),
         "trade_count_closed":trade_n,
         "trade_count_train":sum(pd.Timestamp(t["exit_t"])<pd.Timestamp(SPLIT,tz="UTC") for t in trades),
         "win_pct":100*sum(t["pnl"]>0 for t in trades)/len(trades) if trades else None,
         "profit_factor":(sum(max(0,t["pnl"]) for t in trades)/abs(sum(min(0,t["pnl"]) for t in trades)) if sum(min(0,t["pnl"]) for t in trades)<0 else None),
         "fees_paid_usdt":fee_total,
         "global_halted":halted,
         "open_positions_end":len(openpos),
         "trade_by_reason":{reason:sum(t["reason"]==reason for t in trades) for reason in sorted(set(t["reason"] for t in trades))}}
    return m, equity, trades


def main():
    print("EXPERIMENT: 4H tactical only; base code indicators, BTC 200-day SMA via 1200x4h", flush=True)
    idx, packs, btc_sma = prepare()
    summaries=[]
    for params in VARIANTS:
        start=time.time()
        m, eq, trades = run_variant(params, idx, packs, btc_sma)
        summaries.append(m)
        eq.rename("equity").to_csv(OUT/f"{params[0]}_equity.csv",float_format="%.10f")
        pd.DataFrame(trades).to_csv(OUT/f"{params[0]}_trades.csv", index=False)
        print("RESULT "+json.dumps(m,allow_nan=False),flush=True)
        print(f"TIMING {params[0]} {time.time()-start:.2f}s",flush=True)
    # Ranking based only on 2021–2024; 2025–2026 remains holdout.
    candidates=[m for m in summaries if m["train"] and
                m["train"]["maxdd_pct"]>=-22 and m["trade_count_train"]>=30 and m["train"]["return_pct"]>0]
    # Conservative selection prefers CAGR adjusted for drawdown.
    ranked=sorted(candidates,key=lambda x:(x["train"]["cagr_pct"]/(1+abs(x["train"]["maxdd_pct"])),
                                         x["train"]["cagr_pct"]),reverse=True)
    result={"metadata":{"base":"vinirex7/Krypton main @ 7cc2d4a",
                       "source":SOURCE,"start_data":START_DATA,"start_test":START_TEST,
                       "train_end_exclusive":SPLIT,"holdout_end_exclusive":END_EXCLUSIVE,
                       "symbols":SYMBOLS, "initial_usdt":INITIAL,
                       "risk_tactical":RISK,"hard_stop_atr":HARD_STOP_ATR,
                       "daily_circuit_breaker":CB_DAY,"global_halt_dd":MAX_DD_HALT,
                       "signal_at_previous_4h_close":True,"entry_at_next_4h_open":True,
                       "bias":"stop_first_when_same_candle",
                       "uses_4h_sma1200_in_place_of_daily_sma200":True,
                       "tactical_only":True,"alpha_45pct_not_simulated":True,
                       "rounding_filters_not_replayed":True,"live_equivalence":False,
                       "signals_use_original_indicators_but_continuous_history":True},
            "chosen_using_train_only":ranked[0]["variant"] if ranked else None,
            "summary":summaries}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    pd.json_normalize(summaries).to_csv(OUT/"summary.csv",index=False)
    print("TRAIN_SELECTED",result["chosen_using_train_only"],flush=True)
    print("COMPLETED",len(summaries),"VARIANTS",flush=True)

    # Predesignated robustness sweep: all selection decisions use TRAIN only.
    sweep_design = [
        ("TP3_full",3.0,"signal",None),
        ("TP5_full",5.0,"signal",None),
        ("REV_no_tp",None,"signal",None),
        ("HALF2_rev",2.0,"signal","half"),
        ("HALF3_rev",3.0,"signal","half"),
        ("HALF4_rev",4.0,"signal","half"),
        ("HALF5_rev",5.0,"signal","half"),
        ("TRAIL3_rev",None,"signal","trail3"),
    ]
    risk_grid = (0.005,0.0075,0.01,0.0125,0.015,0.02)
    sweep=[]
    for risk in risk_grid:
        for base in sweep_design:
            params=(base[0]+f"_risk{100*risk:.2f}pct",*base[1:])
            m,eq,trades = run_variant(params,idx,packs,btc_sma,risk_per_entry=risk)
            sweep.append(m)
            print("SWEEP_RESULT "+json.dumps(m,allow_nan=False),flush=True)
    passed=[m for m in sweep if m["train"]
            and m["train"]["maxdd_pct"]>=-22.0
            and m["train"]["return_pct"]>0.0
            and m["trade_count_train"]>=30
            and not (m["global_halted"] and m["holdout"]["return_pct"]==0.0)]
    # The last requirement rejects strategies halted in training or later;
    # it is an ex-post risk diagnostic and is NOT used to rank candidates.
    # Recompute selection without testing future state: train bounds alone.
    passed=[m for m in sweep if m["train"]
            and m["train"]["maxdd_pct"]>=-22.0
            and m["train"]["return_pct"]>0.0
            and m["trade_count_train"]>=30]
    ranked=sorted(passed,key=lambda x:(x["train"]["cagr_pct"]/
            (1.0+abs(x["train"]["maxdd_pct"])),x["train"]["cagr_pct"]),reverse=True)
    chosen=ranked[0] if ranked else None
    sweep_report={
       "risk_grid_pct":[100*r for r in risk_grid],
       "train_period":[START_TEST,SPLIT],
       "holdout_period":[SPLIT,END_EXCLUSIVE],
       "train_maxdd_limit_pct":-22.0,
       "selection_uses_holdout":False,
       "selection_criterion":"train CAGR / (1 + abs(train maxdd in percentage points)); train positive return and >=30 closed trades",
       "chosen_train_only":chosen["variant"] if chosen else None,
       "chosen_candidate":chosen,
       "candidates_passing_train_constraints":len(ranked),
       "rows":sweep,
    }
    (OUT/"risk_sweep_summary.json").write_text(json.dumps(sweep_report,indent=2,allow_nan=False),encoding="utf-8")
    pd.json_normalize(sweep).to_csv(OUT/"risk_sweep_summary.csv",index=False)
    print("RISK_SWEEP_SELECTED "+json.dumps(chosen,allow_nan=False) if chosen else "RISK_SWEEP_SELECTED NONE",flush=True)
    print("RISK_SWEEP_DONE",len(sweep),"configs",len(ranked),"pass train",flush=True)



if __name__=="__main__":
    main()
