"""Safe research selection pipeline for Krypton.

This is the canonical entrypoint used by config_search.py.

Protocol:
1. Evaluate the three predeclared regime hypotheses on selection OOS only.
2. Apply the concentration gate. If no regime passes, STOP: Stage 6 is skipped
   and the reserve remains locked.
3. Only an eligible regime may proceed to the single predefined DD overlay.
4. The reserve is never opened implicitly. It requires --open-reserve and is
   evaluated only after an eligible final candidate exists.

TP remains frozen at 3x ATR. No live configuration is changed here.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

import research_validation as rv
import walk_forward as wf
from config import TAKE_PROFIT_ATR_MULT, TRADING_PAIRS
from research_metrics import concentration_metrics, correlation_report, deflated_sharpe_ratio, white_reality_check


def _summary(results: pd.DataFrame, include_dsr: bool = False) -> pd.DataFrame:
    agg = {
        "mean_return": ("return", "mean"),
        "median_return": ("return", "median"),
        "mean_sharpe": ("sharpe", "mean"),
        "worst_dd": ("max_drawdown", "min"),
        "trades": ("trades", "sum"),
        "concentration_pass": ("concentration_pass", "all"),
    }
    if include_dsr:
        agg["dsr"] = ("deflated_sharpe_ratio", "first")
    return results.groupby("candidate").agg(**agg).reset_index()


def concentration_gate(summary: pd.DataFrame) -> dict:
    """Return a hard selection decision without touching Stage 6 or reserve."""
    if summary.empty:
        raise ValueError("Resumo de candidatas vazio.")
    ranked_all = summary.sort_values(["median_return", "mean_sharpe"], ascending=False)
    provisional = str(ranked_all.iloc[0]["candidate"])
    eligible = summary[summary["concentration_pass"]]
    if eligible.empty:
        return {
            "passed": False,
            "winner": None,
            "provisional": provisional,
            "reason": "Nenhum regime passou o gate de concentração.",
        }
    winner = str(
        eligible.sort_values(["median_return", "mean_sharpe"], ascending=False)
        .iloc[0]["candidate"]
    )
    return {"passed": True, "winner": winner, "provisional": provisional, "reason": None}


def _multiple_testing(candidate_returns: dict[str, pd.Series], bootstrap: int) -> tuple[dict, dict]:
    n_trials = len(candidate_returns)
    dsr = {name: deflated_sharpe_ratio(rets, n_trials) for name, rets in candidate_returns.items()}
    wrc = white_reality_check(candidate_returns, bootstrap_samples=bootstrap)
    return dsr, {"n_trials": n_trials, "dsr": dsr, "white_reality_check": wrc}


def _save_common_outputs(results, candidate_returns, candidate_exposure, symbols, folds, candidate_for_corr, bootstrap):
    dsr, mt = _multiple_testing(candidate_returns, bootstrap)
    results = results.copy()
    results["deflated_sharpe_ratio"] = results["candidate"].map(dsr)
    results.to_csv("config_search_results.csv", index=False)
    summary = _summary(results, include_dsr=True)
    summary.to_csv("config_search_summary.csv", index=False)

    if candidate_for_corr and candidate_for_corr in candidate_exposure:
        corr = correlation_report(
            _save_common_outputs.data,
            symbols,
            folds[0][1],
            folds[-1][2],
            candidate_exposure[candidate_for_corr],
        )
        corr["daily_return_correlation"].to_csv("correlation_daily_returns.csv")
        corr["position_correlation"].to_csv("correlation_positions.csv")
    return results, summary, mt


def _write_report(report: dict) -> None:
    with open("research_validation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


def main():
    if abs(TAKE_PROFIT_ATR_MULT - rv.FIXED_TP) > 1e-12:
        raise RuntimeError("selection_pipeline exige TAKE_PROFIT_ATR_MULT=3.0; TP não será pesquisado.")

    p = argparse.ArgumentParser(description="Krypton robust selection with hard gates")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    p.add_argument("--symbols", nargs="+", default=list(TRADING_PAIRS))
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--test-days", type=int, default=180)
    p.add_argument("--step-days", type=int, default=180)
    p.add_argument("--reserve-days", type=int, default=180)
    p.add_argument("--max-top5-share", type=float, default=0.50)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument(
        "--open-reserve",
        action="store_true",
        help="Abre a reserva final UMA ÚNICA VEZ, somente após todos os gates passarem.",
    )
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    reserve_start = end - timedelta(days=args.reserve_days - 1)
    selection_end = reserve_start - timedelta(days=1)
    symbols = [s.upper() for s in args.symbols]
    if set(symbols) != set(TRADING_PAIRS):
        raise ValueError("Experimento breadth exige exatamente os 3 ativos configurados em TRADING_PAIRS.")

    data = wf._prepare_data(symbols, start, end)
    _save_common_outputs.data = data
    folds = list(rv._folds(start, selection_end, args.train_days, args.test_days, args.step_days))
    if not folds:
        raise RuntimeError("Nenhum fold OOS completo antes da reserva final.")

    all_rows = []
    candidate_returns = {}
    candidate_exposure = {}

    # Stage 5: exactly the three predeclared regime hypotheses.
    for mode in rv.REGIME_MODES:
        label = f"regime_{mode}"
        rows, rets, exposure, _ = rv.evaluate_candidate(
            data, symbols, folds, label, mode, False, args.max_top5_share
        )
        all_rows.append(rows)
        candidate_returns[label] = rets
        candidate_exposure[label] = exposure

    first = pd.concat(all_rows, ignore_index=True)
    gate = concentration_gate(_summary(first))

    # HARD STOP: no eligible regime => no Stage 6 and no reserve access.
    if not gate["passed"]:
        results, final_summary, mt = _save_common_outputs(
            first,
            candidate_returns,
            candidate_exposure,
            symbols,
            folds,
            gate["provisional"],
            args.bootstrap,
        )
        report = {
            "tp_frozen_atr": rv.FIXED_TP,
            "selection_period_end": selection_end.date().isoformat(),
            "reserve_start": reserve_start.date().isoformat(),
            "reserve_end": end.date().isoformat(),
            "selection_status": "REJECTED_CONCENTRATION",
            "regime_winner_before_dd": None,
            "provisional_regime_for_diagnostics_only": gate["provisional"],
            "final_candidate": None,
            "selection_warning": gate["reason"],
            "stage6": {
                "status": "SKIPPED_NO_ELIGIBLE_REGIME",
                "reason": "Stage 6 só pode rodar após um regime passar concentração.",
            },
            "multiple_testing": mt,
            "reserve": {
                "status": "SKIPPED_LOCKED",
                "opened": False,
                "reason": "Nenhuma candidata passou o gate de concentração; holdout não foi acessado.",
            },
        }
        _write_report(report)
        print(final_summary.to_string(index=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    regime_winner = gate["winner"]
    selected_mode = regime_winner.replace("regime_", "")

    # Stage 6: one predefined overlay only, after an eligible regime exists.
    dd_label = f"{regime_winner}_dd_10_15"
    dd_rows, dd_rets, dd_exposure, _ = rv.evaluate_candidate(
        data, symbols, folds, dd_label, selected_mode, True, args.max_top5_share
    )
    all_rows.append(dd_rows)
    candidate_returns[dd_label] = dd_rets
    candidate_exposure[dd_label] = dd_exposure

    results = pd.concat(all_rows, ignore_index=True)
    # Initial candidate for correlation is resolved after DSR is attached.
    results, final_summary, mt = _save_common_outputs(
        results,
        candidate_returns,
        candidate_exposure,
        symbols,
        folds,
        regime_winner,
        args.bootstrap,
    )

    final_eligible = final_summary[final_summary["concentration_pass"]]
    if final_eligible.empty:
        raise RuntimeError("Invariante quebrada: gate inicial passou, mas nenhuma candidata final ficou elegível.")
    final_candidate = str(
        final_eligible.sort_values(["median_return", "mean_sharpe", "dsr"], ascending=False)
        .iloc[0]["candidate"]
    )
    final_mode = selected_mode if final_candidate == dd_label else final_candidate.replace("regime_", "")
    final_dd = final_candidate == dd_label

    report = {
        "tp_frozen_atr": rv.FIXED_TP,
        "selection_period_end": selection_end.date().isoformat(),
        "reserve_start": reserve_start.date().isoformat(),
        "reserve_end": end.date().isoformat(),
        "selection_status": "ELIGIBLE_FOR_RESERVE",
        "regime_winner_before_dd": regime_winner,
        "final_candidate": final_candidate,
        "selection_warning": None,
        "stage6": {"status": "EVALUATED", "candidate": dd_label},
        "multiple_testing": mt,
    }

    if not args.open_reserve:
        report["reserve"] = {
            "status": "LOCKED_NOT_OPENED",
            "opened": False,
            "reason": "Use --open-reserve somente quando a decisão de abrir o holdout for deliberada e final.",
        }
        _write_report(report)
        print(final_summary.to_string(index=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    reserve = rv.simulate(data, symbols, reserve_start, end, final_mode, final_dd)
    reserve_conc = concentration_metrics(reserve["trade_log"], reserve["equity_curve"], rv.INITIAL_CAPITAL)
    report["selection_status"] = "RESERVE_EVALUATED"
    report["reserve"] = {
        "status": "EVALUATED_CONSUMED",
        "opened": True,
        "return": reserve["return"],
        "sharpe": reserve["sharpe"],
        "max_drawdown": reserve["max_drawdown"],
        "trades": reserve["trades"],
        **reserve_conc,
    }
    _write_report(report)
    print(final_summary.to_string(index=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
