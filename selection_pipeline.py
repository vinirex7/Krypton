"""Safe research selection pipeline for Krypton.

Concentration is a robustness diagnostic, not an isolated veto for a
trend-following system. The final reserve is never opened implicitly: it
requires --open-reserve after the research decision has been made.

TP remains frozen at 3x ATR. This module never changes live configuration.
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


def summarize_results(results: pd.DataFrame, include_dsr: bool = False) -> pd.DataFrame:
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


def choose_regime(summary: pd.DataFrame) -> dict:
    """Prefer concentration-clean candidates, but do not hard-reject trend concentration alone."""
    if summary.empty:
        raise ValueError("Resumo de candidatas vazio.")
    eligible = summary[summary["concentration_pass"]]
    pool = eligible if not eligible.empty else summary
    winner = str(pool.sort_values(["median_return", "mean_sharpe"], ascending=False).iloc[0]["candidate"])
    warning = None if not eligible.empty else (
        "Nenhum regime passou todos os testes de concentração. "
        "Concentração será tratada como alerta e investigada por regime de mercado; "
        "a candidata não deve ser promovida apenas por este resultado."
    )
    return {"winner": winner, "warning": warning, "has_concentration_clean_candidate": not eligible.empty}


def _save_correlations(data, symbols, folds, exposure):
    corr = correlation_report(data, symbols, folds[0][1], folds[-1][2], exposure)
    corr["daily_return_correlation"].to_csv("correlation_daily_returns.csv")
    corr["position_correlation"].to_csv("correlation_positions.csv")


def _write_report(report: dict) -> None:
    with open("research_validation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


def main():
    if abs(TAKE_PROFIT_ATR_MULT - rv.FIXED_TP) > 1e-12:
        raise RuntimeError("selection_pipeline exige TAKE_PROFIT_ATR_MULT=3.0; TP não será pesquisado.")

    p = argparse.ArgumentParser(description="Krypton robust selection with explicit holdout access")
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
        help="Abre a reserva final deliberadamente; por padrão o holdout permanece lacrado.",
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
    folds = list(rv._folds(start, selection_end, args.train_days, args.test_days, args.step_days))
    if not folds:
        raise RuntimeError("Nenhum fold OOS completo antes da reserva final.")

    all_rows = []
    candidate_returns = {}
    candidate_exposure = {}

    # Stage 5: only the three predeclared regime hypotheses.
    for mode in rv.REGIME_MODES:
        label = f"regime_{mode}"
        rows, rets, exposure, _ = rv.evaluate_candidate(
            data, symbols, folds, label, mode, False, args.max_top5_share
        )
        all_rows.append(rows)
        candidate_returns[label] = rets
        candidate_exposure[label] = exposure

    first = pd.concat(all_rows, ignore_index=True)
    regime_choice = choose_regime(summarize_results(first))
    regime_winner = regime_choice["winner"]
    selected_mode = regime_winner.replace("regime_", "")

    # Stage 6 remains one predefined overlay, evaluated on selection data only.
    dd_label = f"{regime_winner}_dd_10_15"
    dd_rows, dd_rets, dd_exposure, _ = rv.evaluate_candidate(
        data, symbols, folds, dd_label, selected_mode, True, args.max_top5_share
    )
    all_rows.append(dd_rows)
    candidate_returns[dd_label] = dd_rets
    candidate_exposure[dd_label] = dd_exposure

    results = pd.concat(all_rows, ignore_index=True)
    n_trials = len(candidate_returns)
    dsr = {name: deflated_sharpe_ratio(rets, n_trials) for name, rets in candidate_returns.items()}
    wrc = white_reality_check(candidate_returns, bootstrap_samples=args.bootstrap)
    results["deflated_sharpe_ratio"] = results["candidate"].map(dsr)
    results.to_csv("config_search_results.csv", index=False)

    final_summary = summarize_results(results, include_dsr=True)
    final_summary.to_csv("config_search_summary.csv", index=False)
    final_clean = final_summary[final_summary["concentration_pass"]]
    final_pool = final_clean if not final_clean.empty else final_summary
    final_candidate = str(
        final_pool.sort_values(["median_return", "mean_sharpe", "dsr"], ascending=False)
        .iloc[0]["candidate"]
    )
    final_mode = selected_mode if final_candidate == dd_label else final_candidate.replace("regime_", "")
    final_dd = final_candidate == dd_label

    _save_correlations(data, symbols, folds, candidate_exposure[final_candidate])

    report = {
        "tp_frozen_atr": rv.FIXED_TP,
        "selection_period_end": selection_end.date().isoformat(),
        "reserve_start": reserve_start.date().isoformat(),
        "reserve_end": end.date().isoformat(),
        "selection_status": (
            "SELECTION_COMPLETE_WITH_CONCENTRATION_WARNING"
            if regime_choice["warning"] else "SELECTION_COMPLETE"
        ),
        "regime_winner_before_dd": regime_winner,
        "final_candidate": final_candidate,
        "selection_warning": regime_choice["warning"],
        "concentration_is_hard_gate": False,
        "stage6": {"status": "EVALUATED_SELECTION_ONLY", "candidate": dd_label},
        "multiple_testing": {
            "n_trials": n_trials,
            "dsr": dsr,
            "white_reality_check": wrc,
        },
    }

    if not args.open_reserve:
        report["reserve"] = {
            "status": "LOCKED_NOT_OPENED",
            "opened": False,
            "reason": "Holdout exige --open-reserve; revisar diagnósticos por regime antes de consumi-lo.",
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
