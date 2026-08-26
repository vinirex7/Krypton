"""Small, economically motivated robustness matrix for Krypton.

The goal is NOT to grid-search the whole history. Every variant uses the same rolling
walk-forward folds; TP is optimized only inside each training window and then frozen
for the following OOS window.

All variants keep the frozen SOL/BTC/BNB universe. Diagnostics vary only risk and
the BTC>SMA200 gate, so an excluded ETH market cannot leak back into production research.

Outputs:
- robustness_folds.csv: every OOS fold for every variant
- robustness_summary.csv: comparable aggregate statistics
"""

import argparse
from datetime import datetime, timezone

import pandas as pd

import walk_forward as wf


VARIANTS = [
    {
        "name": "frozen_baseline",
        "symbols": ["SOLUSDT", "BTCUSDT", "BNBUSDT"],
        "risk": 0.010,
        "regime_filter": True,
    },
    {
        "name": "no_regime_diagnostic",
        "symbols": ["SOLUSDT", "BTCUSDT", "BNBUSDT"],
        "risk": 0.010,
        "regime_filter": False,
    },
    {
        "name": "risk_0_5",
        "symbols": ["SOLUSDT", "BTCUSDT", "BNBUSDT"],
        "risk": 0.005,
        "regime_filter": True,
    },
    {
        "name": "risk_1_5_stress",
        "symbols": ["SOLUSDT", "BTCUSDT", "BNBUSDT"],
        "risk": 0.015,
        "regime_filter": True,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Krypton OOS robustness matrix")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--train-days", type=int, default=wf.TRAIN_DAYS)
    parser.add_argument("--test-days", type=int, default=wf.TEST_DAYS)
    parser.add_argument("--step-days", type=int, default=wf.STEP_DAYS)
    parser.add_argument("--candidate-tp", nargs="+", type=float, default=wf.DEFAULT_CANDIDATE_TP)
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Download the union of assets once. BTC is also used by the regime filter.
    all_symbols = ["SOLUSDT", "BTCUSDT", "BNBUSDT"]
    data = wf._prepare_data(all_symbols, start, end)

    fold_frames = []
    summaries = []

    for variant in VARIANTS:
        print("\n" + "=" * 78)
        print(
            f"VARIANT: {variant['name']} | symbols={variant['symbols']} | "
            f"risk={variant['risk']:.2%} | regime={variant['regime_filter']}"
        )
        print("=" * 78)

        out = wf.run_walk_forward(
            data=data,
            symbols=variant["symbols"],
            start=start,
            end=end,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            candidate_tp=args.candidate_tp,
            risk_per_trade=variant["risk"],
            regime_filter=variant["regime_filter"],
            label=variant["name"],
        )
        fold_frames.append(out)
        s = wf.summarize(out)
        if s:
            summaries.append({
                "variant": variant["name"],
                "symbols": ",".join(variant["symbols"]),
                "risk_per_trade": variant["risk"],
                "regime_filter": variant["regime_filter"],
                **s,
            })

    folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    summary = pd.DataFrame(summaries)

    if not summary.empty:
        # Robustness score: reward OOS Sharpe/return consistency, penalize DD and halts.
        # This is descriptive only; it is NOT used to choose parameters inside folds.
        summary["return_over_dd"] = summary.apply(
            lambda r: r["compounded_return"] / abs(r["worst_drawdown"])
            if r["worst_drawdown"] < 0 else 0.0,
            axis=1,
        )
        summary["robustness_score"] = (
            summary["mean_sharpe"]
            + 0.5 * summary["profitable_pct"]
            + 0.5 * summary["return_over_dd"]
            - 0.25 * summary["halted_folds"]
        )
        summary = summary.sort_values(
            ["robustness_score", "mean_sharpe", "compounded_return"],
            ascending=False,
        ).reset_index(drop=True)

    folds.to_csv("robustness_folds.csv", index=False)
    summary.to_csv("robustness_summary.csv", index=False)

    print("\n\nROBUSTNESS SUMMARY")
    if summary.empty:
        print("Nenhum resultado.")
        return

    display_cols = [
        "variant", "folds", "profitable_pct", "compounded_return",
        "mean_sharpe", "worst_drawdown", "mean_win_rate", "trades",
        "halted_folds", "return_over_dd", "robustness_score",
    ]
    print(summary[display_cols].to_string(index=False))
    print("\nArquivos salvos: robustness_folds.csv e robustness_summary.csv")
    print(
        "\nIMPORTANTE: use a tabela para comparar robustez. Não escolha uma variante "
        "somente pelo maior retorno; prefira consistência entre folds, Sharpe maior e DD menor."
    )


if __name__ == "__main__":
    main()
