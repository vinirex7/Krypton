"""Canonical live entrypoint for Krypton Aggressive C.

Default: starts the single long-running Spot-only live process.
Administrative commands only queue a local request for that already-running
process, avoiding a second Binance execution engine against the same state DB.
"""
import argparse

from config import LIVE_STRATEGY
from live_c import (
    CLEAR_HALT_REQUEST,
    DECISION_NOW_REQUEST,
    REBASE_MANUAL_CHANGE_REQUEST,
    LiveAggressiveCTradeBot,
)


def _queue(path, label: str):
    path.touch(exist_ok=True)
    print(f"{label} solicitado. A instância live consumirá o pedido em até ~30 segundos.")


def main():
    parser = argparse.ArgumentParser(description="Krypton Aggressive C live")
    parser.add_argument(
        "--decision-now",
        action="store_true",
        help="Pede à instância live em execução para rodar um ciclo agora, usando apenas candles diários fechados.",
    )
    parser.add_argument(
        "--clear-halt",
        action="store_true",
        help="Pede liberação do portfolio halt somente após reconciliação Spot e DD atual abaixo do limite.",
    )
    parser.add_argument(
        "--rebase-after-manual-change",
        action="store_true",
        help=(
            "Confirma uma alteração manual de saldo e pede à instância live que "
            "reconcilie o Spot e crie uma nova baseline de risco sem apagar posições."
        ),
    )
    args = parser.parse_args()

    if LIVE_STRATEGY != "AGGRESSIVE_C":
        raise RuntimeError(
            f"Unsupported LIVE_STRATEGY={LIVE_STRATEGY!r}. "
            "This release promotes only the validated AGGRESSIVE_C profile."
        )

    if args.clear_halt:
        _queue(CLEAR_HALT_REQUEST, "CLEAR HALT")
    if args.rebase_after_manual_change:
        _queue(REBASE_MANUAL_CHANGE_REQUEST, "REBASE AFTER MANUAL CHANGE")
    if args.decision_now:
        _queue(DECISION_NOW_REQUEST, "DECISION NOW")
    if args.clear_halt or args.rebase_after_manual_change or args.decision_now:
        return

    LiveAggressiveCTradeBot().run()


if __name__ == "__main__":
    main()
