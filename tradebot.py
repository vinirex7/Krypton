"""Canonical live entrypoint for Krypton Aggressive C.

Default: starts the single long-running Spot-only live process.
Administrative commands only queue a local request for that already-running
process, avoiding a second Binance execution engine against the same state DB.
"""
import argparse

from config import LIVE_STRATEGY
from live_c import CLEAR_HALT_REQUEST, DECISION_NOW_REQUEST, LiveAggressiveCTradeBot


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
    args = parser.parse_args()

    if LIVE_STRATEGY != "AGGRESSIVE_C":
        raise RuntimeError(
            f"Unsupported LIVE_STRATEGY={LIVE_STRATEGY!r}. "
            "This release promotes only the validated AGGRESSIVE_C profile."
        )

    if args.clear_halt:
        _queue(CLEAR_HALT_REQUEST, "CLEAR HALT")
    if args.decision_now:
        _queue(DECISION_NOW_REQUEST, "DECISION NOW")
    if args.clear_halt or args.decision_now:
        return

    LiveAggressiveCTradeBot().run()


if __name__ == "__main__":
    main()
