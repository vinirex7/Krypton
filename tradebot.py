"""Canonical live entrypoint for Krypton.

The validated live strategy is Aggressive C. The former tactical-only engine is
preserved in Git history but is no longer started by the production command.
Repository defaults remain TESTNET-safe; real capital requires USE_TESTNET=false
explicitly in the VPS .env.
"""
from config import LIVE_STRATEGY
from live_c import LiveAggressiveCTradeBot


def main():
    if LIVE_STRATEGY != "AGGRESSIVE_C":
        raise RuntimeError(
            f"Unsupported LIVE_STRATEGY={LIVE_STRATEGY!r}. "
            "This release promotes only the validated AGGRESSIVE_C profile."
        )
    LiveAggressiveCTradeBot().run()


if __name__ == "__main__":
    main()
