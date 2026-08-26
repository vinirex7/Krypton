"""Multiple-testing-safe Krypton research entrypoint.

Canonical research flow:
- saves every tested candidate;
- applies concentration as a hard promotion gate;
- skips Stage 6 when no regime is eligible;
- keeps the reserve locked unless --open-reserve is explicitly supplied;
- reports Deflated Sharpe Ratio and White's Reality Check.
"""
from selection_pipeline import main


if __name__ == "__main__":
    main()
