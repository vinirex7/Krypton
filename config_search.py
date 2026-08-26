"""Multiple-testing-safe Krypton research entrypoint.

This intentionally delegates to research_validation, which persists every
candidate to config_search_results.csv and reports Deflated Sharpe Ratio plus
White's Reality Check before touching the locked reserve.
"""
from research_validation import main


if __name__ == "__main__":
    main()
