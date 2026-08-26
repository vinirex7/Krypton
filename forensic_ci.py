"""CI entrypoint for forensic diagnostics.

GitHub-hosted runners may not reach api.binance.com. Binance documents
https://data-api.binance.vision as the base endpoint for public market-data
APIs. This keeps the exact Binance Spot USDT market; it is not proxy data.
"""
import backtest

# Official Binance public market-data-only REST base, same /api/v3/klines schema.
backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"

from forensic_diagnostics import main


if __name__ == "__main__":
    main()
