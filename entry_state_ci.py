"""CI entrypoint using Binance's official public market-data base endpoint."""
import backtest

backtest.BINANCE_GLOBAL_URL = "https://data-api.binance.vision/api/v3/klines"

from entry_state_diagnostics import main


if __name__ == "__main__":
    main()
