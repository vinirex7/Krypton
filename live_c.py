"""Entrypoint do Krypton Aggressive C live.

TESTNET continua sendo o padrão. Para produção, configure USE_TESTNET=false
explicitamente no .env do VPS antes de iniciar este arquivo.
"""
from aggressive_c_live import AggressiveCTradeBot


if __name__ == "__main__":
    AggressiveCTradeBot().run()
