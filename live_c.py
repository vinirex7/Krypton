"""Entrypoint do Krypton Aggressive C live.

TESTNET continua sendo o padrão. Para produção, configure USE_TESTNET=false
explicitamente no .env do VPS antes de iniciar este arquivo.
"""
import logging

from aggressive_c_live import AggressiveCTradeBot
from config import LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)


class LiveAggressiveCTradeBot(AggressiveCTradeBot):
    """Runtime live com auto-reparo da proteção tática após reconciliação."""

    def _reconcile_exchange_state(self, startup=False):
        super()._reconcile_exchange_state(startup=startup)
        for symbol, pos in list(self.state.get("tactical_positions", {}).items()):
            if not self.binance.has_active_oco(symbol, pos.get("order_list_id")):
                pos["order_list_id"] = None
                if not self._protect_tactical(symbol):
                    self.protection_blocked = True
                    logging.getLogger("Krypton.AggressiveC").critical(
                        "Não foi possível restaurar OCO tática | %s", symbol
                    )
        self._refresh_protection_status()
        self._save_state()


if __name__ == "__main__":
    LiveAggressiveCTradeBot().run()
