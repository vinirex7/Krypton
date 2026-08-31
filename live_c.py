"""Entrypoint do Krypton Aggressive C live.

TESTNET continua sendo o padrão. Para produção, configure USE_TESTNET=false
explicitamente no .env do VPS antes de iniciar este arquivo.

O runtime aceita dois pedidos administrativos por arquivos locais, consumidos
pela própria instância já em execução. Isso evita iniciar uma segunda instância
concorrente apenas para forçar uma decisão ou liberar um halt contábil.
"""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from aggressive_c_live import AggressiveCTradeBot
from config import (
    AGGRESSIVE_C_ALPHA_SYMBOLS,
    AGGRESSIVE_C_MAX_DRAWDOWN_PCT,
    LIVE_QUOTE_ASSET,
    LOG_FILE,
    USE_TESTNET,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("Krypton.AggressiveC")

CONTROL_DIR = Path(__file__).resolve().parent
DECISION_NOW_REQUEST = CONTROL_DIR / ".krypton_decision_now"
CLEAR_HALT_REQUEST = CONTROL_DIR / ".krypton_clear_halt"


class LiveAggressiveCTradeBot(AggressiveCTradeBot):
    """Runtime live com auto-reparo OCO e controles administrativos seguros."""

    def _reconcile_exchange_state(self, startup=False):
        super()._reconcile_exchange_state(startup=startup)
        for symbol, pos in list(self.state.get("tactical_positions", {}).items()):
            if not self.binance.has_active_oco(symbol, pos.get("order_list_id")):
                pos["order_list_id"] = None
                if not self._protect_tactical(symbol):
                    self.protection_blocked = True
                    logger.critical("Não foi possível restaurar OCO tática | %s", symbol)
        self._refresh_protection_status()
        self._save_state()

    @staticmethod
    def _consume_request(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def _clear_halt_if_safe(self) -> bool:
        """Libera somente um halt global contábil após reconciliação Spot segura.

        Não altera peak, histórico, posições nem um halt próprio do RiskManager
        tático. Se o alpha ficou zerado durante o halt, marca-o como devido para
        que a próxima decisão possa reconstruir o sleeve pela regra normal.
        """
        self._reconcile_exchange_state()
        equity, dd = self._portfolio_risk_update()
        if dd >= AGGRESSIVE_C_MAX_DRAWDOWN_PCT:
            logger.critical(
                "CLEAR HALT recusado | equity=%.2f %s | DD=%.2f%% >= %.2f%%",
                equity,
                LIVE_QUOTE_ASSET,
                dd * 100,
                AGGRESSIVE_C_MAX_DRAWDOWN_PCT * 100,
            )
            return False
        if self.tactical_risk.halted:
            logger.critical("CLEAR HALT recusado | RiskManager tático permanece halted")
            return False

        alpha_flat = all(
            float(self.state.get("alpha_qty", {}).get(symbol, 0.0)) <= 0
            for symbol in AGGRESSIVE_C_ALPHA_SYMBOLS
        )
        self.state["portfolio_halted"] = False
        if alpha_flat:
            self.state["last_alpha_rebalance"] = None
            logger.warning("Alpha estava zerado durante o halt; próximo ciclo reavalia o target imediatamente")
        self._save_state()
        logger.warning(
            "Portfolio halt liberado manualmente após reconciliação Spot | equity=%.2f %s | DD=%.2f%%",
            equity,
            LIVE_QUOTE_ASSET,
            dd * 100,
        )
        return True

    def run(self):
        logger.info(
            "Krypton Aggressive C iniciado | quote=%s | UTC | mode=%s | capital=SPOT_ONLY",
            LIVE_QUOTE_ASSET,
            "TESTNET" if USE_TESTNET else "PRODUÇÃO",
        )
        now = datetime.now(timezone.utc)
        last_daily_run = None

        # Pedidos deixados antes/restart são consumidos pela única instância live.
        if self._consume_request(CLEAR_HALT_REQUEST):
            try:
                self._clear_halt_if_safe()
            except Exception:
                logger.exception("Falha ao processar pedido manual de CLEAR HALT")

        manual_ran = False
        if self._consume_request(DECISION_NOW_REQUEST):
            try:
                logger.warning("DECISION NOW solicitado | executando ciclo com último candle diário fechado")
                self.daily_cycle()
                manual_ran = True
            except Exception:
                logger.exception("Falha no ciclo manual DECISION NOW")

        if not manual_ran and now.hour == 0 and 5 <= now.minute < 10:
            self.daily_cycle()
            last_daily_run = now.date()
        elif not manual_ran:
            logger.info("Boot fora da janela 00:05 UTC; nenhuma entrada atrasada será enviada.")

        while True:
            now = datetime.now(timezone.utc)

            if self._consume_request(CLEAR_HALT_REQUEST):
                try:
                    self._clear_halt_if_safe()
                except Exception:
                    logger.exception("Falha ao processar pedido manual de CLEAR HALT")

            if self._consume_request(DECISION_NOW_REQUEST):
                try:
                    logger.warning("DECISION NOW solicitado | executando ciclo com último candle diário fechado")
                    self.daily_cycle()
                except Exception:
                    logger.exception("Falha no ciclo manual DECISION NOW")

            if now.hour == 0 and now.minute >= 5 and last_daily_run != now.date():
                try:
                    self.daily_cycle()
                except Exception:
                    logger.exception("Falha no ciclo Aggressive C")
                last_daily_run = now.date()

            try:
                self._reconcile_exchange_state()
                self._portfolio_risk_update()
                self._save_state()
            except Exception:
                logger.exception("Falha de reconciliação; novas operações ficam inseguras")
                self.state["portfolio_halted"] = True
                self._save_state()
            time.sleep(30)


if __name__ == "__main__":
    LiveAggressiveCTradeBot().run()
