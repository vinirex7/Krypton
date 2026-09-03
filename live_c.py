"""Entrypoint do Krypton Aggressive C live.

TESTNET continua sendo o padrão. Para produção, configure USE_TESTNET=false
explicitamente no .env do VPS antes de iniciar este arquivo.

O runtime aceita dois pedidos administrativos por arquivos locais, consumidos
pela própria instância já em execução. Isso evita iniciar uma segunda instância
concorrente apenas para forçar uma decisão ou liberar um halt contábil.

Client-facing telemetry is best-effort and isolated from execution: dashboard
failures are logged but never allowed to block Binance decisions or orders.
"""
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from aggressive_c_live import AggressiveCTradeBot
from config import (
    AGGRESSIVE_C_ALPHA_SYMBOLS,
    AGGRESSIVE_C_MAX_DRAWDOWN_PCT,
    LIVE_QUOTE_ASSET,
    LOG_FILE,
    TRADING_PAIRS,
    USE_TESTNET,
)
from telemetry import TelemetryStore

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
    """Runtime live com auto-reparo OCO, controles administrativos e telemetria."""

    def __init__(self):
        super().__init__()
        self.dashboard_client_id = os.getenv("KRYPTON_CLIENT_ID", "default")
        self.telemetry = TelemetryStore(os.getenv("KRYPTON_DASHBOARD_DB", "krypton_dashboard.db"))
        self._last_telemetry_snapshot = 0.0
        self._record_snapshot_safe()

    def _telemetry(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.exception("Falha na telemetria do dashboard; execução do bot continua")
            return None

    def _record_snapshot_safe(self):
        try:
            equity, dd = self._portfolio_risk_update()
            self.telemetry.record_snapshot(
                self.dashboard_client_id,
                equity=equity,
                drawdown=dd,
                tactical_equity=self._tactical_equity(),
                alpha_equity=self._alpha_equity(),
                halted=bool(self.state.get("portfolio_halted")),
                mode="TESTNET" if USE_TESTNET else "PRODUÇÃO",
            )
            self._last_telemetry_snapshot = time.monotonic()
        except Exception:
            logger.exception("Falha ao registrar snapshot do dashboard")

    def _execute_order(self, symbol: str, side: str, quantity: float, sleeve: str, *, reference_price=None, market=False):
        pre_tactical = dict(self.state.get("tactical_positions", {}).get(symbol, {}))
        result = super()._execute_order(
            symbol, side, quantity, sleeve, reference_price=reference_price, market=market
        )
        if not result:
            return result
        fill = result.get("order", {})
        executed = float(fill.get("executedQty", abs(float(result.get("base_delta", 0.0)))) or 0.0)
        quote = float(fill.get("cummulativeQuoteQty", 0.0) or 0.0)
        fill_price = quote / executed if executed > 0 and quote > 0 else None
        self._telemetry(
            self.telemetry.record_order,
            self.dashboard_client_id,
            symbol=symbol,
            side=side.upper(),
            sleeve=sleeve,
            quantity=executed or float(quantity),
            fill_price=fill_price,
            status=fill.get("status"),
            exchange_order_id=str(fill.get("orderId")) if fill.get("orderId") is not None else None,
            metadata={"market": bool(market)},
        )
        if sleeve == "tactical" and side.upper() == "SELL" and pre_tactical:
            entry = float(pre_tactical.get("entry_price", 0.0) or 0.0)
            sold = max(0.0, -float(result.get("base_delta", 0.0)))
            if sold > 0 and fill_price is not None:
                pnl = (fill_price - entry) * sold if entry > 0 else None
                ret = (fill_price / entry - 1.0) if entry > 0 else None
                self._telemetry(
                    self.telemetry.record_trade,
                    self.dashboard_client_id,
                    symbol=symbol,
                    sleeve=sleeve,
                    quantity=sold,
                    entry_price=entry or None,
                    exit_price=fill_price,
                    pnl=pnl,
                    return_pct=ret,
                    reason="Bot tactical exit",
                )
        return result

    def daily_cycle(self):
        before_tactical = {s: float(p.get("quantity", 0.0)) for s, p in self.state.get("tactical_positions", {}).items()}
        before_alpha = {s: float(self.state.get("alpha_qty", {}).get(s, 0.0)) for s in AGGRESSIVE_C_ALPHA_SYMBOLS}
        super().daily_cycle()
        signal_time = datetime.now(timezone.utc).isoformat()
        after_tactical = {s: float(p.get("quantity", 0.0)) for s, p in self.state.get("tactical_positions", {}).items()}
        after_alpha = {s: float(self.state.get("alpha_qty", {}).get(s, 0.0)) for s in AGGRESSIVE_C_ALPHA_SYMBOLS}

        for symbol in TRADING_PAIRS:
            before, after = before_tactical.get(symbol, 0.0), after_tactical.get(symbol, 0.0)
            if after > before + 1e-12:
                decision, reason = "BUY", "Nova exposição tática aberta após o ciclo de decisão."
            elif after + 1e-12 < before:
                decision, reason = "SELL", "Exposição tática reduzida ou encerrada pelo ciclo."
            elif after > 0:
                decision, reason = "HOLD", "Posição tática mantida; nenhuma alteração executada neste ciclo."
            else:
                decision, reason = "WAIT", "Nenhuma posição tática foi aberta neste ciclo."
            self._telemetry(
                self.telemetry.record_decision,
                self.dashboard_client_id,
                symbol=symbol,
                sleeve="tactical",
                decision=decision,
                public_reason=reason,
                signal_time=signal_time,
            )

        for symbol in AGGRESSIVE_C_ALPHA_SYMBOLS:
            before, after = before_alpha.get(symbol, 0.0), after_alpha.get(symbol, 0.0)
            if after > before + 1e-12:
                decision, reason = "BUY", "Peso do sleeve alpha aumentado no rebalanceamento."
            elif after + 1e-12 < before:
                decision, reason = "SELL", "Peso do sleeve alpha reduzido no rebalanceamento."
            elif after > 0:
                decision, reason = "HOLD", "Alocação alpha mantida."
            else:
                decision, reason = "WAIT", "Ativo não recebeu alocação alpha neste ciclo."
            self._telemetry(
                self.telemetry.record_decision,
                self.dashboard_client_id,
                symbol=symbol,
                sleeve="alpha",
                decision=decision,
                public_reason=reason,
                signal_time=signal_time,
            )
        self._record_snapshot_safe()

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
        self._record_snapshot_safe()
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
                if time.monotonic() - self._last_telemetry_snapshot >= 60:
                    self._record_snapshot_safe()
            except Exception:
                logger.exception("Falha de reconciliação; novas operações ficam inseguras")
                self.state["portfolio_halted"] = True
                self._save_state()
            time.sleep(30)


if __name__ == "__main__":
    LiveAggressiveCTradeBot().run()
