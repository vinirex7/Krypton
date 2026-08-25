# risk_manager.py — Gestão de risco e sizing do Krypton

import logging
from config import (
    CIRCUIT_BREAKER_PCT,
    MAX_DRAWDOWN_PCT,
    RISK_PER_TRADE,
    STOP_LOSS_ATR_MULT,
    TAKE_PROFIT_ATR_MULT,
)

logger = logging.getLogger("Krypton.Risk")


class RiskManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_cap = initial_capital
        self.daily_date = None
        self.halted = False
        self.circuit_breaker = False

    def reset_daily(self, current_capital: float, current_date) -> None:
        """Reset idempotente: só troca a base quando o dia UTC mudou."""
        if self.daily_date != current_date:
            self.daily_date = current_date
            self.daily_start_cap = current_capital
            self.circuit_breaker = False
            logger.info("Reset diário UTC | Capital: $%.2f", current_capital)

    def update_peak(self, current_capital: float) -> None:
        self.peak_capital = max(self.peak_capital, current_capital)

    def check_circuit_breaker(self, current_capital: float) -> bool:
        if self.daily_start_cap <= 0:
            return self.circuit_breaker
        daily_loss = (self.daily_start_cap - current_capital) / self.daily_start_cap
        if daily_loss >= CIRCUIT_BREAKER_PCT:
            self.circuit_breaker = True
            logger.warning(
                "CIRCUIT BREAKER | perda diária %.2f%% >= %.2f%%",
                daily_loss * 100,
                CIRCUIT_BREAKER_PCT * 100,
            )
        return self.circuit_breaker

    def check_max_drawdown(self, current_capital: float) -> bool:
        if self.peak_capital <= 0:
            return self.halted
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        if drawdown >= MAX_DRAWDOWN_PCT:
            self.halted = True
            logger.critical(
                "MAX DRAWDOWN | %.2f%% >= %.2f%% | HALT",
                drawdown * 100,
                MAX_DRAWDOWN_PCT * 100,
            )
        return self.halted

    def can_trade(self, current_capital: float) -> bool:
        self.update_peak(current_capital)
        return not (
            self.check_max_drawdown(current_capital)
            or self.check_circuit_breaker(current_capital)
        )

    def calculate_position_size(self, capital: float, entry_price: float, atr: float) -> dict:
        """1% do capital TOTAL; nunca permite notional maior que o capital disponível."""
        sl_distance = atr * STOP_LOSS_ATR_MULT
        tp_distance = atr * TAKE_PROFIT_ATR_MULT
        risk_amount = capital * RISK_PER_TRADE
        quantity = risk_amount / sl_distance if sl_distance > 0 else 0.0

        # Cap explícito de notional: Spot não pode comprar acima do caixa disponível.
        max_quantity = capital / entry_price if entry_price > 0 else 0.0
        quantity = min(quantity, max_quantity)

        return {
            "quantity": quantity,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "stop_loss_long": entry_price - sl_distance,
            "take_profit_long": entry_price + tp_distance,
            "risk_amount_usd": risk_amount,
            "rr_ratio": tp_distance / sl_distance if sl_distance > 0 else 0.0,
            "notional": quantity * entry_price,
        }

    def status(self, current_capital: float) -> dict:
        dd = (
            (self.peak_capital - current_capital) / self.peak_capital
            if self.peak_capital > 0 else 0.0
        )
        daily_loss = (
            (self.daily_start_cap - current_capital) / self.daily_start_cap
            if self.daily_start_cap > 0 else 0.0
        )
        return {
            "current_capital": round(current_capital, 2),
            "peak_capital": round(self.peak_capital, 2),
            "current_drawdown": f"{dd:.2%}",
            "daily_loss": f"{daily_loss:.2%}",
            "circuit_breaker": self.circuit_breaker,
            "halted": self.halted,
            "can_trade": not (self.halted or self.circuit_breaker),
        }
