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
    def __init__(
        self,
        initial_capital: float,
        *,
        risk_per_trade: float = RISK_PER_TRADE,
        max_drawdown_pct: float = MAX_DRAWDOWN_PCT,
        circuit_breaker_pct: float = CIRCUIT_BREAKER_PCT,
    ):
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_cap = initial_capital
        self.daily_date = None
        self.halted = False
        self.circuit_breaker = False
        self.risk_per_trade = float(risk_per_trade)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.circuit_breaker_pct = float(circuit_breaker_pct)
        if not 0 < self.risk_per_trade < 1:
            raise ValueError("risk_per_trade deve estar em (0,1)")
        if not 0 < self.max_drawdown_pct < 1:
            raise ValueError("max_drawdown_pct deve estar em (0,1)")
        if not 0 < self.circuit_breaker_pct < 1:
            raise ValueError("circuit_breaker_pct deve estar em (0,1)")

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
        if daily_loss >= self.circuit_breaker_pct:
            self.circuit_breaker = True
            logger.warning(
                "CIRCUIT BREAKER | perda diária %.2f%% >= %.2f%%",
                daily_loss * 100,
                self.circuit_breaker_pct * 100,
            )
        return self.circuit_breaker

    def check_max_drawdown(self, current_capital: float) -> bool:
        if self.peak_capital <= 0:
            return self.halted
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        if drawdown >= self.max_drawdown_pct:
            self.halted = True
            logger.critical(
                "MAX DRAWDOWN | %.2f%% >= %.2f%% | HALT",
                drawdown * 100,
                self.max_drawdown_pct * 100,
            )
        return self.halted

    def can_trade(self, current_capital: float) -> bool:
        self.update_peak(current_capital)
        return not (
            self.check_max_drawdown(current_capital)
            or self.check_circuit_breaker(current_capital)
        )

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        atr: float,
        allocation_pct: float = 1.0,
        available_cash: float | None = None,
    ) -> dict:
        """Arrisca sobre a equity, limitado pelo peso do ativo e caixa livre."""
        sl_distance = atr * STOP_LOSS_ATR_MULT
        tp_distance = atr * TAKE_PROFIT_ATR_MULT
        risk_amount = capital * self.risk_per_trade
        quantity = risk_amount / sl_distance if sl_distance > 0 else 0.0

        allocation_cap = capital * max(min(allocation_pct, 1.0), 0.0)
        cash_cap = capital if available_cash is None else max(available_cash, 0.0)
        max_notional = min(allocation_cap, cash_cap)
        max_quantity = max_notional / entry_price if entry_price > 0 else 0.0
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

    def snapshot(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "peak_capital": self.peak_capital,
            "daily_start_cap": self.daily_start_cap,
            "daily_date": self.daily_date.isoformat() if self.daily_date else None,
            "halted": self.halted,
            "circuit_breaker": self.circuit_breaker,
            "risk_per_trade": self.risk_per_trade,
            "max_drawdown_pct": self.max_drawdown_pct,
            "circuit_breaker_pct": self.circuit_breaker_pct,
        }

    def restore(self, state: dict) -> None:
        from datetime import date

        saved_risk = float(state.get("risk_per_trade", self.risk_per_trade))
        saved_dd = float(state.get("max_drawdown_pct", self.max_drawdown_pct))
        saved_cb = float(state.get("circuit_breaker_pct", self.circuit_breaker_pct))
        if abs(saved_risk - self.risk_per_trade) > 1e-12:
            raise RuntimeError("Estado de risco incompatível: risk_per_trade mudou")
        if abs(saved_dd - self.max_drawdown_pct) > 1e-12:
            raise RuntimeError("Estado de risco incompatível: max_drawdown_pct mudou")
        if abs(saved_cb - self.circuit_breaker_pct) > 1e-12:
            raise RuntimeError("Estado de risco incompatível: circuit_breaker_pct mudou")

        self.initial_capital = float(state.get("initial_capital", self.initial_capital))
        self.peak_capital = max(float(state.get("peak_capital", self.peak_capital)), self.peak_capital)
        self.daily_start_cap = float(state.get("daily_start_cap", self.daily_start_cap))
        raw_date = state.get("daily_date")
        self.daily_date = date.fromisoformat(raw_date) if raw_date else None
        self.halted = bool(state.get("halted", False))
        self.circuit_breaker = bool(state.get("circuit_breaker", False))

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
            "risk_per_trade": self.risk_per_trade,
            "max_drawdown_pct": self.max_drawdown_pct,
        }
