# tradebot.py — Krypton Spot TradeBot
# Live quote asset: U. Backtest quote asset: USDT.
# O mercado live é SPOT; SHORT foi removido. Sinal -1 significa EXIT.

import logging
import sqlite3
import time
from datetime import datetime, timezone

from binance_client import BinanceInterface
from config import (
    ENTRY_FILL_TIMEOUT_SEC,
    LIVE_QUOTE_ASSET,
    LOG_FILE,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MAX_SIMULTANEOUS_POS,
    RSI_HIGH,
    RSI_LOW,
    RSI_PERIOD,
    STATE_DB_FILE,
    SUPERTREND_MULTIPLIER,
    SUPERTREND_PERIOD,
    TAKE_PROFIT_ATR_MULT,
    TIMEFRAME,
    TRADING_PAIRS,
    USE_TESTNET,
)
from indicators import compute_atr, compute_signals
from risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("Krypton.Main")


class TradeBot:
    def __init__(self):
        self.binance = BinanceInterface()
        self.positions = {}
        self.symbol_infos = {}
        self._init_state_db()
        self._initialize()

    def _init_state_db(self):
        self.db = sqlite3.connect(STATE_DB_FILE)
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                order_list_id INTEGER,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.db.commit()

    def _save_position(self, symbol, pos):
        self.db.execute(
            """
            INSERT INTO positions(symbol, side, entry_price, quantity, stop_loss,
                                   take_profit, order_list_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                side=excluded.side,
                entry_price=excluded.entry_price,
                quantity=excluded.quantity,
                stop_loss=excluded.stop_loss,
                take_profit=excluded.take_profit,
                order_list_id=excluded.order_list_id,
                updated_at=excluded.updated_at
            """,
            (
                symbol, pos["side"], pos["entry_price"], pos["quantity"],
                pos["stop_loss"], pos["take_profit"], pos.get("order_list_id"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.db.commit()

    def _delete_position(self, symbol):
        self.positions.pop(symbol, None)
        self.db.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
        self.db.commit()

    def _load_positions(self):
        rows = self.db.execute(
            "SELECT symbol, side, entry_price, quantity, stop_loss, take_profit, order_list_id FROM positions"
        ).fetchall()
        for row in rows:
            symbol, side, entry, qty, sl, tp, order_list_id = row
            self.positions[symbol] = {
                "side": side,
                "entry_price": entry,
                "quantity": qty,
                "stop_loss": sl,
                "take_profit": tp,
                "order_list_id": order_list_id,
            }

    def _initialize(self):
        quote_balance = self.binance.get_account_balance(LIVE_QUOTE_ASSET)
        for symbol in TRADING_PAIRS:
            self.symbol_infos[symbol] = self.binance.get_symbol_info(symbol)
        self._load_positions()
        self._reconcile_positions()
        self.risk_manager = RiskManager(initial_capital=self._get_current_capital())
        logger.info(
            "Krypton inicializado | Capital: %.2f %s | Pares: %s | Modo: %s",
            quote_balance, LIVE_QUOTE_ASSET, list(TRADING_PAIRS),
            "TESTNET" if USE_TESTNET else "PRODUÇÃO",
        )

    @staticmethod
    def _base_asset(symbol):
        if not symbol.endswith(LIVE_QUOTE_ASSET):
            raise ValueError(f"Símbolo {symbol} não termina em {LIVE_QUOTE_ASSET}")
        return symbol[: -len(LIVE_QUOTE_ASSET)]

    def _get_current_capital(self):
        capital = self.binance.get_account_balance(LIVE_QUOTE_ASSET)
        for symbol, pos in self.positions.items():
            capital += pos["quantity"] * self.binance.get_current_price(symbol)
        return capital

    def _protect_position(self, symbol):
        pos = self.positions.get(symbol)
        if not pos:
            return False
        order = self.binance.create_oco_order(
            symbol=symbol,
            quantity=pos["quantity"],
            take_profit_price=pos["take_profit"],
            stop_price=pos["stop_loss"],
            symbol_info=self.symbol_infos[symbol],
        )
        if not order:
            logger.critical("POSIÇÃO SEM OCO — não abrir novas posições | %s", symbol)
            return False
        pos["order_list_id"] = order.get("orderListId")
        self._save_position(symbol, pos)
        return True

    def _reconcile_positions(self):
        """Reconcilia RAM/SQLite com saldo real e ordens abertas no boot."""
        for symbol in list(self.positions):
            pos = self.positions[symbol]
            base_total = self.binance.get_asset_total(self._base_asset(symbol))
            if base_total <= 0:
                logger.warning("Posição fantasma removida | %s | saldo real zerado", symbol)
                self._delete_position(symbol)
                continue

            pos["quantity"] = base_total
            open_orders = self.binance.get_open_orders(symbol)
            if not open_orders:
                self._protect_position(symbol)
            self._save_position(symbol, pos)

        # Se existe ativo sem estado, proteger em vez de fingir que não existe.
        for symbol in TRADING_PAIRS:
            if symbol in self.positions:
                continue
            base_total = self.binance.get_asset_total(self._base_asset(symbol))
            price = self.binance.get_current_price(symbol)
            info = self.symbol_infos[symbol]
            if base_total * price < info["min_notional"]:
                continue

            logger.critical("POSIÇÃO ÓRFÃ detectada | %s | qty %.8f", symbol, base_total)
            df = self.binance.get_ohlcv(symbol, interval=TIMEFRAME, limit=300)
            atr = float(compute_atr(df["high"], df["low"], df["close"]).iloc[-1])
            if atr <= 0:
                continue
            self.positions[symbol] = {
                "side": "LONG",
                "entry_price": price,
                "quantity": base_total,
                "stop_loss": price - 2.0 * atr,
                "take_profit": price + TAKE_PROFIT_ATR_MULT * atr,
                "order_list_id": None,
            }
            self._save_position(symbol, self.positions[symbol])
            self._protect_position(symbol)

    def _open_position(self, symbol, atr, signal_price):
        if symbol in self.positions or len(self.positions) >= MAX_SIMULTANEOUS_POS:
            return
        capital = self._get_current_capital()
        if not self.risk_manager.can_trade(capital):
            return

        # 1% do capital TOTAL; TRADING_PAIRS allocation não reduz o risco por trade.
        mid = self.binance.get_current_price(symbol)
        sizing = self.risk_manager.calculate_position_size(capital, mid, atr)
        if sizing["quantity"] <= 0:
            return

        limit_price = mid * 1.0005
        order = self.binance.place_limit_order(
            symbol=symbol,
            side="BUY",
            quantity=sizing["quantity"],
            price=limit_price,
            symbol_info=self.symbol_infos[symbol],
            reference_price=signal_price,
        )
        if not order:
            return

        # Aceite da LIMIT não é fill: só persistimos posição após FILLED.
        filled = self.binance.wait_for_fill(symbol, order["orderId"], ENTRY_FILL_TIMEOUT_SEC)
        if not filled:
            logger.info("Entrada cancelada por timeout sem fill | %s | order=%s", symbol, order["orderId"])
            return

        executed_qty = float(filled.get("executedQty", 0))
        quote_qty = float(filled.get("cummulativeQuoteQty", 0))
        if executed_qty <= 0:
            logger.error("Fill sem executedQty | %s | order=%s", symbol, order["orderId"])
            return
        entry_price = quote_qty / executed_qty if quote_qty > 0 else float(filled["price"])
        sl = entry_price - sizing["sl_distance"]
        tp = entry_price + sizing["tp_distance"]

        self.positions[symbol] = {
            "side": "LONG",
            "entry_price": entry_price,
            "quantity": executed_qty,
            "stop_loss": sl,
            "take_profit": tp,
            "order_list_id": None,
        }
        self._save_position(symbol, self.positions[symbol])

        if not self._protect_position(symbol):
            logger.critical("Entrada FILLED mas OCO falhou | %s | reconciliação obrigatória", symbol)

        logger.info(
            "LONG FILLED | %s | qty %.8f | entry %.8f | SL %.8f | TP %.8f | RR %.2f",
            symbol, executed_qty, entry_price, sl, tp, sizing["rr_ratio"],
        )

    def _close_position(self, symbol, reason="Signal"):
        pos = self.positions.get(symbol)
        if not pos:
            return

        if pos.get("order_list_id") is not None:
            self.binance.cancel_oco_order(symbol, pos["order_list_id"])
            pos["order_list_id"] = None
            self._save_position(symbol, pos)

        price = self.binance.get_current_price(symbol)
        order = self.binance.place_limit_order(
            symbol=symbol,
            side="SELL",
            quantity=pos["quantity"],
            price=price * 0.9995,
            symbol_info=self.symbol_infos[symbol],
            reference_price=price,
        )
        if not order:
            self._protect_position(symbol)
            return

        filled = self.binance.wait_for_fill(symbol, order["orderId"])
        if not filled:
            logger.warning("Saída não executada; restaurando OCO | %s", symbol)
            self._protect_position(symbol)
            return

        logger.info("Posição encerrada | %s | razão=%s", symbol, reason)
        self._delete_position(symbol)

    def _check_orders_and_reconcile(self):
        """Exchange/OCO é a proteção; RAM/SQLite serve para estado e reconciliação."""
        for symbol in list(self.positions):
            pos = self.positions[symbol]
            base_total = self.binance.get_asset_total(self._base_asset(symbol))
            if base_total <= 0:
                logger.info("OCO executado/posição zerada | %s", symbol)
                self._delete_position(symbol)
                continue

            step = self.symbol_infos[symbol]["step_size"]
            if abs(base_total - pos["quantity"]) > step:
                pos["quantity"] = base_total
                self._save_position(symbol, pos)

            if not self.binance.get_open_orders(symbol):
                self._protect_position(symbol)

    def daily_cycle(self):
        now = datetime.now(timezone.utc)
        logger.info("=" * 70)
        logger.info("CICLO DIÁRIO UTC | %s", now.strftime("%Y-%m-%d %H:%M UTC"))

        # Reset só no início de um novo dia UTC; não zera a perda antes de can_trade.
        current_capital = self._get_current_capital()
        self.risk_manager.reset_daily(current_capital, now.date())
        self._check_orders_and_reconcile()
        current_capital = self._get_current_capital()

        if not self.risk_manager.can_trade(current_capital):
            logger.warning("Trading pausado pelos controles de risco.")
            return

        for symbol in TRADING_PAIRS:
            try:
                df = self.binance.get_ohlcv(symbol, interval=TIMEFRAME, limit=300)
                signals = compute_signals(
                    df,
                    st_period=SUPERTREND_PERIOD,
                    st_mult=SUPERTREND_MULTIPLIER,
                    rsi_period=RSI_PERIOD,
                    rsi_low=RSI_LOW,
                    rsi_high=RSI_HIGH,
                    macd_fast=MACD_FAST,
                    macd_slow=MACD_SLOW,
                    macd_sig=MACD_SIGNAL,
                )
                atr = float(compute_atr(df["high"], df["low"], df["close"]).iloc[-1])
                signal = int(signals.iloc[-1])
                signal_price = float(df["close"].iloc[-1])

                logger.info(
                    "%s | sinal=%s | close sinal=%.8f | ATR=%.8f",
                    symbol,
                    "LONG" if signal == 1 else "EXIT/FLAT",
                    signal_price,
                    atr,
                )

                # Spot: -1 nunca abre short; qualquer sinal diferente de LONG fecha.
                if symbol in self.positions and signal != 1:
                    self._close_position(symbol, "Signal reversal/flat")
                elif symbol not in self.positions and signal == 1:
                    self._open_position(symbol, atr, signal_price)

            except Exception as exc:
                logger.error("Erro no ciclo %s: %s", symbol, exc, exc_info=True)

        final_capital = self._get_current_capital()
        self.risk_manager.update_peak(final_capital)
        self.risk_manager.check_circuit_breaker(final_capital)
        self.risk_manager.check_max_drawdown(final_capital)
        logger.info("Risco: %s", self.risk_manager.status(final_capital))

    def run(self):
        logger.info("Krypton iniciado | Live quote=%s | UTC", LIVE_QUOTE_ASSET)
        self.daily_cycle()
        last_daily_run = datetime.now(timezone.utc).date()
        logger.info("Próximo ciclo diário: 00:05 UTC")

        # Não usamos schedule.every().day.at(): schedule 1.2.1 usa timezone local.
        # O relógio é comparado explicitamente em UTC, independentemente do VPS.
        while True:
            now = datetime.now(timezone.utc)
            if now.hour == 0 and now.minute == 5 and last_daily_run != now.date():
                self.daily_cycle()
                last_daily_run = now.date()
            self._check_orders_and_reconcile()
            time.sleep(30)


if __name__ == "__main__":
    TradeBot().run()
