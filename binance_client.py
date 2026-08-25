# binance_client.py — Interface Spot segura para o Krypton

import logging
import math
import time

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    ENTRY_FILL_TIMEOUT_SEC,
    ORDER_POLL_SEC,
    SLIPPAGE_LIMIT_PCT,
    USE_TESTNET,
)

logger = logging.getLogger("Krypton.Binance")
MAX_RETRIES = 3


class BinanceInterface:
    def __init__(self):
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=USE_TESTNET)
        self.client.ping()
        logger.info("Binance API conectada | Modo: %s", "TESTNET" if USE_TESTNET else "PRODUÇÃO")

    def get_ohlcv(self, symbol: str, interval: str = "1d", limit: int = 300) -> pd.DataFrame:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
                df = pd.DataFrame(
                    klines,
                    columns=[
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
                    ],
                )
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                df.set_index("open_time", inplace=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                return df[["open", "high", "low", "close", "volume"]]
            except BinanceAPIException as exc:
                logger.warning("get_ohlcv tentativa %s/%s: %s", attempt, MAX_RETRIES, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(5)
        raise RuntimeError(f"Falha ao buscar OHLCV para {symbol}.")

    def get_symbol_info(self, symbol: str) -> dict:
        info = self.client.get_symbol_info(symbol)
        if not info:
            raise RuntimeError(f"Símbolo inexistente/indisponível: {symbol}")
        filters = {f["filterType"]: f for f in info["filters"]}
        lot = filters.get("LOT_SIZE")
        price_filter = filters.get("PRICE_FILTER")
        # Binance usa MIN_NOTIONAL em alguns pares e NOTIONAL em outros (ex.: SOLU).
        notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL")
        if not lot or not price_filter or not notional_filter:
            raise RuntimeError(f"Filtros incompletos para {symbol}")
        return {
            "step_size": float(lot["stepSize"]),
            "tick_size": float(price_filter["tickSize"]),
            "min_notional": float(notional_filter["minNotional"]),
        }

    def get_account_balance(self, asset: str = "U") -> float:
        balances = self.client.get_account()["balances"]
        for balance in balances:
            if balance["asset"] == asset:
                return float(balance["free"])
        return 0.0

    def get_asset_total(self, asset: str) -> float:
        balances = self.client.get_account()["balances"]
        for balance in balances:
            if balance["asset"] == asset:
                return float(balance["free"]) + float(balance["locked"])
        return 0.0

    def get_current_price(self, symbol: str) -> float:
        ticker = self.client.get_orderbook_ticker(symbol=symbol)
        return (float(ticker["bidPrice"]) + float(ticker["askPrice"])) / 2.0

    @staticmethod
    def _step_precision(step: float) -> int:
        text = f"{step:.16f}".rstrip("0")
        return len(text.split(".")[-1]) if "." in text else 0

    def _round_step(self, qty: float, step: float) -> float:
        if step <= 0:
            return qty
        precision = self._step_precision(step)
        return round(math.floor(qty / step) * step, precision)

    def _ceil_step(self, qty: float, step: float) -> float:
        if step <= 0:
            return qty
        precision = self._step_precision(step)
        return round(math.ceil(qty / step) * step, precision)

    def _round_tick(self, price: float, tick: float) -> float:
        if tick <= 0:
            return price
        precision = self._step_precision(tick)
        return round(round(price / tick) * tick, precision)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        symbol_info: dict,
        reference_price: float | None = None,
    ) -> dict | None:
        """Envia LIMIT e valida slippage contra o preço do sinal, não contra um mid recém-buscado."""
        side = side.upper()
        if reference_price is not None and reference_price > 0:
            deviation = abs(price - reference_price) / reference_price
            if deviation > SLIPPAGE_LIMIT_PCT:
                logger.warning(
                    "Ordem %s rejeitada | %s | desvio %.2f%% > %.2f%% do preço do sinal",
                    side, symbol, deviation * 100, SLIPPAGE_LIMIT_PCT * 100,
                )
                return None

        qty = self._round_step(quantity, symbol_info["step_size"])
        rounded_price = self._round_tick(price, symbol_info["tick_size"])
        notional = qty * rounded_price
        min_notional = symbol_info["min_notional"]
        if qty <= 0 or notional < min_notional:
            logger.warning(
                "Ordem %s rejeitada | %s | notional %.8f < mínimo %.8f",
                side, symbol, notional, min_notional,
            )
            return None

        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_LIMIT,
                timeInForce=Client.TIME_IN_FORCE_GTC,
                quantity=qty,
                price=f"{rounded_price:.8f}",
            )
            logger.info("LIMIT aceita pela Binance | %s | order=%s", symbol, order["orderId"])
            return order
        except BinanceAPIException as exc:
            logger.error("Erro ao enviar LIMIT %s %s: %s", side, symbol, exc)
            return None

    def wait_for_fill(self, symbol: str, order_id: int, timeout_sec: int = ENTRY_FILL_TIMEOUT_SEC) -> dict | None:
        """Só considera posição criada depois de status FILLED; timeout cancela a ordem."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                order = self.client.get_order(symbol=symbol, orderId=order_id)
                status = order.get("status")
                if status == "FILLED":
                    return order
                if status in {"CANCELED", "REJECTED", "EXPIRED"}:
                    return None
            except BinanceAPIException as exc:
                logger.warning("get_order %s/%s falhou: %s", symbol, order_id, exc)
            time.sleep(ORDER_POLL_SEC)

        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            if order.get("status") != "FILLED":
                self.cancel_order(symbol, order_id)
                return None
            return order
        except BinanceAPIException as exc:
            logger.error("Falha final ao reconciliar ordem %s/%s: %s", symbol, order_id, exc)
            return None

    def get_order(self, symbol: str, order_id: int) -> dict:
        return self.client.get_order(symbol=symbol, orderId=order_id)

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        try:
            self.client.cancel_order(symbol=symbol, orderId=order_id)
            return True
        except BinanceAPIException as exc:
            logger.error("Erro ao cancelar ordem %s/%s: %s", symbol, order_id, exc)
            return False

    def get_open_orders(self, symbol: str) -> list:
        return self.client.get_open_orders(symbol=symbol)

    def create_oco_order(self, symbol: str, quantity: float, take_profit_price: float, stop_price: float, symbol_info: dict) -> dict | None:
        """Cria OCO SELL real para proteger uma posição LONG spot."""
        qty = self._round_step(quantity, symbol_info["step_size"])
        tp = self._round_tick(take_profit_price, symbol_info["tick_size"])
        stop = self._round_tick(stop_price, symbol_info["tick_size"])
        stop_limit = self._round_tick(stop * (1.0 - min(SLIPPAGE_LIMIT_PCT / 2.0, 0.0025)), symbol_info["tick_size"])
        try:
            order = self.client.create_oco_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                quantity=qty,
                price=f"{tp:.8f}",
                stopPrice=f"{stop:.8f}",
                stopLimitPrice=f"{stop_limit:.8f}",
                stopLimitTimeInForce=Client.TIME_IN_FORCE_GTC,
            )
            logger.info("OCO criada | %s | qty %.8f | TP %.8f | SL %.8f", symbol, qty, tp, stop)
            return order
        except BinanceAPIException as exc:
            logger.critical("FALHA OCO | %s | TP %.8f | SL %.8f | %s", symbol, tp, stop, exc)
            return None

    def cancel_oco_order(self, symbol: str, order_list_id: int) -> bool:
        try:
            self.client.cancel_order_list(symbol=symbol, orderListId=order_list_id)
            return True
        except BinanceAPIException as exc:
            logger.error("Erro ao cancelar OCO %s/%s: %s", symbol, order_list_id, exc)
            return False
