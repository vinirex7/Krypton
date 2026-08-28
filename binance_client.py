# binance_client.py — Interface Spot segura para o Krypton

import logging
import math
import time
from datetime import datetime, timezone

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

from config import (
    AGGRESSIVE_C_MARGIN_TRANSFER_BUFFER_PCT,
    AGGRESSIVE_C_USE_MARGIN_CAPITAL_POOL,
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    ENTRY_FILL_TIMEOUT_SEC,
    FEE_RATE,
    LIVE_QUOTE_ASSET,
    LIVE_STRATEGY,
    ORDER_POLL_SEC,
    SLIPPAGE_LIMIT_PCT,
    USE_TESTNET,
)

logger = logging.getLogger("Krypton.Binance")
MAX_RETRIES = 3
MARGIN_EPS = 1e-12


class BinanceInterface:
    def __init__(self):
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=USE_TESTNET)
        self.client.ping()
        self.margin_pool_enabled = bool(
            LIVE_STRATEGY == "AGGRESSIVE_C" and AGGRESSIVE_C_USE_MARGIN_CAPITAL_POOL
        )
        if self.margin_pool_enabled and USE_TESTNET:
            raise RuntimeError(
                "AGGRESSIVE_C_USE_MARGIN_CAPITAL_POOL exige Binance PRODUÇÃO; "
                "Spot Testnet não oferece o fluxo Cross Margin usado pelo pool."
            )
        logger.info("Binance API conectada | Modo: %s", "TESTNET" if USE_TESTNET else "PRODUÇÃO")
        if self.margin_pool_enabled:
            snap = self.get_margin_capital_snapshot(LIVE_QUOTE_ASSET)
            logger.info(
                "Pool Spot+Margin habilitado | %s margin próprio transferível=%.8f | "
                "borrowed=%.8f interest=%.8f",
                LIVE_QUOTE_ASSET,
                snap["available_own"],
                snap["borrowed"],
                snap["interest"],
            )

    def get_ohlcv(self, symbol: str, interval: str = "1d", limit: int = 300, closed_only: bool = True) -> pd.DataFrame:
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
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
                df.set_index("open_time", inplace=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                if closed_only:
                    now = pd.Timestamp(datetime.now(timezone.utc))
                    df = df.loc[df["close_time"] <= now]
                return df[["open", "high", "low", "close", "volume", "close_time"]]
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
        # Binance usa MIN_NOTIONAL em alguns pares e NOTIONAL em outros.
        notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL")
        if not lot or not price_filter or not notional_filter:
            raise RuntimeError(f"Filtros incompletos para {symbol}")
        return {
            "step_size": float(lot["stepSize"]),
            "tick_size": float(price_filter["tickSize"]),
            "min_notional": float(notional_filter["minNotional"]),
            "status": info.get("status"),
            "quote_asset": info.get("quoteAsset"),
            "is_spot_trading_allowed": bool(info.get("isSpotTradingAllowed", False)),
        }

    def _get_spot_free_balance(self, asset: str) -> float:
        balances = self.client.get_account()["balances"]
        for balance in balances:
            if balance["asset"] == asset:
                return float(balance["free"])
        return 0.0

    def get_margin_capital_snapshot(self, asset: str = "USDT") -> dict:
        """Cross Margin próprio e transferível, nunca crédito/borrow.

        Fail-closed: se existir qualquer liability na conta Cross Margin, o
        Krypton considera zero capital de Margin. Isso evita remover colateral
        de uma posição alavancada criada fora do bot.
        """
        if not self.margin_pool_enabled:
            return {
                "free": 0.0,
                "locked": 0.0,
                "borrowed": 0.0,
                "interest": 0.0,
                "net_asset": 0.0,
                "max_transferable": 0.0,
                "available_own": 0.0,
                "has_any_liability": False,
            }

        try:
            account = self.client.get_margin_account()
            total_liability_btc = float(account.get("totalLiabilityOfBtc", 0.0) or 0.0)
            item = next(
                (x for x in account.get("userAssets", []) if x.get("asset") == asset),
                None,
            )
            if item is None:
                free = locked = borrowed = interest = net_asset = 0.0
            else:
                free = float(item.get("free", 0.0) or 0.0)
                locked = float(item.get("locked", 0.0) or 0.0)
                borrowed = float(item.get("borrowed", 0.0) or 0.0)
                interest = float(item.get("interest", 0.0) or 0.0)
                net_asset = float(item.get("netAsset", 0.0) or 0.0)

            has_any_liability = (
                total_liability_btc > MARGIN_EPS
                or borrowed > MARGIN_EPS
                or interest > MARGIN_EPS
            )
            if has_any_liability:
                logger.warning(
                    "Cross Margin possui liability; capital Margin ignorado por segurança | "
                    "totalLiabilityOfBtc=%.12f %s borrowed=%.8f interest=%.8f",
                    total_liability_btc,
                    asset,
                    borrowed,
                    interest,
                )
                max_transferable = 0.0
                available_own = 0.0
            else:
                transfer = self.client.get_max_margin_transfer(asset=asset)
                max_transferable = max(0.0, float(transfer.get("amount", 0.0) or 0.0))
                # Sem liabilities, free é capital próprio. Ainda limitamos pelo
                # netAsset e pelo maxTransferable calculado pela Binance.
                available_own = max(
                    0.0,
                    min(free, max(net_asset, 0.0), max_transferable),
                )

            return {
                "free": free,
                "locked": locked,
                "borrowed": borrowed,
                "interest": interest,
                "net_asset": net_asset,
                "max_transferable": max_transferable,
                "available_own": available_own,
                "has_any_liability": has_any_liability,
            }
        except BinanceAPIException as exc:
            raise RuntimeError(
                "Falha ao consultar Cross Margin para o pool Spot+Margin. "
                "Verifique as permissões da API (leitura + transferência universal) "
                "e a restrição de IP."
            ) from exc

    def get_account_balance(self, asset: str = "USDT") -> float:
        """Caixa operacional.

        No Aggressive C com pool habilitado, o quote asset é Spot livre +
        Cross Margin próprio/transferível. Para qualquer outro ativo/modo,
        mantém a semântica Spot original.
        """
        spot_free = self._get_spot_free_balance(asset)
        if not self.margin_pool_enabled or asset != LIVE_QUOTE_ASSET:
            return spot_free
        margin = self.get_margin_capital_snapshot(asset)["available_own"]
        return float(spot_free + margin)

    def get_asset_total(self, asset: str) -> float:
        balances = self.client.get_account()["balances"]
        for balance in balances:
            if balance["asset"] == asset:
                return float(balance["free"]) + float(balance["locked"])
        return 0.0

    def _ensure_spot_quote(self, required_quote: float) -> bool:
        """Move apenas USDT próprio Cross Margin -> Spot quando necessário.

        Nunca chama endpoints de borrow/loan e nunca move Spot -> Margin.
        """
        required_quote = max(0.0, float(required_quote))
        if required_quote <= 0:
            return True

        spot_free = self._get_spot_free_balance(LIVE_QUOTE_ASSET)
        if spot_free + 1e-8 >= required_quote:
            return True
        if not self.margin_pool_enabled:
            return False

        shortfall = required_quote - spot_free
        snap = self.get_margin_capital_snapshot(LIVE_QUOTE_ASSET)
        available = float(snap["available_own"])
        if available + 1e-8 < shortfall:
            logger.warning(
                "Pool Spot+Margin insuficiente | precisa %.8f %s | Spot %.8f | Margin próprio %.8f",
                required_quote,
                LIVE_QUOTE_ASSET,
                spot_free,
                available,
            )
            return False

        # 8 casas são suficientes para USDT e evitam transferir acima do saldo.
        amount = math.floor(min(shortfall, available) * 1e8) / 1e8
        if amount <= 0:
            return False
        try:
            result = self.client.make_universal_transfer(
                type="MARGIN_MAIN",
                asset=LIVE_QUOTE_ASSET,
                amount=f"{amount:.8f}",
            )
            logger.info(
                "Cross Margin -> Spot | %s %.8f | tranId=%s",
                LIVE_QUOTE_ASSET,
                amount,
                result.get("tranId"),
            )
        except BinanceAPIException as exc:
            logger.error("Falha ao transferir Cross Margin -> Spot: %s", exc)
            return False

        # Confirma a liquidez física antes de enviar a ordem Spot.
        for _ in range(3):
            spot_free = self._get_spot_free_balance(LIVE_QUOTE_ASSET)
            if spot_free + 1e-8 >= required_quote:
                return True
            time.sleep(1)
        logger.error(
            "Transferência confirmada mas saldo Spot ainda insuficiente | precisa %.8f | Spot %.8f",
            required_quote,
            spot_free,
        )
        return False

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

    def _round_tick(self, price: float, tick: float) -> float:
        if tick <= 0:
            return price
        precision = self._step_precision(tick)
        return round(round(price / tick) * tick, precision)

    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float, symbol_info: dict, reference_price: float | None = None) -> dict | None:
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
        if qty <= 0 or notional < symbol_info["min_notional"]:
            logger.warning("Ordem %s rejeitada | %s | notional %.8f < mínimo %.8f", side, symbol, notional, symbol_info["min_notional"])
            return None

        if side == Client.SIDE_BUY:
            required = notional * (
                1.0 + FEE_RATE + AGGRESSIVE_C_MARGIN_TRANSFER_BUFFER_PCT
            )
            if not self._ensure_spot_quote(required):
                logger.warning(
                    "BUY %s não enviada: caixa Spot+Margin insuficiente/indisponível",
                    symbol,
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
        """Só considera posição após fill; timeout cancela e preserva fills parciais."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                order = self.client.get_order(symbol=symbol, orderId=order_id)
                status = order.get("status")
                if status == "FILLED":
                    return order
                if status in {"CANCELED", "REJECTED", "EXPIRED"}:
                    executed = float(order.get("executedQty", 0))
                    return order if executed > 0 else None
            except BinanceAPIException as exc:
                logger.warning("get_order %s/%s falhou: %s", symbol, order_id, exc)
            time.sleep(ORDER_POLL_SEC)

        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            if order.get("status") != "FILLED":
                self.cancel_order(symbol, order_id)
                # Reconciliar imediatamente: a ordem pode ter tido fill parcial antes do cancel.
                order = self.client.get_order(symbol=symbol, orderId=order_id)
                self.get_open_orders(symbol)
                executed = float(order.get("executedQty", 0))
                return order if executed > 0 else None
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

    def has_active_oco(self, symbol: str, order_list_id: int | None) -> bool:
        if order_list_id is None:
            return False
        return any(int(o.get("orderListId", -1)) == int(order_list_id) for o in self.get_open_orders(symbol))

    def place_market_order(self, symbol: str, side: str, quantity: float, symbol_info: dict) -> dict | None:
        side = side.upper()
        qty = self._round_step(quantity, symbol_info["step_size"])
        if qty <= 0:
            return None
        if side == Client.SIDE_BUY:
            mid = self.get_current_price(symbol)
            required = qty * mid * (
                1.0
                + FEE_RATE
                + SLIPPAGE_LIMIT_PCT
                + AGGRESSIVE_C_MARGIN_TRANSFER_BUFFER_PCT
            )
            if not self._ensure_spot_quote(required):
                logger.warning(
                    "MARKET BUY %s não enviada: caixa Spot+Margin insuficiente/indisponível",
                    symbol,
                )
                return None
        try:
            return self.client.create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_MARKET,
                quantity=qty,
            )
        except BinanceAPIException as exc:
            logger.error("Erro ao enviar MARKET %s %s: %s", side, symbol, exc)
            return None

    def create_oco_order(self, symbol: str, quantity: float, take_profit_price: float, stop_price: float, symbol_info: dict) -> dict | None:
        """Cria OCO SELL real para proteger uma posição LONG spot."""
        qty = self._round_step(quantity, symbol_info["step_size"])
        tp = self._round_tick(take_profit_price, symbol_info["tick_size"])
        stop = self._round_tick(stop_price, symbol_info["tick_size"])
        try:
            order = self.client.create_oco_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                quantity=qty,
                price=f"{tp:.8f}",
                stopPrice=f"{stop:.8f}",
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
