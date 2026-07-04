# binance_client.py — Interface Segura com a API da Binance
# Krypton TradeBot | Estratégia: Supertrend + RSI + MACD Filter
#
# Funcionalidades:
#   - Busca de candles OHLCV (get_ohlcv)
#   - Informações do par (step_size, tick_size, min_notional)
#   - Saldo da conta (get_account_balance)
#   - Preço atual via orderbook midpoint (get_current_price)
#   - Execução de ordens limit com validação de slippage (place_limit_order)
#   - Cancelamento de ordens (cancel_order)
#   - Listagem de ordens abertas (get_open_orders)
#
# Segurança:
#   - Apenas limit orders (nunca market orders)
#   - Validação de slippage: max 0,5% do mid-price
#   - Permissões API: apenas 'Enable Spot & Margin Trading'
#   - NUNCA habilitar saque na API key

import logging
import math
import time

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    USE_TESTNET,
    TIMEFRAME,
    SLIPPAGE_LIMIT_PCT,
    FEE_RATE,
)

logger = logging.getLogger("Krypton.Binance")

# Constantes de retry
MAX_RETRIES  = 3
RETRY_DELAY  = 5  # segundos entre tentativas
MIN_NOTIONAL_ROUND_UP_THRESHOLD = 8.00  # U: compra entre 8 e o mínimo da Binance sobe para o mínimo


class BinanceInterface:
    """
    Interface segura com a API Binance para o Krypton TradeBot.

    Todas as ordens são do tipo LIMIT (nunca MARKET) para controle de slippage.
    O desvio máximo aceito em relação ao mid-price é 0,5%.
    """

    def __init__(self):
        self.client = Client(
            BINANCE_API_KEY,
            BINANCE_API_SECRET,
            testnet=USE_TESTNET,
        )
        self.client.ping()
        mode = "TESTNET ⚠️" if USE_TESTNET else "PRODUÇÃO 🔴"
        logger.info(f"Binance API conectada | Modo: {mode}")

    # ─── Dados de Mercado ─────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 300,
    ) -> pd.DataFrame:
        """
        Busca candles OHLCV e retorna DataFrame pandas.

        Parâmetros
        ----------
        symbol   : str  — Par de trading (ex: 'SOLU')
        interval : str  — Timeframe (padrão: '1d')
        limit    : int  — Número de candles (padrão: 300)

        Retorna
        -------
        pd.DataFrame com colunas [open, high, low, close, volume]
        indexado por open_time (datetime).
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                )
                df = pd.DataFrame(
                    klines,
                    columns=[
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_vol", "trades",
                        "taker_base", "taker_quote", "ignore",
                    ],
                )
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                df.set_index("open_time", inplace=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                return df[["open", "high", "low", "close", "volume"]]
            except BinanceAPIException as e:
                logger.warning(f"get_ohlcv tentativa {attempt}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        raise RuntimeError(f"Falha ao buscar OHLCV para {symbol} após {MAX_RETRIES} tentativas.")

    def get_symbol_info(self, symbol: str) -> dict:
        """
        Retorna informações do par: step_size, tick_size, min_notional.
        Necessário para arredondar quantity e price corretamente.
        """
        info    = self.client.get_symbol_info(symbol)
        filters = {f["filterType"]: f for f in info["filters"]}
        return {
            "step_size"   : float(filters["LOT_SIZE"]["stepSize"]),
            "tick_size"   : float(filters["PRICE_FILTER"]["tickSize"]),
            "min_notional": float(
                filters.get("MIN_NOTIONAL", {}).get("minNotional", 10)
            ),
        }

    def get_account_balance(self, asset: str = "U") -> float:
        """Retorna saldo disponível (free) do ativo especificado."""
        balances = self.client.get_account()["balances"]
        for b in balances:
            if b["asset"] == asset:
                return float(b["free"])
        return 0.0

    def get_current_price(self, symbol: str) -> float:
        """Retorna preço atual via midpoint entre melhor bid e melhor ask."""
        ticker = self.client.get_orderbook_ticker(symbol=symbol)
        bid    = float(ticker["bidPrice"])
        ask    = float(ticker["askPrice"])
        return (bid + ask) / 2

    # ─── Utilitários de Arredondamento ────────────────────────────────────────

    def _step_precision(self, step: float) -> int:
        """Retorna a precisão decimal necessária para um step_size/tick_size."""
        return len(str(step).rstrip("0").split(".")[-1])

    def _round_step(self, qty: float, step: float) -> float:
        """Arredonda quantidade para o step_size do par (floor)."""
        if step == 0:
            return qty
        precision = self._step_precision(step)
        return round(math.floor(qty / step) * step, precision)

    def _ceil_step(self, qty: float, step: float) -> float:
        """Arredonda quantidade para cima respeitando o step_size do par."""
        if step == 0:
            return qty
        precision = self._step_precision(step)
        return round(math.ceil(qty / step) * step, precision)

    def _round_tick(self, price: float, tick: float) -> float:
        """Arredonda preço para o tick_size do par."""
        if tick == 0:
            return price
        precision = self._step_precision(tick)
        return round(round(price / tick) * tick, precision)

    # ─── Execução de Ordens ───────────────────────────────────────────────────

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        symbol_info: dict,
    ) -> dict | None:
        """
        Coloca ordem limit com validação de slippage.

        Regras de segurança:
          - Rejeita se desvio do mid-price > SLIPPAGE_LIMIT_PCT (0,5%)
          - Para BUY entre $8 e o mínimo da Binance, ajusta para o mínimo
          - Rejeita se notional < min_notional após o ajuste
          - Arredonda quantity ao step_size e price ao tick_size

        Parâmetros
        ----------
        side : 'BUY' ou 'SELL'

        Retorna
        -------
        dict com dados da ordem, ou None se rejeitada/falhou.
        """
        mid_price       = self.get_current_price(symbol)
        price_deviation = abs(price - mid_price) / mid_price

        if price_deviation > SLIPPAGE_LIMIT_PCT:
            logger.warning(
                f"Ordem {side} rejeitada — slippage {price_deviation:.2%} "
                f"> limite {SLIPPAGE_LIMIT_PCT:.1%} | {symbol}"
            )
            return None

        side_upper     = side.upper()
        qty_rounded   = self._round_step(quantity, symbol_info["step_size"])
        price_rounded = self._round_tick(price, symbol_info["tick_size"])
        notional      = qty_rounded * price_rounded
        min_notional  = symbol_info["min_notional"]

        if (
            side_upper == "BUY"
            and MIN_NOTIONAL_ROUND_UP_THRESHOLD <= notional < min_notional
        ):
            target_qty  = min_notional / price_rounded
            qty_rounded = self._ceil_step(target_qty, symbol_info["step_size"])
            notional    = qty_rounded * price_rounded
            logger.info(
                f"Ajuste de compra mínima | {symbol} | "
                f"notional calculado abaixo do mínimo e >= ${MIN_NOTIONAL_ROUND_UP_THRESHOLD:.2f}; "
                f"nova qty: {qty_rounded} | novo notional: ${notional:.2f}"
            )

        if notional < min_notional:
            logger.warning(
                f"Ordem {side_upper} rejeitada — notional ${notional:.2f} "
                f"< mínimo ${min_notional:.2f} | {symbol}"
            )
            return None

        try:
            order = self.client.create_order(
                symbol      = symbol,
                side        = side_upper,
                type        = Client.ORDER_TYPE_LIMIT,
                timeInForce = Client.TIME_IN_FORCE_GTC,
                quantity    = qty_rounded,
                price       = f"{price_rounded:.8f}",
            )
            logger.info(
                f"✅ Ordem {side_upper} enviada | {symbol} | "
                f"Qty: {qty_rounded} | Price: {price_rounded:.4f} | "
                f"Notional: ${notional:.2f}"
            )
            return order
        except BinanceAPIException as e:
            logger.error(f"Erro ao enviar ordem {side_upper} {symbol}: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancela uma ordem aberta. Retorna True se bem-sucedido."""
        try:
            self.client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Ordem {order_id} cancelada | {symbol}")
            return True
        except BinanceAPIException as e:
            logger.error(f"Erro ao cancelar ordem {order_id} ({symbol}): {e}")
            return False

    def get_open_orders(self, symbol: str) -> list:
        """Retorna lista de ordens abertas para o par."""
        return self.client.get_open_orders(symbol=symbol)
