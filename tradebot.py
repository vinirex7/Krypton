# tradebot.py — Loop Principal do Krypton TradeBot
# Estratégia: Supertrend + RSI + MACD Filter
#
# Ciclo diário executado às 00:05 UTC (após fechamento do candle D1):
#   1. Verifica SL/TP das posições abertas
#   2. Reset diário do circuit breaker
#   3. Para cada par: calcula sinais e abre/fecha posições
#   4. Entre ciclos: verifica SL/TP a cada 5 minutos (tempo real)
#
# ⚠️  USE_TESTNET = True por padrão em config.py.
#     Mude para False SOMENTE após ≥30 dias em testnet sem erros.

import logging
import time
from datetime import datetime

import schedule

from binance_client import BinanceInterface
from config import (
    LOG_FILE,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MAX_SIMULTANEOUS_POS,
    RSI_HIGH,
    RSI_LOW,
    RSI_PERIOD,
    SUPERTREND_MULTIPLIER,
    SUPERTREND_PERIOD,
    TIMEFRAME,
    TRADING_PAIRS,
    USE_TESTNET,
)
from indicators import compute_atr, compute_signals
from risk_manager import RiskManager

# ─── Configuração de Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("Krypton.Main")


class TradeBot:
    """
    Krypton TradeBot — Estratégia Supertrend + RSI + MACD Filter.

    Executa uma vez por dia após o fechamento do candle diário (00:05 UTC).
    Verifica SL/TP em tempo real a cada 5 minutos.

    Performance (backtest Jan/2022 – Mai/2026):
      Retorno Total : +37,7% (SOLU)
      Sharpe Ratio  : 0,932
      Max Drawdown  : -5,0%
      Win Rate      : 54,0%
      Profit Factor : 1,748
    """

    def __init__(self):
        self.binance      = BinanceInterface()
        self.risk_manager : RiskManager | None = None
        self.positions    : dict = {}         # {symbol: {side, entry_price, quantity, sl, tp, order_id}}
        self.symbol_infos : dict = {}
        self._initialize()

    # ─── Inicialização ────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Inicializa capital, symbol_infos e RiskManager."""
        u_balance = self.binance.get_account_balance("U")
        for sym in TRADING_PAIRS:
            self.symbol_infos[sym] = self.binance.get_symbol_info(sym)
        self.risk_manager = RiskManager(initial_capital=u_balance)
        logger.info(
            f"🟢 Krypton TradeBot inicializado | "
            f"Capital: ${u_balance:,.2f} U | "
            f"Modo: {'TESTNET ⚠️' if USE_TESTNET else 'PRODUÇÃO 🔴'}"
        )

    # ─── Cálculo de Capital ───────────────────────────────────────────────────

    def _get_current_capital(self) -> float:
        """
        Calcula capital total: U livre + valor mark-to-market das posições abertas.
        """
        u_balance = self.binance.get_account_balance("U")
        for sym, pos in self.positions.items():
            price = self.binance.get_current_price(sym)
            if pos["side"] == "LONG":
                pnl = pos["quantity"] * (price - pos["entry_price"])
            else:
                pnl = pos["quantity"] * (pos["entry_price"] - price)
            u_balance += pos["quantity"] * pos["entry_price"] + pnl
        return u_balance

    # ─── Gerenciamento de Posições ────────────────────────────────────────────

    def _close_position(self, symbol: str, reason: str = "Signal") -> None:
        """Fecha posição existente com ordem limit ao preço atual."""
        if symbol not in self.positions:
            return
        pos        = self.positions[symbol]
        price      = self.binance.get_current_price(symbol)
        close_side = "SELL" if pos["side"] == "LONG" else "BUY"
        order = self.binance.place_limit_order(
            symbol      = symbol,
            side        = close_side,
            quantity    = pos["quantity"],
            price       = price,
            symbol_info = self.symbol_infos[symbol],
        )
        if order:
            entry = pos["entry_price"]
            pnl_pct = (
                (price - entry) / entry * 100 if pos["side"] == "LONG"
                else (entry - price) / entry * 100
            )
            logger.info(
                f"📤 Posição fechada | {symbol} | Razão: {reason} | "
                f"Entry: {entry:.4f} | Exit: {price:.4f} | "
                f"PnL: {pnl_pct:+.2f}%"
            )
            del self.positions[symbol]

    def _open_position(
        self,
        symbol: str,
        direction: int,
        capital_allocation: float,
        atr: float,
    ) -> None:
        """
        Abre nova posição long (+1) ou short (-1) com sizing ATR-based.

        direction : +1 = LONG | -1 = SHORT
        """
        if symbol in self.positions:
            return  # já tem posição neste par

        current_capital = self._get_current_capital()
        if not self.risk_manager.can_trade(current_capital):
            return

        pair_capital = current_capital * capital_allocation
        price        = self.binance.get_current_price(symbol)
        sizing       = self.risk_manager.calculate_position_size(pair_capital, price, atr)

        if sizing["quantity"] <= 0:
            logger.warning(f"Position size calculado como zero para {symbol}. Pulando.")
            return

        side = "BUY" if direction == 1 else "SELL"
        # Limit price ligeiramente favorável ao mercado (0,05%)
        limit_price = price * (1.0005 if direction == 1 else 0.9995)

        order = self.binance.place_limit_order(
            symbol      = symbol,
            side        = side,
            quantity    = sizing["quantity"],
            price       = limit_price,
            symbol_info = self.symbol_infos[symbol],
        )

        if order:
            sl = sizing["stop_loss_long"]  if direction == 1 else sizing["stop_loss_short"]
            tp = sizing["take_profit_long"] if direction == 1 else sizing["take_profit_short"]
            self.positions[symbol] = {
                "side"        : "LONG" if direction == 1 else "SHORT",
                "entry_price" : price,
                "quantity"    : sizing["quantity"],
                "stop_loss"   : sl,
                "take_profit" : tp,
                "order_id"    : order["orderId"],
            }
            logger.info(
                f"📥 Nova posição | {symbol} | {self.positions[symbol]['side']} | "
                f"Entry: {price:.4f} | SL: {sl:.4f} | TP: {tp:.4f} | "
                f"Risco: ${sizing['risk_amount_usd']:.2f} | R:R {sizing['rr_ratio']:.1f}"
            )

    # ─── Verificação de SL/TP ─────────────────────────────────────────────────

    def _check_sl_tp(self) -> None:
        """Verifica se alguma posição aberta atingiu o SL ou TP. Executa a cada 5 min."""
        for symbol in list(self.positions.keys()):
            pos   = self.positions[symbol]
            price = self.binance.get_current_price(symbol)

            hit_sl = (
                (pos["side"] == "LONG"  and price <= pos["stop_loss"]) or
                (pos["side"] == "SHORT" and price >= pos["stop_loss"])
            )
            hit_tp = (
                (pos["side"] == "LONG"  and price >= pos["take_profit"]) or
                (pos["side"] == "SHORT" and price <= pos["take_profit"])
            )

            if hit_sl:
                self._close_position(symbol, "StopLoss 🔴")
            elif hit_tp:
                self._close_position(symbol, "TakeProfit ✅")

    # ─── Ciclo Diário Principal ───────────────────────────────────────────────

    def daily_cycle(self) -> None:
        """
        Ciclo principal executado às 00:05 UTC:
          1. Verifica SL/TP
          2. Reset diário do circuit breaker
          3. Avalia sinais e gerencia posições para cada par
        """
        logger.info("=" * 70)
        logger.info(
            f"🔄 CICLO DIÁRIO | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )

        current_capital = self._get_current_capital()
        logger.info(f"💰 Capital atual: ${current_capital:,.2f} U")
        logger.info(f"📊 {self.risk_manager.status(current_capital)}")

        # Passo 1: Verificar SL/TP antes de qualquer decisão
        self._check_sl_tp()

        # Passo 2: Reset do circuit breaker (novo dia)
        self.risk_manager.reset_daily(current_capital)

        # Passo 3: Verificar se pode operar
        if not self.risk_manager.can_trade(current_capital):
            logger.warning("⛔ Trading pausado pelos controles de risco.")
            return

        # Passo 4: Processar cada par de trading
        for symbol, allocation in TRADING_PAIRS.items():
            try:
                df = self.binance.get_ohlcv(symbol, interval=TIMEFRAME, limit=300)

                signals = compute_signals(
                    df,
                    st_period  = SUPERTREND_PERIOD,
                    st_mult    = SUPERTREND_MULTIPLIER,
                    rsi_period = RSI_PERIOD,
                    rsi_low    = RSI_LOW,
                    rsi_high   = RSI_HIGH,
                    macd_fast  = MACD_FAST,
                    macd_slow  = MACD_SLOW,
                    macd_sig   = MACD_SIGNAL,
                )

                current_signal = int(signals.iloc[-1])
                current_atr    = compute_atr(
                    df["high"], df["low"], df["close"]
                ).iloc[-1]

                signal_label = {1: "LONG ▲", -1: "SHORT ▼", 0: "FLAT —"}
                logger.info(
                    f"  {symbol} | Sinal: {signal_label[current_signal]} | "
                    f"ATR: {current_atr:.4f}"
                )

                # Fechar posição se sinal reverteu
                if symbol in self.positions:
                    pos_dir = 1 if self.positions[symbol]["side"] == "LONG" else -1
                    if current_signal != pos_dir:
                        self._close_position(symbol, "Signal reversal")

                # Abrir nova posição se há sinal e nenhuma posição aberta
                if symbol not in self.positions and current_signal != 0:
                    if len(self.positions) < MAX_SIMULTANEOUS_POS:
                        self._open_position(symbol, current_signal, allocation, current_atr)
                    else:
                        logger.info(f"  {symbol}: máx de posições simultâneas atingido ({MAX_SIMULTANEOUS_POS}).")

            except Exception as e:
                logger.error(f"Erro no ciclo de {symbol}: {e}", exc_info=True)

        logger.info(
            f"📂 Posições abertas: {len(self.positions)}/{MAX_SIMULTANEOUS_POS} | "
            f"{list(self.positions.keys())}"
        )

    # ─── Loop de Execução ─────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Inicia o bot:
          - Executa um ciclo imediato ao iniciar
          - Agenda ciclos diários às 00:05 UTC
          - Verifica SL/TP a cada 5 minutos
        """
        logger.info("🚀 Krypton TradeBot iniciado!")
        logger.info(f"   Estratégia: Supertrend({SUPERTREND_PERIOD},{SUPERTREND_MULTIPLIER}) + RSI({RSI_PERIOD}) + MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})")
        logger.info(f"   Pares: {list(TRADING_PAIRS.keys())}")
        logger.info(f"   Modo: {'TESTNET ⚠️  — NÃO opera com dinheiro real' if USE_TESTNET else 'PRODUÇÃO 🔴 — capital real em risco'}")

        # Ciclo imediato ao iniciar
        self.daily_cycle()

        # Agendamento diário às 00:05 UTC
        schedule.every().day.at("00:05").do(self.daily_cycle)

        logger.info("⏰ Próximo ciclo agendado: 00:05 UTC")
        logger.info("   Verificando SL/TP a cada 5 minutos...")

        while True:
            schedule.run_pending()
            self._check_sl_tp()
            time.sleep(300)  # 5 minutos


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = TradeBot()
    bot.run()
