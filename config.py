# config.py — Configurações do TradeBot Krypton
import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# LIVE usa U; BACKTEST usa USDT. Os dois representam a mesma quote asset operacionalmente.
TRADING_PAIRS = {
    "SOLU": 0.25,
    "BTCU": 0.40,
    "ETHU": 0.20,
    "BNBU": 0.15,
}

SUPERTREND_PERIOD = 7
SUPERTREND_MULTIPLIER = 3.0

RSI_PERIOD = 14
RSI_LOW = 40
RSI_HIGH = 70

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

RISK_PER_TRADE = 0.01
STOP_LOSS_ATR_MULT = 2.0
# R:R = 4.5 / 2.0 = 2.25:1; break-even WR ≈ 30.8% antes de custos.
TAKE_PROFIT_ATR_MULT = 4.5
CIRCUIT_BREAKER_PCT = 0.04
MAX_DRAWDOWN_PCT = 0.20
MAX_SIMULTANEOUS_POS = 4

TIMEFRAME = "1d"
SLIPPAGE_LIMIT_PCT = 0.005
FEE_RATE = 0.001

# Execução/reconciliação live.
ENTRY_FILL_TIMEOUT_SEC = 120
EXIT_FILL_TIMEOUT_SEC = 120
ORDER_POLL_SEC = 5

LIVE_QUOTE_ASSET = "U"
BACKTEST_QUOTE_ASSET = "USDT"
STATE_DB_FILE = "krypton_state.db"

USE_TESTNET = True
LOG_FILE = "tradebot.log"
