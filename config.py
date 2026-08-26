# config.py — Configurações do TradeBot Krypton
import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Live e pesquisa usam exatamente os mesmos mercados Spot USDT.
TRADING_PAIRS = {
    "SOLUSDT": 0.3125,
    "BTCUSDT": 0.5000,
    "BNBUSDT": 0.1875,
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
# Parâmetro congelado antes do holdout final. Revalidar antes de alterá-lo.
TAKE_PROFIT_ATR_MULT = 3.0
CIRCUIT_BREAKER_PCT = 0.04
MAX_DRAWDOWN_PCT = 0.20
MAX_SIMULTANEOUS_POS = 3

REGIME_FILTER = True
REGIME_SMA_PERIOD = 200

TIMEFRAME = "1d"
SLIPPAGE_LIMIT_PCT = 0.005
FEE_RATE = 0.001
ENTRY_SLIPPAGE_PCT = 0.0005
EXIT_SLIPPAGE_PCT = 0.0005

# Execução/reconciliação live.
ENTRY_FILL_TIMEOUT_SEC = 120
EXIT_FILL_TIMEOUT_SEC = 120
ORDER_POLL_SEC = 5

LIVE_QUOTE_ASSET = "USDT"
BACKTEST_QUOTE_ASSET = "USDT"
STATE_DB_FILE = "krypton_state.db"

# Não transforma saldos comprados manualmente em posições do bot.
ADOPT_ORPHAN_POSITIONS = False

USE_TESTNET = True
LOG_FILE = "tradebot.log"
