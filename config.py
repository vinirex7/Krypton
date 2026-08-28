# config.py — Configurações do TradeBot Krypton
import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Live e pesquisa usam exatamente os mesmos mercados Spot USDT no sleeve tático.
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

# Perfil legado/congelado. Aggressive C usa constantes próprias abaixo para não
# alterar silenciosamente backtests ou o modo legado.
RISK_PER_TRADE = 0.01
STOP_LOSS_ATR_MULT = 2.0
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

# ---------------------------------------------------------------------------
# Aggressive C — configuração promovida após a bateria C-only de 2026-08-27.
# Estes números reproduzem a candidata C validada; não otimizar em produção.
# ---------------------------------------------------------------------------
LIVE_STRATEGY = os.getenv("LIVE_STRATEGY", "AGGRESSIVE_C").strip().upper()
AGGRESSIVE_C_TACTICAL_WEIGHT = 0.55
AGGRESSIVE_C_ALPHA_WEIGHT = 0.45
AGGRESSIVE_C_TACTICAL_RISK_PER_TRADE = 0.02
AGGRESSIVE_C_ALPHA_TARGET_VOL = 0.30
AGGRESSIVE_C_ALPHA_TOP_N = 2
AGGRESSIVE_C_ALPHA_MIN_SELECTED = 2
AGGRESSIVE_C_ALPHA_REBALANCE_DAYS = 45
AGGRESSIVE_C_SLEEVE_REBALANCE_DAYS = 90
AGGRESSIVE_C_TRANSFER_COST_ASSUMPTION = 0.003
AGGRESSIVE_C_MAX_DRAWDOWN_PCT = 0.30
AGGRESSIVE_C_MOMENTUM_WINDOWS = (30, 90, 180)
AGGRESSIVE_C_VOL_WINDOW = 20
AGGRESSIVE_C_COV_WINDOW = 60
AGGRESSIVE_C_CONTINUITY_MIN_AGE = 90
AGGRESSIVE_C_CONTINUITY_SHORT_WINDOW = 30
AGGRESSIVE_C_CONTINUITY_LONG_WINDOW = 90
AGGRESSIVE_C_ALPHA_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
AGGRESSIVE_C_STATE_DB_FILE = os.getenv("AGGRESSIVE_C_STATE_DB_FILE", "krypton_c_state.db")
AGGRESSIVE_C_REQUIRE_CLEAN_START = os.getenv("AGGRESSIVE_C_REQUIRE_CLEAN_START", "true").strip().lower() in {"1", "true", "yes", "on"}

# Pool opcional de capital: Spot USDT + USDT próprio/transferível em Cross Margin.
# Continua Spot long-only 1x: o bot nunca toma empréstimo e nunca envia ordem Margin.
# O default é false para exigir opt-in explícito no VPS.
AGGRESSIVE_C_USE_MARGIN_CAPITAL_POOL = os.getenv(
    "AGGRESSIVE_C_USE_MARGIN_CAPITAL_POOL", "false"
).strip().lower() in {"1", "true", "yes", "on"}
# Pequena folga física ao mover Margin -> Spot antes de uma BUY. Não altera sizing.
AGGRESSIVE_C_MARGIN_TRANSFER_BUFFER_PCT = 0.002

# Segurança operacional: o repositório continua iniciando em TESTNET por padrão.
# Para capital real, definir USE_TESTNET=false explicitamente no .env do VPS.
USE_TESTNET = os.getenv("USE_TESTNET", "true").strip().lower() in {"1", "true", "yes", "on"}
LOG_FILE = "tradebot.log"
