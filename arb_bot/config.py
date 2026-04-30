"""
Cross-Exchange Arbitrage Bot Configuration
All parameters in one place
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# TRADING MODE
# =============================================================================
LIVE_TRADING = False  # False = observation mode, True = live trading

# =============================================================================
# EXCHANGE API KEYS
# =============================================================================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

OKX_API_KEY = os.getenv("OKX_API_KEY", "")
OKX_API_SECRET = os.getenv("OKX_API_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

# =============================================================================
# TELEGRAM NOTIFICATIONS
# =============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# =============================================================================
# TRADING PARAMETERS
# =============================================================================
SYMBOL = "BTC/USDT"
TRADE_SIZE_USD = 100.0  # Fixed $100 per leg

# Polling interval
POLL_INTERVAL_MS = 500  # 500ms between price checks

# =============================================================================
# FEE STRUCTURE (taker fees)
# =============================================================================
FEES = {
    "binance": {
        "spot": 0.0010,    # 0.10%
        "futures": 0.0004  # 0.04%
    },
    "bybit": {
        "spot": 0.0006,    # 0.06%
        "futures": 0.0006  # 0.06%
    },
    "okx": {
        "spot": 0.0005,    # 0.05%
        "futures": 0.0005  # 0.05%
    }
}

# Use spot fees for arbitrage
EXCHANGE_FEES = {
    "binance": 0.0010,  # 0.10%
    "bybit": 0.0006,    # 0.06%
    "okx": 0.0005       # 0.05%
}

# =============================================================================
# SPREAD THRESHOLDS
# =============================================================================
# Minimum net spread to trigger trade (after all fees + buffer for slippage)
MIN_SPREAD_TO_TRADE = 0.0015  # 0.15%

# Log but don't trade range
LOG_ONLY_SPREAD_MIN = 0.0008  # 0.08%
LOG_ONLY_SPREAD_MAX = 0.0015  # 0.15%

# =============================================================================
# RISK CONTROLS
# =============================================================================
MAX_SIMULTANEOUS_POSITIONS = 3
MAX_DAILY_LOSS_USD = 20.0
MAX_SINGLE_TRADE_LOSS_USD = 5.0
MAX_UNHEDGED_SECONDS = 30
CIRCUIT_BREAKER_SPREAD_MOVE = 0.001  # 0.1% adverse move triggers exit

# =============================================================================
# DATA STORAGE
# =============================================================================
DATABASE_PATH = "data/arb_bot.db"

# =============================================================================
# EXCHANGE PAIRS FOR ARBITRAGE
# =============================================================================
EXCHANGE_PAIRS = [
    ("binance", "bybit"),
    ("binance", "okx"),
    ("bybit", "okx")
]

EXCHANGES = ["binance", "bybit", "okx"]
