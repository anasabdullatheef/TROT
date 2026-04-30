"""
Configuration settings for the trading bot.
All parameters in one place for easy tuning.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API Configuration
# =============================================================================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
TRADING_MODE = os.getenv("TRADING_MODE", "testnet")  # 'testnet' or 'live'

# =============================================================================
# Trading Pair Configuration
# =============================================================================
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"

# =============================================================================
# Strategy Parameters
# =============================================================================
EMA_PERIOD = 20
RSI_PERIOD = 14
RSI_THRESHOLD = 50
VOLUME_MA_PERIOD = 20
VOLUME_MULTIPLIER = 1.0  # Volume must be >= this * average volume

# =============================================================================
# Risk Management
# =============================================================================
RISK_PER_TRADE = 0.02          # 2% of capital per trade
STOP_LOSS_PCT = 0.015          # 1.5% stop-loss
TAKE_PROFIT_PCT = 0.03         # 3% take-profit
MAX_OPEN_POSITIONS = 3
DAILY_DRAWDOWN_LIMIT = 0.05    # 5% daily drawdown kill switch

# =============================================================================
# Capital Configuration
# =============================================================================
INITIAL_CAPITAL = 10000.0      # Starting capital in USDT

# =============================================================================
# Scheduler Configuration
# =============================================================================
CHECK_INTERVAL_MINUTES = 1     # How often to check for signals

# =============================================================================
# Backtest Configuration
# =============================================================================
BACKTEST_DAYS = 180            # 6 months of historical data
BACKTEST_START = None          # If None, calculates from BACKTEST_DAYS

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_TO_FILE = True
LOG_FILE_PATH = "logs/trades.csv"
CONSOLE_LOG = True
