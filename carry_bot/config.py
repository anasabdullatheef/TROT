"""
Carry Bot Configuration
Simple funding rate collection strategy
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# EXCHANGE API
# =============================================================================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# =============================================================================
# TRADING PARAMETERS
# =============================================================================
SYMBOL = "BTC/USDT:USDT"  # Binance BTCUSDT perpetual
POSITION_SIZE_USD = 200.0  # Fixed $200 position

# =============================================================================
# FUNDING RATE THRESHOLDS
# =============================================================================
# Funding rate is per 8 hours on Binance
# +0.08% = very high positive funding -> shorts pay longs -> GO SHORT to collect
# -0.04% = negative funding -> longs pay shorts -> GO LONG to collect

FUNDING_SHORT_THRESHOLD = 0.0008   # +0.08% - go short when funding above this
FUNDING_LONG_THRESHOLD = -0.0004   # -0.04% - go long when funding below this

# Exit when funding normalizes
FUNDING_EXIT_HIGH = 0.0001   # +0.01%
FUNDING_EXIT_LOW = -0.0001   # -0.01%

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
STOP_LOSS_PCT = 0.02  # 2% stop loss on position

# =============================================================================
# POLLING
# =============================================================================
POLL_INTERVAL_SECONDS = 3600  # Check every hour (funding settles every 8 hours)

# =============================================================================
# DATA
# =============================================================================
DATABASE_PATH = "data/carry_bot.db"
