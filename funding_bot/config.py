"""
Configuration parameters for the Funding Rate Arbitrage Bot
All tunable parameters in one place
"""

# =============================================================================
# TRADING PAIR
# =============================================================================
SYMBOL = "BTC/USDT"
BINANCE_SYMBOL = "BTCUSDT"  # Binance API format

# =============================================================================
# FUNDING RATE THRESHOLDS
# =============================================================================
# Positive funding = longs pay shorts (overleveraged long)
FUNDING_LONG_THRESHOLD = 0.0005   # +0.05% -> look for short entry
# Negative funding = shorts pay longs (overleveraged short)
FUNDING_SHORT_THRESHOLD = -0.0003  # -0.03% -> look for long entry
# Neutral zone for exit
FUNDING_NEUTRAL_LOW = -0.0001     # -0.01%
FUNDING_NEUTRAL_HIGH = 0.0001    # +0.01%

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================
# RSI settings
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65  # For short entry confirmation
RSI_OVERSOLD = 35    # For long entry confirmation

# ADX settings (market regime filter)
ADX_PERIOD = 14
ADX_THRESHOLD = 25   # Only trade when ADX < 25 (ranging market)

# =============================================================================
# POSITION MANAGEMENT
# =============================================================================
TAKE_PROFIT_PCT = 0.02    # 2% take profit
STOP_LOSS_PCT = 0.015     # 1.5% stop loss

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
RISK_PER_TRADE = 0.02     # 2% of capital per trade
MAX_POSITIONS = 1          # Max 1 open position at a time
DAILY_LOSS_LIMIT = 0.03   # Skip signals if daily PnL down 3%

# =============================================================================
# CAPITAL
# =============================================================================
INITIAL_CAPITAL = 10000   # Starting capital in USDT

# =============================================================================
# DATA FETCHING
# =============================================================================
CANDLE_INTERVAL = "1h"
LOOKBACK_DAYS = 730       # 2 years of data for backtest
FUNDING_INTERVAL_HOURS = 8  # Binance funding every 8 hours

# =============================================================================
# API SETTINGS
# =============================================================================
BINANCE_BASE_URL = "https://fapi.binance.com"
RATE_LIMIT_DELAY = 0.1    # Seconds between API calls

# =============================================================================
# LOGGING
# =============================================================================
LOG_FILE = "data/trades.csv"
PNL_CHART_FILE = "data/pnl_chart.png"

# =============================================================================
# SCHEDULER
# =============================================================================
CHECK_INTERVAL_MINUTES = 60  # Check for signals every hour
