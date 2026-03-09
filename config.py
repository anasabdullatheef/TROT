"""
Atlas Bot Configuration
Advanced ML-driven trading bot with regime detection and sentiment analysis
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# TRADING PAIR
# =============================================================================
SYMBOL = "BTC/USDT"
BINANCE_SYMBOL = "BTCUSDT"

# =============================================================================
# API KEYS
# =============================================================================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")

# =============================================================================
# DATA FETCHING
# =============================================================================
BINANCE_BASE_URL = "https://fapi.binance.com"
FNG_API_URL = "https://api.alternative.me/fng/"
CRYPTOPANIC_API_URL = "https://cryptopanic.com/api/free/v1/posts/"
CANDLE_INTERVAL = "1h"
LOOKBACK_DAYS = 730  # 2 years
RATE_LIMIT_DELAY = 0.1

# =============================================================================
# REGIME DETECTION (HMM)
# =============================================================================
HMM_N_STATES = 4
HMM_N_ITER = 100
HMM_COVARIANCE_TYPE = "full"
HMM_FEATURES = ["returns", "volatility", "volume_ratio", "atr", "adx"]

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
# Price returns
RETURN_PERIODS = [1, 4, 24]  # 1h, 4h, 24h

# Technical indicators
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
STOCH_K = 14
STOCH_D = 3
VOLATILITY_WINDOW = 24
VOLUME_MA_PERIOD = 20

# Sentiment
SENTIMENT_LOOKBACK_1H = 1
SENTIMENT_LOOKBACK_4H = 4
SENTIMENT_SPIKE_THRESHOLD = 2.0  # Standard deviations
CONSECUTIVE_NEGATIVE_THRESHOLD = 3

# =============================================================================
# LSTM MODEL
# =============================================================================
LSTM_SEQUENCE_LENGTH = 24  # Last 24 candles
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_LEARNING_RATE = 0.001
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_PREDICTION_HORIZON = 4  # Predict 4h ahead
LSTM_CONFIDENCE_THRESHOLD = 0.65

# =============================================================================
# FUNDING RATE (from Config B)
# =============================================================================
FUNDING_LONG_THRESHOLD = 0.0003   # 0.03% -> short
FUNDING_SHORT_THRESHOLD = -0.0001  # -0.01% -> long
FUNDING_BOOST_WEIGHT = 0.20  # 20% confidence boost when funding signal triggers

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
INITIAL_CAPITAL = 10000
MAX_CAPITAL_PER_TRADE = 0.03  # 3% max per trade
KELLY_FRACTION = 0.25  # Quarter Kelly
ATR_STOP_LOSS_MULTIPLIER = 1.5
ATR_TAKE_PROFIT_MULTIPLIER = 3.0
DAILY_DRAWDOWN_LIMIT = 0.05  # 5% daily kill switch
MAX_POSITIONS = 1
MACRO_BLACKOUT_HOURS = 2  # Hours around major events

# =============================================================================
# DATABASE
# =============================================================================
DATABASE_PATH = "data/atlas.db"

# =============================================================================
# MODEL PATHS
# =============================================================================
HMM_MODEL_PATH = "models/hmm_model.pkl"
LSTM_MODEL_PATH_TEMPLATE = "models/lstm_regime_{}.pt"
SCALER_PATH = "models/scaler.pkl"

# =============================================================================
# LOGGING
# =============================================================================
LOG_FILE = "data/trades.csv"
PNL_CHART_FILE = "data/pnl_chart.png"

# =============================================================================
# SCHEDULER
# =============================================================================
CHECK_INTERVAL_MINUTES = 60
RETRAIN_INTERVAL_DAYS = 7

# =============================================================================
# MACRO EVENTS (UTC times - sample list)
# =============================================================================
MACRO_EVENTS = [
    # Format: (month, day, hour) - FOMC meetings, CPI releases, etc.
    # These would be updated regularly
]
