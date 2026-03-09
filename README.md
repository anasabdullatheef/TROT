# Atlas Bot

ML-driven BTC/USDT perpetual futures trading system with regime detection, order flow analysis, and on-chain features.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ATLAS BOT                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   HMM        │    │   LSTM       │    │   Strategy   │       │
│  │   Regime     │───▶│   Per        │───▶│   Engine     │       │
│  │   Detection  │    │   Regime     │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                    FEATURES                           │       │
│  ├──────────────┬──────────────┬──────────────┐         │       │
│  │ Order Flow   │ On-Chain     │ Funding      │         │       │
│  │ - Orderbook  │ - F&G Index  │ - Rate       │         │       │
│  │ - VPIN       │ - Hashrate   │ - Momentum   │         │       │
│  │ - Liquids    │ - Tx Count   │ - Z-score    │         │       │
│  └──────────────┴──────────────┴──────────────┘         │       │
│                                                          │       │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Regime Detection
- **Hidden Markov Model (HMM)** with 4 states
- States: Low Vol Ranging, Med Vol Trending, Med Vol Ranging, High Vol Trending
- Transition probability matrix for regime persistence

### LSTM Prediction
- Separate LSTM model trained per regime
- Predicts probability of positive 4-hour return
- Uses only leading indicators (no lagging technical indicators)

### Data Collection
- Real-time order book snapshots (top 20 levels)
- Trade flow toxicity (VPIN - buy vs sell initiated volume)
- Cross-exchange price divergence (Binance, Bybit, OKX)
- Liquidation data

### On-Chain Features
- Fear & Greed Index
- Blockchain hashrate
- Transaction count
- Exchange flow estimates

### Risk Management
- Fixed 2% position sizing per trade
- 0.15% transaction costs (0.1% fee + 0.05% slippage)
- Dynamic ATR-based TP/SL (3x ATR take profit, 1.5x ATR stop loss)
- 20% maximum drawdown circuit breaker

## Installation

```bash
# Clone repository
git clone https://github.com/anasabdullatheef/TROT.git
cd TROT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Edit `.env` file:

```env
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
GLASSNODE_API_KEY=your_key  # Optional, for on-chain data
```

## Usage

### Run Backtest

```bash
# Fixed backtest with proper train/test split
python fixed_backtest.py

# Order flow features backtest
python orderflow_backtest.py

# On-chain features backtest
python onchain_backtest.py
```

### Start Data Collector

The collector runs every minute to gather real order flow data:

```bash
# Single collection
python collector.py

# Setup cron job (runs every minute 24/7)
./setup_collector_cron.sh

# Check collector logs
tail -f logs/collector.log
```

### Live Trading

```bash
python main.py
```

## Project Structure

```
atlas_bot/
├── main.py                 # Live trading entry point
├── config.py               # Configuration parameters
├── collector.py            # 24/7 order flow data collector
│
├── features.py             # Technical feature engineering
├── orderflow_features.py   # Order flow features
├── onchain_features.py     # On-chain data integration
│
├── regime.py               # HMM regime detection
├── models/
│   ├── __init__.py
│   └── lstm_model.py       # LSTM models per regime
│
├── strategy.py             # Trading strategy logic
├── risk.py                 # Risk management & position sizing
├── exchange.py             # Binance API wrapper
├── funding.py              # Funding rate data
│
├── backtest.py             # Full backtest implementation
├── fixed_backtest.py       # Backtest with proper train/test split
├── orderflow_backtest.py   # Order flow features backtest
├── onchain_backtest.py     # On-chain features backtest
│
├── database.py             # SQLite storage
├── logger.py               # Trade logging
├── telegram_alerts.py      # Telegram notifications
│
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── .gitignore
```

## Backtest Results

### Current Performance (6-month out-of-sample test)

| Metric | Value |
|--------|-------|
| Return | -1.33% |
| Annualized | -2.65% |
| Win Rate | 33.5% |
| Profit Factor | 0.76 |
| Max Drawdown | 1.63% |
| Sharpe Ratio | -1.60 |

### Model AUC by Regime

| Regime | AUC | Status |
|--------|-----|--------|
| R0 (Med Vol Trending) | 0.518 | Needs improvement |
| R1 (Med Vol Ranging) | 0.516 | Needs improvement |
| R2 (Low Vol Ranging) | 0.582 | Marginal |
| R3 (High Vol Trending) | 0.498 | Random |

**Target: AUC > 0.60 for viable strategy**

## Data Collection Status

The collector is gathering real order flow data for future model training:

```
Database Statistics:
  orderbook: Collecting every minute
  tradeflow: VPIN, buy/sell volume
  liquidations: When available
  cross_exchange: Price divergence
```

After 6 months of collection, retrain models with real order flow data.

## Dependencies

- Python 3.10+
- PyTorch (LSTM models)
- hmmlearn (Regime detection)
- ccxt (Exchange connectivity)
- pandas, numpy, scikit-learn
- ta (Technical analysis)

## License

MIT

## Development Status

### Completed

- [x] HMM regime detection (4 states)
- [x] LSTM models per regime
- [x] Proper train/test split (18 months train, 6 months test)
- [x] Fixed 2% position sizing (no Kelly criterion)
- [x] Transaction costs modeling (0.15% per trade)
- [x] 20% max drawdown circuit breaker
- [x] Order flow data collector (orderbook, VPIN, cross-exchange)
- [x] On-chain features (Fear & Greed, blockchain metrics)
- [x] Funding rate momentum features
- [x] Backtesting framework with proper validation

### In Progress

- [ ] Collecting real order flow data (6 months needed)
- [ ] Fear & Greed integration working (730 days historical)
- [ ] Blockchain.com metrics working (hashrate, tx count)

### Next Steps

1. **Data Collection (6 months)**
   - Run `collector.py` via cron 24/7
   - Accumulate real orderbook, VPIN, liquidation data
   - Build historical cross-exchange divergence dataset

2. **Feature Engineering**
   - Test daily timeframe instead of hourly
   - Add Glassnode API for real exchange netflow
   - Integrate options/derivatives flow data

3. **Model Improvement**
   - Target AUC > 0.60 (currently 0.50-0.58)
   - Experiment with attention mechanisms
   - Try regime-specific feature selection

4. **Live Trading**
   - Paper trade for 1 month minimum
   - Implement Telegram alerts
   - Add position monitoring dashboard

### Known Issues

- LSTM AUC near random (~0.52) due to simulated order flow features
- On-chain features estimated from price (not truly leading)
- Need real historical data for proper backtesting

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.
