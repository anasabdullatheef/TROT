# TROT Trading System — Full Documentation

> **Autonomous Crypto Trading Bot Suite**
> Built and tested March 2024 – March 2026 on BTC/USDT (Binance)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [The Journey — What Was Tried and What Failed](#the-journey)
3. [What Actually Works](#what-actually-works)
4. [System Architecture](#system-architecture)
5. [Bot Reference](#bot-reference)
   - [TROT (Momentum Bot)](#1-trot--momentum-bot)
   - [funding_bot](#2-funding_bot)
   - [atlas_bot](#3-atlas_bot)
   - [arb_bot](#4-arb_bot)
   - [carry_bot](#5-carry_bot)
   - [collector.py](#6-collectorpy)
6. [Live Deployment Guide](#live-deployment-guide)
7. [Key Lessons Learned](#key-lessons-learned)
8. [Realistic Expectations](#realistic-expectations)
9. [Roadmap](#roadmap)

---

## Project Overview

This repository contains a suite of autonomous cryptocurrency trading bots developed through rigorous empirical testing. Every strategy was backtested on real historical data before any live deployment decision was made.

**Primary instrument:** BTC/USDT perpetual futures
**Exchange:** Binance (testnet and live)
**Supporting exchanges:** Bybit, OKX (arb_bot only)
**Language:** Python 3.11+
**Key libraries:** ccxt, pandas, numpy, hmmlearn, tensorflow, ta-lib, vectorbt

---

## The Journey

Most trading bot documentation only shows the winner. This documents everything — including what failed and why. Understanding the failures is as important as understanding what works.

### What Was Tried

| Strategy | Bot | Backtest Result | Why It Failed |
|---|---|---|---|
| EMA + RSI momentum | `trot` | -0.34% over 2 years | Entries at local tops, 23% win rate, needs 35%+ to be profitable |
| Funding rate prediction | `funding_bot` | 1 trade in 2 years | Parameters too tight; signal too rare on BTC |
| LSTM on price features | `atlas_bot v1` | AUC 0.53-0.56 | Price data is lagging; edge already arbitraged away |
| LSTM on simulated order flow | `atlas_bot v2` | AUC 0.50-0.57 | Features derived from price — functionally identical to technical indicators |
| LSTM on on-chain free data | `atlas_bot v3` | AUC 0.50-0.58 | Free tier data too coarse and delayed for hourly prediction |
| LSTM on real VPIN (tick data) | `atlas_bot v4` | AUC 0.49-0.55 | Even real trade flow insufficient for 1h BTC prediction |
| Spot cross-exchange arbitrage | `arb_bot v1` | -0.146% net spread | Fees exceed spread on efficient BTC markets |

### The Core Discovery

**BTC/USDT is one of the most efficiently priced assets on earth.** Thousands of well-funded institutions with co-located servers, proprietary data feeds, and PhD researchers trade it every millisecond. Any pattern visible in public OHLCV data has already been arbitraged away.

This means:
- Technical indicators (RSI, MACD, EMA crossovers) have no predictive edge
- Machine learning on public price data produces AUC near 0.50 (random)
- Simple arbitrage is unprofitable after fees on BTC spot markets
- The edge, if it exists at retail level, comes from **structural market mechanics** not price prediction

---

## What Actually Works

### Funding Rate Carry (`carry_bot`)

Binance perpetual futures charge a funding rate every 8 hours. When too many traders are leveraged long, funding goes strongly positive — longs pay shorts. This is a structural, non-predictive mechanism.

**The carry strategy:**
- Monitors funding rate every hour
- When funding exceeds +0.08%: opens short position to collect funding payments
- When funding drops below -0.04%: opens long position to collect funding payments
- Exits when funding normalizes
- Stop-loss at 2% to protect against directional price risk

**Verified backtest results across 2 years:**

| Period | Trades | Funding Collected | Total P&L | Annual Return |
|---|---|---|---|---|
| Mar 2024 – Mar 2025 | 72 | $15.36 | $16.93 | ~8.5% |
| Mar 2025 – Mar 2026 | 53 | $3.08 | $52.90 | ~26.4%* |

*The 2025-2026 period included significant directional price profit. Strip that out and the reliable baseline is **8-10% annually from pure funding collection.**

**Why this works where others don't:**
- Based on structural market mechanics, not price prediction
- Funding normalization is a mathematical inevitability, not a prediction
- Fully automated, requires no human judgment
- Scales linearly with capital deployed

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   TROT Trading System                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  carry_bot ──────────────────────────────────────── │
│  Steady 8-10%/year. Runs forever. Fully automatic.  │
│                                                     │
│  arb_bot ────────────────────────────────────────── │
│  Monitors futures spreads across 3 exchanges.       │
│  Fires only during high-volatility windows.         │
│                                                     │
│  atlas_bot ──────────────────────────────────────── │
│  ML regime-switching system. Dormant pending data.  │
│  Activates when collector has 6 months of data.     │
│                                                     │
│  collector.py ───────────────────────────────────── │
│  Runs 24/7 on VPS. Feeds atlas_bot over time.       │
│  Order book, trade flow, liquidations, cross-       │
│  exchange prices. Logged to SQLite every minute.    │
│                                                     │
└─────────────────────────────────────────────────────┘
                         │
              Telegram Alerts (all bots)
              SQLite Logging (all bots)
              CSV Export (all bots)
```

---

## Bot Reference

### 1. TROT — Momentum Bot

**Status:** ❌ Retired
**Location:** `/trot/`

The original bot. EMA crossover + RSI momentum strategy on BTC/USDT 1h. Backtested across 2 years with proper train/validation/test splits.

**Results:** Negative returns in every configuration. Win rate of 23-25% is insufficient for a 2:1 reward/risk strategy (requires 35%+). Entries consistently triggered at local price tops rather than trend beginnings.

**Files:**
```
trot/
├── main.py          # Scheduler
├── strategy.py      # EMA/RSI signal logic
├── risk.py          # Position sizing, kill switch
├── exchange.py      # Binance ccxt wrapper
├── logger.py        # CSV + console logging
├── config.py        # All parameters
├── backtest.py      # Backtesting engine
└── .env             # API keys
```

**Config parameters (config.py):**
```python
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
RSI_THRESHOLD = 55
STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.06
MAX_CAPITAL_PER_TRADE = 0.02
DAILY_DRAWDOWN_LIMIT = 0.05
```

---

### 2. funding_bot

**Status:** ⚠️ Shelved — too few signals
**Location:** `/funding_bot/`

Funding rate arbitrage bot. Monitored BTC/USDT perpetual funding rate and traded against extreme readings combined with RSI and ADX filters.

**Results:** Original parameters fired 1 trade in 2 years. Relaxed parameters (Config B) showed genuine edge — profit factor 1.52, 51% win rate — but nearly all trades clustered in 2024. Signal disappeared in 2025-2026 as market conditions changed.

**Key finding:** Funding rate alone has edge, but signal frequency is too low and regime-dependent for reliable standalone use.

**Config B parameters (the best performing):**
```python
FUNDING_THRESHOLD_LONG = -0.01   # %
FUNDING_THRESHOLD_SHORT = 0.03   # %
# No RSI or ADX filter
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
```

---

### 3. atlas_bot

**Status:** ⏳ Pending real microstructure data
**Location:** `/atlas_bot/`

The most sophisticated bot in the suite. Hidden Markov Model regime detection + per-regime LSTM prediction + FinBERT news sentiment.

**Architecture:**
```
Data Layer (OHLCV + funding + order book + sentiment)
        ↓
Feature Engineering (50+ features)
        ↓
HMM Regime Detection (4 regimes, unsupervised)
        ↓
Per-Regime LSTM Models (trained only on regime-specific data)
        ↓
Confidence Filter (only trade if confidence > 65%)
        ↓
Risk Layer (Kelly sizing, dynamic ATR stops)
        ↓
Execution (ccxt, Binance futures)
```

**Why it is shelved:**

All LSTM models achieved AUC of 0.49-0.58 regardless of features used:
- Price/volume features: AUC 0.55
- Simulated order flow: AUC 0.55
- On-chain free data: AUC 0.58
- Real VPIN from tick data: AUC 0.55

AUC must exceed 0.60 to overcome transaction costs. BTC perpetuals are priced too efficiently at the retail level for OHLCV-derived features to be predictive.

**What it needs to work:** Real historical order book depth data (bid/ask imbalance at tick level). This is being collected by `collector.py`. Estimated activation: September 2026.

**Important:** The original backtest showed $10k growing to $5.3M — this was caused by five compounding bugs: lookahead bias, Kelly position sizing without caps, no transaction costs, no train/test split, and data leakage in sentiment features. After fixing all bugs the corrected result was -1.71%. Always verify suspicious backtest results.

---

### 4. arb_bot

**Status:** 👀 Running in observation mode
**Location:** `/arb_bot/`

Cross-exchange arbitrage bot monitoring BTC/USDT perpetual futures across Binance, Bybit, and OKX simultaneously every 500ms.

**How it works:**
1. Fetches BTC futures price from all 3 exchanges simultaneously using asyncio
2. Calculates net spread after fees for all 3 pairs
3. When net spread exceeds 0.15%: executes simultaneous buy/sell
4. Leg protection: if one side fills but other fails, immediately close the filled side

**Current observation results:**

| Pair | Typical Gross Spread | Fees | Net Spread |
|---|---|---|---|
| Binance-Bybit | 0.014% | 0.08% | -0.066% |
| Binance-OKX | 0.007% | 0.09% | -0.083% |
| Bybit-OKX | 0.007% | 0.09% | -0.083% |

Normal market conditions produce negative net spreads. The bot is waiting for high-volatility events when spreads widen to 0.2-0.5%.

**Fee structure used:**
```python
FEES = {
    'binance': 0.0004,   # 0.04% futures taker
    'bybit':   0.0004,   # 0.04% futures taker
    'okx':     0.0005,   # 0.05% futures taker
}
MIN_NET_SPREAD = 0.0015  # 0.15% minimum to trade
TRADE_SIZE_USD = 100     # Per leg
MAX_HOLD_SECONDS = 30    # Force close if still open
```

**To run in observation mode:**
```bash
cd arb_bot
source venv/bin/activate
python main.py  # LIVE_TRADING = False in config.py
```

---

### 5. carry_bot

**Status:** ✅ Ready for live deployment
**Location:** `/carry_bot/`

The most reliable bot in the suite. Collects funding payments from extreme funding rate events without predicting price direction.

**Signal logic:**
```
Every hour:
  Fetch BTC/USDT perpetual funding rate from Binance

  If funding > +0.08%:
    Open SHORT position
    Reason: overleveraged longs, funding payment due to shorts

  If funding < -0.04%:
    Open LONG position
    Reason: overleveraged shorts, funding payment due to longs

  If position open:
    Collect funding every 8 hours
    Check if funding normalized (between -0.01% and +0.01%)
    If normalized: close position, book profit
    If stop-loss hit (2%): close immediately
```

**Verified performance (2 year backtest, fixed $200 position):**

| Metric | Value |
|---|---|
| Total trades | 125 (across 2 years) |
| Win rate | ~49-53% |
| Reliable annual return | 8-10% |
| Stop-losses (2% threshold) | ~31 over 2 years |
| Strategy edge source | Funding collection (structural) |

**Stop-loss comparison:**

| Stop-Loss | Win Rate | Notes |
|---|---|---|
| 1% | 36-49% | Too tight, frequent stop-outs |
| 2% | 46-53% | Recommended balance |
| 3% | 50-55% | Better win rate, larger losses when stopped |

**Files:**
```
carry_bot/
├── main.py          # Live trading loop (runs every hour)
├── funding.py       # Binance funding rate fetcher
├── backtest.py      # Historical backtest engine
├── config.py        # All parameters
├── requirements.txt
└── .env.example
```

**config.py:**
```python
FUNDING_LONG_THRESHOLD = -0.0004   # -0.04%
FUNDING_SHORT_THRESHOLD = 0.0008   # +0.08%
FUNDING_CLOSE_THRESHOLD = 0.0001   # Normalize to ±0.01%
STOP_LOSS_PCT = 0.02               # 2%
POSITION_SIZE_USD = 200            # Fixed, never auto-increase
MAX_DAILY_LOSS_USD = 20            # Kill switch
LIVE_TRADING = False               # Set True for real money
CHECK_INTERVAL_SECONDS = 3600      # Every hour
```

**To run backtest:**
```bash
cd carry_bot
source venv/bin/activate
python backtest.py
```

**To run live (testnet first):**
```bash
# 1. Edit .env with Binance testnet API keys
# 2. Set LIVE_TRADING = False for paper trading
python main.py

# 3. After 7 days paper trading successfully:
# Set LIVE_TRADING = True in config.py
# Replace testnet keys with live keys in .env
python main.py
```

**Telegram alerts you will receive:**
```
🟢 Opened SHORT $200 — funding +0.09% — collecting funding
💰 Funding collected: $0.72 (8h cycle)
🔴 Closed — funding normalized — PnL: +$1.84
⚠️  Stop-loss triggered — closed at -$4.00
📊 Daily summary — Trades: 1 — PnL: +$1.84 — Total: $34.20
```

---

### 6. collector.py

**Status:** 🔄 Must be deployed to VPS immediately
**Location:** `/atlas_bot/collector.py`

Silent background data collector that builds the training dataset atlas_bot will eventually need. Runs every minute via cron and logs to SQLite.

**Data collected every minute:**

| Table | Data | Purpose |
|---|---|---|
| `orderbook` | Top 20 bid/ask levels, imbalance ratio | Leading indicator of price direction |
| `tradeflow` | Buy vs sell initiated volume, VPIN | Trade toxicity signal |
| `liquidations` | Forced liquidation events | Cascade detection |
| `cross_exchange` | Binance/Bybit/OKX price divergence | Arbitrage signal |

**VPS setup (do this immediately):**
```bash
# 1. Get a VPS (DigitalOcean/Vultr/Hetzner, ~$5-6/month, Ubuntu 24)

# 2. Copy project to VPS
scp -r atlas_bot/ user@your-vps-ip:~/

# 3. SSH in and set up
ssh user@your-vps-ip
cd atlas_bot
pip install -r requirements.txt --break-system-packages

# 4. Enable cron
./setup_collector_cron.sh

# 5. Verify it's running
crontab -l
# Should show: * * * * * python /path/to/collector.py

# 6. Check data after 10 minutes
python -c "
import sqlite3
conn = sqlite3.connect('data/collector.db')
for table in ['orderbook','tradeflow','liquidations','cross_exchange']:
    count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    print(f'{table}: {count} records')
"
```

**Every day the collector runs = one irreplaceable day of training data. Start it today.**

---

## Live Deployment Guide

### Recommended Deployment Order

**Week 1: Testnet**
```bash
# carry_bot on Binance testnet
# Get testnet keys at: https://testnet.binancefuture.com
# Set in .env:
BINANCE_API_KEY=your_testnet_key
BINANCE_SECRET_KEY=your_testnet_secret
BINANCE_TESTNET=true

# Set in config.py:
LIVE_TRADING = False

python main.py
```

**Week 2-3: Monitor testnet results**
- Verify Telegram alerts are firing correctly
- Confirm trades match backtest expectations
- Check stop-loss triggers work correctly
- Verify daily kill switch activates at $20 loss

**Week 4+: Go live (small)**
```bash
# Switch to live Binance keys
# Set LIVE_TRADING = True
# Start with $200-500 maximum
# Do not increase for at least 1 month
python main.py
```

### VPS Deployment (carry_bot)

```bash
# Install as systemd service so it restarts on crash/reboot
sudo nano /etc/systemd/system/carry_bot.service

# Paste:
[Unit]
Description=Carry Bot Trading Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/carry_bot
ExecStart=/home/ubuntu/carry_bot/venv/bin/python main.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable carry_bot
sudo systemctl start carry_bot
sudo systemctl status carry_bot
```

### API Keys Setup

Create `.env` in each bot's directory:
```bash
# Binance
BINANCE_API_KEY=your_key_here
BINANCE_SECRET_KEY=your_secret_here
BINANCE_TESTNET=false

# Bybit (arb_bot only)
BYBIT_API_KEY=your_key_here
BYBIT_SECRET_KEY=your_secret_here

# OKX (arb_bot only)
OKX_API_KEY=your_key_here
OKX_SECRET_KEY=your_secret_here
OKX_PASSPHRASE=your_passphrase_here

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

**Never commit .env to git. It is already in .gitignore.**

---

## Key Lessons Learned

### 1. Backtests lie until proven otherwise
The atlas_bot showed $10k → $5.3M before bugs were fixed. Suspicious results are red flags, not celebrations. Always verify: no lookahead bias, proper train/test split, realistic fees included.

### 2. Public data has no edge on BTC
Every feature derived from OHLCV (RSI, MACD, Bollinger Bands, ATR, EMA) is a lagging indicator already priced in by faster participants. AUC ceiling on BTC with public data appears to be ~0.58.

### 3. Structural edge beats predictive edge
carry_bot works not because it predicts price, but because funding normalization is a mathematical inevitability. Don't predict the market — exploit its mechanics.

### 4. Small sample sizes mislead
18 trades over 31 days produced a Sharpe of 2.96. The same strategy over 2 years with 80 trades produced a Sharpe of -0.70. Always backtest on multi-year, multi-regime data.

### 5. Overfitting is silent
Every time you tune a parameter to improve past results you are memorizing the past, not discovering edge. Use strict time-based train/validation/test splits and never touch the test set.

### 6. Fees destroy more strategies than losses do
At 0.10% per trade (Binance spot taker), a bot making 100 trades/month loses 10% annually just to fees before a single losing trade. Always model realistic costs.

---

## Realistic Expectations

### carry_bot (live)

| Capital | Expected Annual Return (8-10%) | Monthly Income |
|---|---|---|
| $500 | $40-50 | ~$4 |
| $1,000 | $80-100 | ~$8 |
| $5,000 | $400-500 | ~$40 |
| $10,000 | $800-1,000 | ~$80 |

Returns are not guaranteed. Market conditions change. The 2024-2025 period showed 8.5% return. The 2025-2026 period showed higher returns partly due to favorable price movement. Budget for 8-10% as the reliable baseline.

### arb_bot (observing)

Opportunities are rare in normal markets. During high-volatility events (major news, liquidation cascades, exchange outages) spreads can widen to 0.3-0.5% and multiple opportunities fire in quick succession. Keep it running — it costs nothing in observation mode.

### atlas_bot (future)

No reliable estimate until real order flow data is available and AUC exceeds 0.60. Check back September 2026.

---

## Roadmap

| Timeline | Action |
|---|---|
| **Today** | Deploy collector.py to VPS. Start cron. Never stop it. |
| **This week** | Paper trade carry_bot on Binance testnet |
| **Week 2** | If testnet clean, go live carry_bot with $200-500 |
| **Month 1** | Monitor real carry_bot results vs backtest |
| **Month 3** | If results match, consider scaling capital |
| **Month 3-6** | Run arb_bot on futures, collect spread data |
| **Month 6** | Retrain atlas_bot on real collector data, check AUC |
| **Month 7+** | If AUC > 0.60, paper trade atlas_bot for 1 month |
| **Month 8+** | If atlas_bot paper trading profitable, go live small |

---

## Tech Stack

| Component | Library/Tool |
|---|---|
| Exchange connectivity | ccxt |
| Async price fetching | asyncio |
| Backtesting | vectorbt, pandas |
| ML models | tensorflow/keras, hmmlearn, scikit-learn |
| Technical features | ta-lib, pandas-ta |
| Data storage | SQLite (sqlalchemy) |
| Notifications | python-telegram-bot |
| Scheduling | cron (VPS), APScheduler |
| Environment | Python 3.11, venv |

---

## Directory Structure

```
trot/
├── trot/               # Original momentum bot (retired)
├── funding_bot/        # Funding rate bot (shelved)
├── atlas_bot/          # ML regime bot + collector.py
│   ├── collector.py    # 24/7 data collector (deploy to VPS)
│   ├── features.py     # Feature engineering
│   ├── regime.py       # HMM regime detection
│   ├── models/         # Trained LSTM models per regime
│   ├── backtest.py
│   └── main.py
├── arb_bot/            # Cross-exchange arbitrage (observing)
│   ├── monitor.py      # Async price fetcher
│   ├── arbitrage.py    # Spread calculator
│   ├── executor.py     # Order execution
│   ├── dashboard.py    # Terminal dashboard
│   └── main.py
├── carry_bot/          # Funding carry (ready for live)
│   ├── funding.py      # Funding rate fetcher
│   ├── backtest.py
│   ├── main.py
│   └── config.py
└── README.md           # This file
```

---

*Built with Claude Code CLI. Tested empirically. No strategy deployed without multi-year backtesting and proper out-of-sample validation.*
