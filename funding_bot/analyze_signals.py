"""
Analyze signal distribution to understand why we get few trades
"""

import pandas as pd
from datetime import datetime, timedelta
import config
from funding import get_historical_funding_rates
from backtest import fetch_historical_ohlcv
from strategy import calculate_indicators

# Fetch data
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

print("Fetching OHLCV data...")
ohlcv_df = fetch_historical_ohlcv(start_date=start_date, end_date=end_date)
print(f"OHLCV: {len(ohlcv_df)} candles from {ohlcv_df['timestamp'].min()} to {ohlcv_df['timestamp'].max()}")

print("\nFetching funding data...")
funding_df = get_historical_funding_rates(start_time=start_date, end_time=end_date)
print(f"Funding: {len(funding_df)} records from {funding_df['timestamp'].min()} to {funding_df['timestamp'].max()}")

# Calculate indicators
ohlcv_df = calculate_indicators(ohlcv_df)

# Merge funding with OHLCV
df = pd.merge_asof(
    ohlcv_df.sort_values("timestamp"),
    funding_df.sort_values("timestamp"),
    on="timestamp",
    direction="backward",
    tolerance=pd.Timedelta("8h")
)
df["funding_rate"] = df["funding_rate"].ffill()

# Filter out rows with missing data or indicator warmup (ADX < 1)
df_valid = df.dropna(subset=["funding_rate", "rsi", "adx"])
df_valid = df_valid[df_valid["adx"] >= 1]  # Skip warmup period
print(f"\nValid rows with all data (after warmup): {len(df_valid)}")

# Analyze funding rate distribution
print("\n=== FUNDING RATE ANALYSIS ===")
print(f"Mean: {df_valid['funding_rate'].mean()*100:.4f}%")
print(f"Std: {df_valid['funding_rate'].std()*100:.4f}%")
print(f"Min: {df_valid['funding_rate'].min()*100:.4f}%")
print(f"Max: {df_valid['funding_rate'].max()*100:.4f}%")

# Count extreme funding
extreme_positive = (df_valid["funding_rate"] >= config.FUNDING_LONG_THRESHOLD)
extreme_negative = (df_valid["funding_rate"] <= config.FUNDING_SHORT_THRESHOLD)
print(f"\nExtreme positive (>={config.FUNDING_LONG_THRESHOLD*100:.2f}%): {extreme_positive.sum()} ({extreme_positive.mean()*100:.1f}%)")
print(f"Extreme negative (<={config.FUNDING_SHORT_THRESHOLD*100:.2f}%): {extreme_negative.sum()} ({extreme_negative.mean()*100:.1f}%)")

# Analyze RSI when funding is extreme
print("\n=== RSI WHEN FUNDING IS EXTREME ===")
extreme_pos_df = df_valid[extreme_positive]
extreme_neg_df = df_valid[extreme_negative]

if len(extreme_pos_df) > 0:
    rsi_overbought = (extreme_pos_df["rsi"] >= config.RSI_OVERBOUGHT)
    print(f"When funding is extreme positive:")
    print(f"  RSI mean: {extreme_pos_df['rsi'].mean():.1f}")
    print(f"  RSI >= {config.RSI_OVERBOUGHT}: {rsi_overbought.sum()} ({rsi_overbought.mean()*100:.1f}%)")

if len(extreme_neg_df) > 0:
    rsi_oversold = (extreme_neg_df["rsi"] <= config.RSI_OVERSOLD)
    print(f"When funding is extreme negative:")
    print(f"  RSI mean: {extreme_neg_df['rsi'].mean():.1f}")
    print(f"  RSI <= {config.RSI_OVERSOLD}: {rsi_oversold.sum()} ({rsi_oversold.mean()*100:.1f}%)")

# Analyze ADX
print("\n=== ADX ANALYSIS ===")
print(f"ADX mean: {df_valid['adx'].mean():.1f}")
print(f"ADX < {config.ADX_THRESHOLD} (ranging): {(df_valid['adx'] < config.ADX_THRESHOLD).sum()} ({(df_valid['adx'] < config.ADX_THRESHOLD).mean()*100:.1f}%)")

# Count full signal conditions
print("\n=== FULL SIGNAL ANALYSIS ===")
short_signals = (
    (df_valid["funding_rate"] >= config.FUNDING_LONG_THRESHOLD) &
    (df_valid["rsi"] >= config.RSI_OVERBOUGHT) &
    (df_valid["adx"] < config.ADX_THRESHOLD)
)
long_signals = (
    (df_valid["funding_rate"] <= config.FUNDING_SHORT_THRESHOLD) &
    (df_valid["rsi"] <= config.RSI_OVERSOLD) &
    (df_valid["adx"] < config.ADX_THRESHOLD)
)

print(f"Short signals (all conditions met): {short_signals.sum()}")
print(f"Long signals (all conditions met): {long_signals.sum()}")
print(f"Total potential signals: {short_signals.sum() + long_signals.sum()}")

# Show some signal examples
if short_signals.sum() > 0:
    print("\nFirst 5 short signal candles:")
    print(df_valid[short_signals][["timestamp", "close", "funding_rate", "rsi", "adx"]].head())

if long_signals.sum() > 0:
    print("\nFirst 5 long signal candles:")
    print(df_valid[long_signals][["timestamp", "close", "funding_rate", "rsi", "adx"]].head())
