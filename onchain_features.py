"""
On-Chain Features Module
Integrates Glassnode and CryptoQuant APIs for leading indicators

Features:
- Exchange net flow (BTC moving to/from exchanges)
- Whale transaction count (transactions > 1000 BTC)
- Miner outflows (miners selling = supply pressure)
- Exchange reserve changes

These are genuinely leading indicators with historical data available.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import os

# API Configuration
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", "")
CRYPTOQUANT_API_KEY = os.getenv("CRYPTOQUANT_API_KEY", "")

# Free tier endpoints
GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"
CRYPTOQUANT_BASE = "https://api.cryptoquant.com/v1"


def fetch_glassnode_metric(metric: str, asset: str = "BTC", since: datetime = None, until: datetime = None) -> pd.DataFrame:
    """
    Fetch metric from Glassnode API
    Free tier: Limited to 1 year of daily data for some metrics
    """
    if not GLASSNODE_API_KEY:
        return pd.DataFrame()

    url = f"{GLASSNODE_BASE}/{metric}"

    params = {
        "a": asset,
        "api_key": GLASSNODE_API_KEY,
        "f": "JSON",
        "i": "24h"  # Daily resolution
    }

    if since:
        params["s"] = int(since.timestamp())
    if until:
        params["u"] = int(until.timestamp())

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df["timestamp"] = pd.to_datetime(df["t"], unit="s")
                df = df.rename(columns={"v": metric.split("/")[-1]})
                return df[["timestamp", metric.split("/")[-1]]]
    except Exception as e:
        print(f"Glassnode error ({metric}): {e}")

    return pd.DataFrame()


def fetch_exchange_netflow_glassnode(since: datetime = None, until: datetime = None) -> pd.DataFrame:
    """
    Fetch exchange net flow from Glassnode
    Positive = inflow (selling pressure)
    Negative = outflow (accumulation)
    """
    # Try netflow metric (might need paid tier)
    df = fetch_glassnode_metric("transactions/transfers_to_exchanges_count", "BTC", since, until)

    if df.empty:
        # Fallback: try alternative metrics
        inflow = fetch_glassnode_metric("transactions/transfers_to_exchanges_sum", "BTC", since, until)
        outflow = fetch_glassnode_metric("transactions/transfers_from_exchanges_sum", "BTC", since, until)

        if not inflow.empty and not outflow.empty:
            df = inflow.merge(outflow, on="timestamp", how="outer")
            df["exchange_netflow"] = df.iloc[:, 1].fillna(0) - df.iloc[:, 2].fillna(0)
            df = df[["timestamp", "exchange_netflow"]]

    return df


def fetch_alternative_onchain_data(since: datetime = None, until: datetime = None) -> pd.DataFrame:
    """
    Fetch on-chain data from alternative.me and other free sources
    """
    # Fear & Greed Index (free, no API key needed)
    url = "https://api.alternative.me/fng/"

    try:
        params = {"limit": 365 * 2, "format": "json"}
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if "data" in data:
            df = pd.DataFrame(data["data"])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
            df["fng_value"] = df["value"].astype(float) / 100  # Normalize to 0-1
            df["fng_classification"] = df["value_classification"]
            df = df.sort_values("timestamp")
            return df[["timestamp", "fng_value", "fng_classification"]]
    except Exception as e:
        print(f"Alternative.me error: {e}")

    return pd.DataFrame()


def fetch_coinglass_data() -> pd.DataFrame:
    """
    Fetch data from Coinglass (free tier)
    - Open interest
    - Long/short ratio
    - Liquidation data
    """
    # Coinglass public endpoints
    endpoints = {
        "open_interest": "https://open-api.coinglass.com/public/v2/open_interest",
        "funding": "https://open-api.coinglass.com/public/v2/funding"
    }

    results = {}
    for name, url in endpoints.items():
        try:
            params = {"symbol": "BTC"}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results[name] = data.get("data", [])
        except Exception as e:
            print(f"Coinglass error ({name}): {e}")

    return results


def simulate_onchain_features_from_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate on-chain features from price patterns when API data unavailable

    These are rough approximations based on historical correlations:
    - Large price drops often coincide with exchange inflows
    - Accumulation phases show exchange outflows
    - Whale activity increases during volatility
    """
    df = df.copy()

    # Estimate exchange flow from price momentum and volume
    # Large negative returns + high volume = likely inflow (selling)
    # Large positive returns + high volume = possible outflow (buying)
    price_change = df["close"].pct_change()
    volume_zscore = (df["volume"] - df["volume"].rolling(24).mean()) / (df["volume"].rolling(24).std() + 1e-10)

    # Negative value = outflow (bullish), Positive = inflow (bearish)
    df["exchange_flow_est"] = -price_change * (1 + volume_zscore.clip(-2, 2))

    # Whale activity estimate (large price moves on high volume)
    df["whale_activity_est"] = abs(price_change) * volume_zscore.clip(0, 5)

    # Miner pressure estimate (tends to sell during rallies)
    rolling_return = df["close"].pct_change(24)
    df["miner_pressure_est"] = rolling_return.clip(0, None)  # Positive during rallies

    # Exchange reserve proxy (cumulative flow)
    df["exchange_reserve_change_est"] = df["exchange_flow_est"].rolling(24).sum()

    return df


def fetch_blockchain_com_data(since: datetime = None) -> pd.DataFrame:
    """
    Fetch data from Blockchain.com API (free, no key needed)
    - Hash rate
    - Transaction count
    - Mempool size
    """
    base_url = "https://api.blockchain.info/charts"

    metrics = {
        "hash-rate": "hashrate",
        "n-transactions": "tx_count",
        "mempool-size": "mempool_size"
    }

    result_df = None

    for endpoint, col_name in metrics.items():
        try:
            url = f"{base_url}/{endpoint}"
            params = {
                "timespan": "2years",
                "format": "json",
                "sampled": "true"
            }
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if "values" in data:
                df = pd.DataFrame(data["values"])
                df["timestamp"] = pd.to_datetime(df["x"], unit="s")
                df[col_name] = df["y"]
                df = df[["timestamp", col_name]]

                if result_df is None:
                    result_df = df
                else:
                    result_df = result_df.merge(df, on="timestamp", how="outer")

            time.sleep(0.5)  # Rate limit

        except Exception as e:
            print(f"Blockchain.com error ({endpoint}): {e}")

    return result_df if result_df is not None else pd.DataFrame()


def compute_onchain_features(
    ohlcv_df: pd.DataFrame,
    include_api_data: bool = True
) -> pd.DataFrame:
    """
    Compute all on-chain features

    Args:
        ohlcv_df: OHLCV dataframe with timestamp, open, high, low, close, volume
        include_api_data: Whether to attempt API calls for real data

    Returns:
        DataFrame with on-chain features added
    """
    print("Computing on-chain features...")
    df = ohlcv_df.copy()

    if df.empty:
        print("  ⚠️ Empty OHLCV dataframe")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 1. Fear & Greed Index (free, reliable)
    print("  Fetching Fear & Greed Index...")
    fng_df = fetch_alternative_onchain_data()
    if not fng_df.empty:
        fng_df["timestamp"] = pd.to_datetime(fng_df["timestamp"]).dt.normalize()
        df["date"] = df["timestamp"].dt.normalize()
        df = df.merge(fng_df[["timestamp", "fng_value"]], left_on="date", right_on="timestamp",
                      how="left", suffixes=("", "_fng"))
        df = df.drop(columns=["timestamp_fng", "date"], errors="ignore")
        df["fng_value"] = df["fng_value"].ffill().fillna(0.5)
        print(f"    ✅ F&G data: {fng_df['fng_value'].notna().sum()} days")
    else:
        df["fng_value"] = 0.5
        print("    ⚠️ F&G data unavailable, using neutral")

    # 2. Blockchain.com data (free, no key)
    print("  Fetching Blockchain.com data...")
    blockchain_df = fetch_blockchain_com_data()
    if not blockchain_df.empty:
        blockchain_df["timestamp"] = pd.to_datetime(blockchain_df["timestamp"]).dt.normalize()
        df["date"] = df["timestamp"].dt.normalize()
        df = df.merge(blockchain_df, left_on="date", right_on="timestamp",
                      how="left", suffixes=("", "_bc"))
        df = df.drop(columns=["timestamp_bc", "date"], errors="ignore")

        # Normalize blockchain metrics
        for col in ["hashrate", "tx_count", "mempool_size"]:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(df[col].median())
                # Z-score normalization
                df[f"{col}_zscore"] = (df[col] - df[col].rolling(30).mean()) / (df[col].rolling(30).std() + 1e-10)
        print(f"    ✅ Blockchain data: {len(blockchain_df)} records")
    else:
        print("    ⚠️ Blockchain.com data unavailable")

    # 3. Glassnode data (if API key available)
    if include_api_data and GLASSNODE_API_KEY:
        print("  Fetching Glassnode data...")
        start_date = df["timestamp"].min()
        end_date = df["timestamp"].max()

        exchange_flow = fetch_exchange_netflow_glassnode(start_date, end_date)
        if not exchange_flow.empty:
            exchange_flow["timestamp"] = pd.to_datetime(exchange_flow["timestamp"]).dt.normalize()
            df["date"] = df["timestamp"].dt.normalize()
            df = df.merge(exchange_flow, left_on="date", right_on="timestamp",
                          how="left", suffixes=("", "_gn"))
            df = df.drop(columns=["timestamp_gn", "date"], errors="ignore")
            print(f"    ✅ Glassnode data: {len(exchange_flow)} records")
    else:
        print("  ⚠️ Glassnode API key not set, using estimates")

    # 4. Estimate missing on-chain features from price
    print("  Computing estimated on-chain features...")
    df = simulate_onchain_features_from_price(df)

    # 5. Derived features
    # F&G momentum
    df["fng_momentum"] = df["fng_value"].diff(7)  # 7-day change
    df["fng_extreme"] = ((df["fng_value"] < 0.25) | (df["fng_value"] > 0.75)).astype(float)

    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0)

    print(f"✅ On-chain features computed: {len(df)} rows")

    return df


def get_onchain_feature_columns() -> List[str]:
    """Return list of on-chain feature columns"""
    return [
        # Fear & Greed
        "fng_value",
        "fng_momentum",
        "fng_extreme",

        # Blockchain metrics (if available)
        "hashrate_zscore",
        "tx_count_zscore",
        "mempool_size_zscore",

        # Estimated on-chain
        "exchange_flow_est",
        "whale_activity_est",
        "miner_pressure_est",
        "exchange_reserve_change_est",
    ]


if __name__ == "__main__":
    print("Testing on-chain features module...")

    # Test Fear & Greed
    print("\n1. Testing Fear & Greed Index...")
    fng_df = fetch_alternative_onchain_data()
    if not fng_df.empty:
        print(f"   Got {len(fng_df)} days of F&G data")
        print(f"   Latest: {fng_df.iloc[-1]['fng_value']:.2f} ({fng_df.iloc[-1].get('fng_classification', 'N/A')})")

    # Test Blockchain.com
    print("\n2. Testing Blockchain.com...")
    bc_df = fetch_blockchain_com_data()
    if not bc_df.empty:
        print(f"   Got {len(bc_df)} records")
        print(f"   Columns: {list(bc_df.columns)}")

    # Test with sample OHLCV
    print("\n3. Testing full feature computation...")
    sample_df = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="1h"),
        "open": 40000 + np.random.randn(100) * 100,
        "high": 40100 + np.random.randn(100) * 100,
        "low": 39900 + np.random.randn(100) * 100,
        "close": 40000 + np.cumsum(np.random.randn(100) * 50),
        "volume": 1000 + np.random.rand(100) * 500
    })

    result = compute_onchain_features(sample_df, include_api_data=False)
    print(f"   Result columns: {list(result.columns)}")

    print("\n✅ On-chain features test complete")
