"""
Funding Rate Module
Fetches and processes funding rate data from Binance
Adapted from funding_bot
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
import config


def get_current_funding_rate() -> Optional[float]:
    """Fetch the current/latest funding rate for BTC/USDT perpetual"""
    url = f"{config.BINANCE_BASE_URL}/fapi/v1/premiumIndex"
    params = {"symbol": config.BINANCE_SYMBOL}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data["lastFundingRate"])
    except Exception as e:
        print(f"Error fetching current funding rate: {e}")
        return None


def get_historical_funding_rates(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch historical funding rates from Binance
    Returns DataFrame with columns: timestamp, funding_rate
    """
    url = f"{config.BINANCE_BASE_URL}/fapi/v1/fundingRate"

    all_rates = []
    current_end = end_time or datetime.now()

    if start_time is None:
        start_time = current_end - timedelta(days=365)

    current_start = start_time

    while current_start < current_end:
        params = {
            "symbol": config.BINANCE_SYMBOL,
            "limit": limit,
            "startTime": int(current_start.timestamp() * 1000),
            "endTime": int(current_end.timestamp() * 1000)
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_rates.extend(data)

            latest_ts = max(int(d["fundingTime"]) for d in data)
            current_start = datetime.fromtimestamp(latest_ts / 1000) + timedelta(seconds=1)

            time.sleep(config.RATE_LIMIT_DELAY)

            if len(data) < limit:
                break

        except Exception as e:
            print(f"Error fetching historical funding rates: {e}")
            break

    if not all_rates:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame(all_rates)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]]
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset=["timestamp"])

    return df


def compute_funding_features(df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute funding-related features:
    - Current funding rate
    - 8h change
    - 24h cumulative
    """
    df = df.copy()

    # Merge funding rates with OHLCV
    df = pd.merge_asof(
        df.sort_values("timestamp"),
        funding_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("8h")
    )

    # Forward fill funding rates
    df["funding_rate"] = df["funding_rate"].ffill()

    # 8h change (previous funding rate)
    df["funding_8h_change"] = df["funding_rate"].diff(periods=8)

    # 24h cumulative (3 funding periods)
    df["funding_24h_cumulative"] = df["funding_rate"].rolling(window=24, min_periods=1).sum()

    return df


def classify_funding_signal(funding_rate: float) -> Optional[str]:
    """
    Classify funding rate into trading signal (Config B style)

    Returns:
        'short' - funding extremely positive (longs overleveraged)
        'long' - funding extremely negative (shorts overleveraged)
        None - no signal
    """
    if funding_rate >= config.FUNDING_LONG_THRESHOLD:
        return "short"
    elif funding_rate <= config.FUNDING_SHORT_THRESHOLD:
        return "long"
    return None


def get_funding_stats(df: pd.DataFrame) -> Dict:
    """Calculate statistics on funding rate data"""
    if df.empty:
        return {}

    return {
        "mean": df["funding_rate"].mean(),
        "std": df["funding_rate"].std(),
        "min": df["funding_rate"].min(),
        "max": df["funding_rate"].max(),
        "positive_pct": (df["funding_rate"] > 0).mean() * 100,
        "extreme_positive_pct": (df["funding_rate"] >= config.FUNDING_LONG_THRESHOLD).mean() * 100,
        "extreme_negative_pct": (df["funding_rate"] <= config.FUNDING_SHORT_THRESHOLD).mean() * 100,
        "count": len(df)
    }


if __name__ == "__main__":
    print("Testing funding module...")

    rate = get_current_funding_rate()
    if rate:
        print(f"Current funding rate: {rate*100:.4f}%")
        signal = classify_funding_signal(rate)
        print(f"Signal: {signal}")

    print("\nFetching 30 days of historical rates...")
    hist_df = get_historical_funding_rates(
        start_time=datetime.now() - timedelta(days=30)
    )
    print(f"Fetched {len(hist_df)} records")

    if not hist_df.empty:
        stats = get_funding_stats(hist_df)
        print(f"\nStats:")
        print(f"  Mean: {stats['mean']*100:.4f}%")
        print(f"  Std: {stats['std']*100:.4f}%")
