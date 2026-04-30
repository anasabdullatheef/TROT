"""
Funding Rate Module
Fetches and parses funding rate data from Binance Futures API
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import config


def get_current_funding_rate() -> Optional[float]:
    """
    Fetch the current/latest funding rate for BTC/USDT perpetual
    Returns the funding rate as a decimal (e.g., 0.0001 = 0.01%)
    """
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


def get_next_funding_time() -> Optional[datetime]:
    """
    Get the timestamp of the next funding rate payment
    """
    url = f"{config.BINANCE_BASE_URL}/fapi/v1/premiumIndex"
    params = {"symbol": config.BINANCE_SYMBOL}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        next_funding_ts = int(data["nextFundingTime"])
        return datetime.fromtimestamp(next_funding_ts / 1000)
    except Exception as e:
        print(f"Error fetching next funding time: {e}")
        return None


def get_historical_funding_rates(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch historical funding rates from Binance
    The API returns up to 1000 records per request
    Funding rates are recorded every 8 hours

    Returns DataFrame with columns: timestamp, funding_rate
    """
    url = f"{config.BINANCE_BASE_URL}/fapi/v1/fundingRate"

    all_rates = []
    current_end = end_time or datetime.now()

    # If no start_time, go back 1 year (API limit)
    if start_time is None:
        start_time = current_end - timedelta(days=365)

    # Use startTime and paginate forward for more reliable results
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

            # Move start time to after the latest record we got
            latest_ts = max(int(d["fundingTime"]) for d in data)
            current_start = datetime.fromtimestamp(latest_ts / 1000) + timedelta(seconds=1)

            # Rate limiting
            time.sleep(config.RATE_LIMIT_DELAY)

            print(f"Fetched {len(data)} funding rates, total: {len(all_rates)}")

            # If we got fewer than limit, we've reached the end
            if len(data) < limit:
                break

        except Exception as e:
            print(f"Error fetching historical funding rates: {e}")
            break

    if not all_rates:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    # Convert to DataFrame
    df = pd.DataFrame(all_rates)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]]
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset=["timestamp"])

    return df


def classify_funding_signal(funding_rate: float) -> str:
    """
    Classify the funding rate into a trading signal direction

    Returns:
        'short' - funding is extremely positive (longs overleveraged)
        'long' - funding is extremely negative (shorts overleveraged)
        'neutral' - funding in normal range
        'exit' - funding has normalized (for position exit)
    """
    if funding_rate >= config.FUNDING_LONG_THRESHOLD:
        return "short"  # Too many longs, expect reversal down
    elif funding_rate <= config.FUNDING_SHORT_THRESHOLD:
        return "long"   # Too many shorts, expect reversal up
    elif config.FUNDING_NEUTRAL_LOW <= funding_rate <= config.FUNDING_NEUTRAL_HIGH:
        return "exit"   # Funding normalized, exit positions
    else:
        return "neutral"


def get_funding_stats(df: pd.DataFrame) -> Dict:
    """
    Calculate statistics on funding rate data
    """
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
    # Test the module
    print("Testing funding rate module...")

    # Get current rate
    rate = get_current_funding_rate()
    print(f"Current funding rate: {rate:.6f} ({rate*100:.4f}%)")

    signal = classify_funding_signal(rate)
    print(f"Signal classification: {signal}")

    # Get next funding time
    next_time = get_next_funding_time()
    print(f"Next funding time: {next_time}")

    # Get some historical data
    print("\nFetching recent historical rates...")
    hist_df = get_historical_funding_rates(
        start_time=datetime.now() - timedelta(days=30)
    )
    print(f"Fetched {len(hist_df)} historical rates")

    if not hist_df.empty:
        stats = get_funding_stats(hist_df)
        print(f"\nFunding Stats (last 30 days):")
        print(f"  Mean: {stats['mean']*100:.4f}%")
        print(f"  Std: {stats['std']*100:.4f}%")
        print(f"  Min: {stats['min']*100:.4f}%")
        print(f"  Max: {stats['max']*100:.4f}%")
        print(f"  Positive: {stats['positive_pct']:.1f}%")
        print(f"  Extreme positive (>{config.FUNDING_LONG_THRESHOLD*100}%): {stats['extreme_positive_pct']:.1f}%")
        print(f"  Extreme negative (<{config.FUNDING_SHORT_THRESHOLD*100}%): {stats['extreme_negative_pct']:.1f}%")
