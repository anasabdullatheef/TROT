"""
Funding Rate Monitor
Fetches current and historical funding rates from Binance
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

import ccxt.async_support as ccxt

import config


class FundingMonitor:
    """Monitors Binance perpetual funding rates"""

    def __init__(self):
        self.exchange: Optional[ccxt.binance] = None
        self.last_funding_rate: float = 0.0
        self.last_funding_time: Optional[datetime] = None

    async def initialize(self):
        """Initialize exchange connection"""
        self.exchange = ccxt.binance({
            "apiKey": config.BINANCE_API_KEY,
            "secret": config.BINANCE_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        await self.exchange.load_markets()
        print("Funding monitor initialized")

    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()

    async def get_current_funding_rate(self) -> Dict:
        """Get current funding rate for BTC/USDT perpetual"""
        try:
            # Fetch funding rate
            funding = await self.exchange.fetch_funding_rate(config.SYMBOL)

            self.last_funding_rate = funding.get("fundingRate", 0)
            self.last_funding_time = datetime.now()

            return {
                "rate": self.last_funding_rate,
                "rate_pct": self.last_funding_rate * 100,
                "next_funding_time": funding.get("fundingTimestamp"),
                "timestamp": datetime.now()
            }
        except Exception as e:
            print(f"Error fetching funding rate: {e}")
            return {"rate": 0, "rate_pct": 0, "timestamp": datetime.now()}

    async def get_current_price(self) -> float:
        """Get current BTC price"""
        try:
            ticker = await self.exchange.fetch_ticker(config.SYMBOL)
            return ticker.get("last", 0)
        except Exception as e:
            print(f"Error fetching price: {e}")
            return 0


def fetch_historical_funding_sync(days: int = 365) -> pd.DataFrame:
    """
    Fetch historical funding rates from Binance (synchronous version for backtest)
    Uses REST API directly
    """
    import requests
    from datetime import datetime, timedelta

    print(f"Fetching {days} days of historical funding rates...")

    # Binance funding rate endpoint
    url = "https://fapi.binance.com/fapi/v1/fundingRate"

    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Binance returns max 1000 records per request
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": "BTCUSDT",
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if not data:
                break

            all_data.extend(data)

            # Move start time to after last record
            last_time = data[-1]["fundingTime"]
            current_start = last_time + 1

            print(f"  Fetched {len(all_data)} funding records...")

        except Exception as e:
            print(f"Error: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df.rename(columns={"fundingTime": "timestamp", "fundingRate": "rate"})
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Total: {len(df)} funding rate records")
    print(f"Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df[["timestamp", "rate"]]


def fetch_historical_prices_sync(days: int = 365) -> pd.DataFrame:
    """Fetch historical hourly BTC prices for stop loss simulation"""
    import requests

    print(f"Fetching {days} days of hourly price data...")

    url = "https://fapi.binance.com/fapi/v1/klines"

    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1500
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if not data:
                break

            all_data.extend(data)

            # Move start time
            last_time = data[-1][0]
            current_start = last_time + 1

            if len(all_data) % 3000 == 0:
                print(f"  Fetched {len(all_data)} price records...")

        except Exception as e:
            print(f"Error: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    print(f"Total: {len(df)} hourly price records")

    return df[["timestamp", "open", "high", "low", "close"]]


async def test_funding():
    """Test funding rate fetching"""
    monitor = FundingMonitor()
    await monitor.initialize()

    funding = await monitor.get_current_funding_rate()
    price = await monitor.get_current_price()

    print(f"\nCurrent BTC Price: ${price:,.2f}")
    print(f"Current Funding Rate: {funding['rate_pct']:.4f}%")

    # Determine signal
    if funding['rate'] >= config.FUNDING_SHORT_THRESHOLD:
        print(f"SIGNAL: GO SHORT (funding {funding['rate_pct']:.4f}% > {config.FUNDING_SHORT_THRESHOLD*100:.2f}%)")
    elif funding['rate'] <= config.FUNDING_LONG_THRESHOLD:
        print(f"SIGNAL: GO LONG (funding {funding['rate_pct']:.4f}% < {config.FUNDING_LONG_THRESHOLD*100:.2f}%)")
    else:
        print("SIGNAL: NO POSITION (funding in normal range)")

    await monitor.close()


if __name__ == "__main__":
    asyncio.run(test_funding())
