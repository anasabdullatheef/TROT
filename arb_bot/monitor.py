"""
Price Monitor - Fetches prices from all 3 exchanges simultaneously using asyncio
Speed is critical for arbitrage
"""

import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

import ccxt.async_support as ccxt

import config


@dataclass
class PriceData:
    """Price data from an exchange"""
    exchange: str
    symbol: str
    bid: float
    ask: float
    mid: float
    timestamp: datetime
    latency_ms: float


class PriceMonitor:
    """Monitors prices across multiple exchanges simultaneously"""

    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.last_prices: Dict[str, PriceData] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize exchange connections"""
        print("Initializing exchange connections...")

        # Binance
        self.exchanges["binance"] = ccxt.binance({
            "apiKey": config.BINANCE_API_KEY,
            "secret": config.BINANCE_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        })

        # Bybit
        self.exchanges["bybit"] = ccxt.bybit({
            "apiKey": config.BYBIT_API_KEY,
            "secret": config.BYBIT_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        })

        # OKX
        self.exchanges["okx"] = ccxt.okx({
            "apiKey": config.OKX_API_KEY,
            "secret": config.OKX_API_SECRET,
            "password": config.OKX_PASSPHRASE,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        })

        # Load markets
        tasks = [ex.load_markets() for ex in self.exchanges.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

        self._initialized = True
        print(f"Connected to {len(self.exchanges)} exchanges")

    async def close(self):
        """Close all exchange connections"""
        tasks = [ex.close() for ex in self.exchanges.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def fetch_price(self, exchange_name: str) -> Optional[PriceData]:
        """Fetch price from a single exchange"""
        if exchange_name not in self.exchanges:
            return None

        exchange = self.exchanges[exchange_name]
        start_time = time.time()

        try:
            # Fetch order book (top of book for best bid/ask)
            orderbook = await exchange.fetch_order_book(config.SYMBOL, limit=5)

            latency_ms = (time.time() - start_time) * 1000

            if not orderbook["bids"] or not orderbook["asks"]:
                return None

            bid = orderbook["bids"][0][0]
            ask = orderbook["asks"][0][0]
            mid = (bid + ask) / 2

            price_data = PriceData(
                exchange=exchange_name,
                symbol=config.SYMBOL,
                bid=bid,
                ask=ask,
                mid=mid,
                timestamp=datetime.now(),
                latency_ms=latency_ms
            )

            self.last_prices[exchange_name] = price_data
            return price_data

        except Exception as e:
            print(f"Error fetching price from {exchange_name}: {e}")
            return None

    async def fetch_all_prices(self) -> Dict[str, PriceData]:
        """Fetch prices from all exchanges simultaneously"""
        tasks = [self.fetch_price(ex) for ex in config.EXCHANGES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices = {}
        for ex, result in zip(config.EXCHANGES, results):
            if isinstance(result, PriceData):
                prices[ex] = result
            elif isinstance(result, Exception):
                print(f"Exception fetching {ex}: {result}")

        return prices

    def get_last_prices(self) -> Dict[str, PriceData]:
        """Get last fetched prices"""
        return self.last_prices.copy()


async def test_monitor():
    """Test the price monitor"""
    monitor = PriceMonitor()
    await monitor.initialize()

    print("\nFetching prices from all exchanges...")
    for _ in range(5):
        prices = await monitor.fetch_all_prices()

        print(f"\n{datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        for ex, price in prices.items():
            print(f"  {ex:10s}: Bid={price.bid:,.2f} Ask={price.ask:,.2f} "
                  f"Spread={((price.ask-price.bid)/price.mid)*100:.4f}% "
                  f"Latency={price.latency_ms:.0f}ms")

        await asyncio.sleep(0.5)

    await monitor.close()


if __name__ == "__main__":
    asyncio.run(test_monitor())
