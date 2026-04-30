#!/usr/bin/env python3
"""
Carry Bot - Funding Rate Collection Strategy

Simple strategy:
- When funding rate >= +0.01%: Go SHORT to collect funding
- When funding rate <= -0.01%: Go LONG to collect funding
- Exit when funding normalizes (between -0.006% and +0.006%)
- Stop loss: 2% adverse price movement

No ML, no prediction - just collect funding when rates are extreme.
"""

import asyncio
import os
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

import ccxt.async_support as ccxt

import config


@dataclass
class Position:
    """Current position state"""
    direction: str  # "long" or "short"
    entry_price: float
    entry_time: datetime
    size_usd: float
    size_btc: float
    funding_collected: float = 0.0
    funding_payments: int = 0


class CarryBot:
    """Funding rate carry trading bot"""

    def __init__(self):
        self.exchange: Optional[ccxt.binance] = None
        self.position: Optional[Position] = None
        self.running = False

        # Strategy parameters
        self.short_threshold = 0.0001   # +0.01%
        self.long_threshold = -0.0001   # -0.01%
        self.exit_high = 0.00006        # +0.006%
        self.exit_low = -0.00006        # -0.006%
        self.stop_loss_pct = config.STOP_LOSS_PCT
        self.position_size = config.POSITION_SIZE_USD

    async def initialize(self):
        """Initialize exchange connection"""
        print("Initializing Carry Bot...")

        self.exchange = ccxt.binance({
            "apiKey": config.BINANCE_API_KEY,
            "secret": config.BINANCE_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })

        await self.exchange.load_markets()
        print("Connected to Binance Futures")

        # Check for existing position
        await self._check_existing_position()

    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()

    async def _check_existing_position(self):
        """Check if we have an existing position on the exchange"""
        try:
            positions = await self.exchange.fetch_positions([config.SYMBOL])
            for pos in positions:
                if pos["contracts"] and pos["contracts"] > 0:
                    side = pos["side"]
                    entry_price = pos["entryPrice"]
                    contracts = pos["contracts"]

                    self.position = Position(
                        direction="long" if side == "long" else "short",
                        entry_price=entry_price,
                        entry_time=datetime.now(),
                        size_usd=contracts * entry_price,
                        size_btc=contracts
                    )
                    print(f"Found existing {side} position: {contracts} BTC @ ${entry_price:,.2f}")
        except Exception as e:
            print(f"Error checking positions: {e}")

    async def get_funding_rate(self) -> float:
        """Get current funding rate"""
        try:
            funding = await self.exchange.fetch_funding_rate(config.SYMBOL)
            return funding.get("fundingRate", 0)
        except Exception as e:
            print(f"Error fetching funding rate: {e}")
            return 0

    async def get_price(self) -> float:
        """Get current BTC price"""
        try:
            ticker = await self.exchange.fetch_ticker(config.SYMBOL)
            return ticker.get("last", 0)
        except Exception as e:
            print(f"Error fetching price: {e}")
            return 0

    async def open_position(self, direction: str, price: float):
        """Open a new position"""
        size_btc = self.position_size / price

        print(f"\nOpening {direction.upper()} position:")
        print(f"  Size: ${self.position_size} ({size_btc:.6f} BTC)")
        print(f"  Price: ${price:,.2f}")

        try:
            side = "buy" if direction == "long" else "sell"
            order = await self.exchange.create_market_order(
                config.SYMBOL,
                side,
                size_btc
            )

            self.position = Position(
                direction=direction,
                entry_price=price,
                entry_time=datetime.now(),
                size_usd=self.position_size,
                size_btc=size_btc
            )

            print(f"  Order filled: {order.get('id')}")
            return True

        except Exception as e:
            print(f"  Error opening position: {e}")
            return False

    async def close_position(self, price: float, reason: str):
        """Close current position"""
        if not self.position:
            return

        print(f"\nClosing {self.position.direction.upper()} position ({reason}):")
        print(f"  Entry: ${self.position.entry_price:,.2f}")
        print(f"  Exit: ${price:,.2f}")

        # Calculate PnL
        if self.position.direction == "long":
            price_pnl = self.position.size_usd * (price - self.position.entry_price) / self.position.entry_price
        else:
            price_pnl = self.position.size_usd * (self.position.entry_price - price) / self.position.entry_price

        total_pnl = price_pnl + self.position.funding_collected

        print(f"  Funding collected: ${self.position.funding_collected:+.2f}")
        print(f"  Price P&L: ${price_pnl:+.2f}")
        print(f"  Total P&L: ${total_pnl:+.2f}")

        try:
            side = "sell" if self.position.direction == "long" else "buy"
            order = await self.exchange.create_market_order(
                config.SYMBOL,
                side,
                self.position.size_btc
            )

            print(f"  Order filled: {order.get('id')}")
            self.position = None
            return True

        except Exception as e:
            print(f"  Error closing position: {e}")
            return False

    async def run_iteration(self):
        """Run one iteration of the strategy"""
        funding_rate = await self.get_funding_rate()
        current_price = await self.get_price()

        if current_price == 0:
            return

        now = datetime.now()
        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}]")
        print(f"  BTC Price: ${current_price:,.2f}")
        print(f"  Funding Rate: {funding_rate*100:.4f}%")

        # If we have a position
        if self.position:
            print(f"  Position: {self.position.direction.upper()} @ ${self.position.entry_price:,.2f}")

            # Check stop loss
            if self.position.direction == "long":
                stop_price = self.position.entry_price * (1 - self.stop_loss_pct)
                if current_price <= stop_price:
                    await self.close_position(current_price, "stop_loss")
                    return
            else:  # short
                stop_price = self.position.entry_price * (1 + self.stop_loss_pct)
                if current_price >= stop_price:
                    await self.close_position(current_price, "stop_loss")
                    return

            # Collect funding (simplified - actual collection happens automatically)
            if self.position.direction == "short" and funding_rate > 0:
                payment = self.position.size_usd * funding_rate
                self.position.funding_collected += payment
                print(f"  Funding payment: ${payment:+.4f}")
            elif self.position.direction == "long" and funding_rate < 0:
                payment = self.position.size_usd * abs(funding_rate)
                self.position.funding_collected += payment
                print(f"  Funding payment: ${payment:+.4f}")

            # Check if funding normalized
            if self.exit_low <= funding_rate <= self.exit_high:
                await self.close_position(current_price, "funding_normalized")
                return

            print(f"  Total funding collected: ${self.position.funding_collected:+.2f}")

        # Check for new entry
        else:
            if funding_rate >= self.short_threshold:
                print(f"  SIGNAL: GO SHORT (funding {funding_rate*100:.4f}% >= {self.short_threshold*100:.3f}%)")
                await self.open_position("short", current_price)

            elif funding_rate <= self.long_threshold:
                print(f"  SIGNAL: GO LONG (funding {funding_rate*100:.4f}% <= {self.long_threshold*100:.3f}%)")
                await self.open_position("long", current_price)

            else:
                print(f"  No signal (funding in normal range)")

    async def run(self):
        """Main run loop"""
        print("\n" + "=" * 60)
        print("CARRY BOT - Funding Rate Collection")
        print("=" * 60)
        print(f"Short threshold: >= {self.short_threshold*100:.3f}%")
        print(f"Long threshold: <= {self.long_threshold*100:.3f}%")
        print(f"Exit range: {self.exit_low*100:.3f}% to {self.exit_high*100:.3f}%")
        print(f"Position size: ${self.position_size}")
        print(f"Stop loss: {self.stop_loss_pct*100}%")
        print(f"Poll interval: {config.POLL_INTERVAL_SECONDS}s")
        print("=" * 60)

        await self.initialize()

        self.running = True

        try:
            while self.running:
                await self.run_iteration()
                await asyncio.sleep(config.POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            await self.close()


async def main():
    """Main entry point"""
    bot = CarryBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
