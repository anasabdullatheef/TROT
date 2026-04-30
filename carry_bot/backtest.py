#!/usr/bin/env python3
"""
Carry Bot Backtest
Simulates funding rate collection strategy on historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from funding import fetch_historical_funding_sync, fetch_historical_prices_sync
import config


@dataclass
class Position:
    """Represents an open position"""
    entry_time: datetime
    entry_price: float
    direction: str  # "long" or "short"
    size_usd: float
    size_btc: float
    entry_funding_rate: float
    funding_collected: float = 0.0
    funding_payments: int = 0


@dataclass
class Trade:
    """Completed trade record"""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    size_usd: float
    funding_collected: float
    funding_payments: int
    price_pnl: float
    total_pnl: float
    exit_reason: str  # "funding_normalized", "stop_loss"
    duration_hours: float


class CarryBacktest:
    """Backtests the funding rate carry strategy"""

    def __init__(
        self,
        funding_short_threshold: float = config.FUNDING_SHORT_THRESHOLD,
        funding_long_threshold: float = config.FUNDING_LONG_THRESHOLD,
        funding_exit_high: float = config.FUNDING_EXIT_HIGH,
        funding_exit_low: float = config.FUNDING_EXIT_LOW,
        position_size_usd: float = config.POSITION_SIZE_USD,
        stop_loss_pct: float = config.STOP_LOSS_PCT
    ):
        self.funding_short_threshold = funding_short_threshold
        self.funding_long_threshold = funding_long_threshold
        self.funding_exit_high = funding_exit_high
        self.funding_exit_low = funding_exit_low
        self.position_size_usd = position_size_usd
        self.stop_loss_pct = stop_loss_pct

        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.funding_df: Optional[pd.DataFrame] = None
        self.price_df: Optional[pd.DataFrame] = None

    def load_data(self, days: int = 365):
        """Load historical data"""
        self.funding_df = fetch_historical_funding_sync(days)
        self.price_df = fetch_historical_prices_sync(days)

        if self.funding_df.empty or self.price_df.empty:
            raise ValueError("Failed to load historical data")

        # Merge price data to funding times
        self.funding_df = self.funding_df.sort_values("timestamp")
        self.price_df = self.price_df.sort_values("timestamp")

        # For each funding time, get the price at that time
        self.funding_df = pd.merge_asof(
            self.funding_df,
            self.price_df[["timestamp", "close", "high", "low"]],
            on="timestamp",
            direction="backward"
        )
        self.funding_df = self.funding_df.rename(columns={"close": "price"})

    def get_price_extremes_between(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict:
        """Get high and low prices between two times (for stop loss check)"""
        mask = (self.price_df["timestamp"] >= start_time) & (self.price_df["timestamp"] <= end_time)
        subset = self.price_df[mask]

        if subset.empty:
            return {"high": 0, "low": float("inf")}

        return {
            "high": subset["high"].max(),
            "low": subset["low"].min()
        }

    def check_stop_loss(
        self,
        position: Position,
        current_time: datetime,
        last_check_time: datetime
    ) -> bool:
        """Check if stop loss was hit between funding payments"""
        extremes = self.get_price_extremes_between(last_check_time, current_time)

        if position.direction == "long":
            # Stop loss triggers if price drops 2% below entry
            stop_price = position.entry_price * (1 - self.stop_loss_pct)
            return extremes["low"] <= stop_price
        else:  # short
            # Stop loss triggers if price rises 2% above entry
            stop_price = position.entry_price * (1 + self.stop_loss_pct)
            return extremes["high"] >= stop_price

    def run(self) -> Dict:
        """Run the backtest"""
        print("\n" + "=" * 70)
        print("CARRY BOT BACKTEST")
        print("=" * 70)
        print(f"Position Size: ${self.position_size_usd}")
        print(f"Short Threshold: {self.funding_short_threshold*100:.2f}%")
        print(f"Long Threshold: {self.funding_long_threshold*100:.2f}%")
        print(f"Exit Range: {self.funding_exit_low*100:.2f}% to {self.funding_exit_high*100:.2f}%")
        print(f"Stop Loss: {self.stop_loss_pct*100:.1f}%")
        print("=" * 70)

        if self.funding_df is None:
            self.load_data()

        print(f"\nData range: {self.funding_df['timestamp'].min()} to {self.funding_df['timestamp'].max()}")
        print(f"Total funding periods: {len(self.funding_df)}")

        # Track statistics
        entry_signals = {"long": 0, "short": 0}
        last_funding_time = None

        for idx, row in self.funding_df.iterrows():
            current_time = row["timestamp"]
            funding_rate = row["rate"]
            current_price = row["price"]

            if pd.isna(current_price) or current_price == 0:
                continue

            # Check stop loss if we have a position
            if self.position and last_funding_time:
                if self.check_stop_loss(self.position, current_time, last_funding_time):
                    # Stop loss hit - close position
                    if self.position.direction == "long":
                        exit_price = self.position.entry_price * (1 - self.stop_loss_pct)
                    else:
                        exit_price = self.position.entry_price * (1 + self.stop_loss_pct)

                    price_pnl = self._calculate_price_pnl(self.position, exit_price)
                    total_pnl = price_pnl + self.position.funding_collected

                    trade = Trade(
                        entry_time=self.position.entry_time,
                        exit_time=current_time,
                        direction=self.position.direction,
                        entry_price=self.position.entry_price,
                        exit_price=exit_price,
                        size_usd=self.position.size_usd,
                        funding_collected=self.position.funding_collected,
                        funding_payments=self.position.funding_payments,
                        price_pnl=price_pnl,
                        total_pnl=total_pnl,
                        exit_reason="stop_loss",
                        duration_hours=(current_time - self.position.entry_time).total_seconds() / 3600
                    )
                    self.trades.append(trade)
                    self.position = None

            # If we have a position, collect funding and check exit
            if self.position:
                # Collect funding
                # Funding payment: if we're short and funding is positive, we collect
                # If we're long and funding is negative, we collect
                funding_payment = 0

                if self.position.direction == "short" and funding_rate > 0:
                    # Shorts receive funding when rate is positive
                    funding_payment = self.position.size_usd * funding_rate
                elif self.position.direction == "long" and funding_rate < 0:
                    # Longs receive funding when rate is negative
                    funding_payment = self.position.size_usd * abs(funding_rate)
                elif self.position.direction == "short" and funding_rate < 0:
                    # Shorts pay funding when rate is negative
                    funding_payment = -self.position.size_usd * abs(funding_rate)
                elif self.position.direction == "long" and funding_rate > 0:
                    # Longs pay funding when rate is positive
                    funding_payment = -self.position.size_usd * funding_rate

                self.position.funding_collected += funding_payment
                self.position.funding_payments += 1

                # Check if funding normalized - time to exit
                if self.funding_exit_low <= funding_rate <= self.funding_exit_high:
                    # Close position
                    price_pnl = self._calculate_price_pnl(self.position, current_price)
                    total_pnl = price_pnl + self.position.funding_collected

                    trade = Trade(
                        entry_time=self.position.entry_time,
                        exit_time=current_time,
                        direction=self.position.direction,
                        entry_price=self.position.entry_price,
                        exit_price=current_price,
                        size_usd=self.position.size_usd,
                        funding_collected=self.position.funding_collected,
                        funding_payments=self.position.funding_payments,
                        price_pnl=price_pnl,
                        total_pnl=total_pnl,
                        exit_reason="funding_normalized",
                        duration_hours=(current_time - self.position.entry_time).total_seconds() / 3600
                    )
                    self.trades.append(trade)
                    self.position = None

            # Check for new entry signals (only if no position)
            if not self.position:
                if funding_rate >= self.funding_short_threshold:
                    # Go short
                    entry_signals["short"] += 1
                    self.position = Position(
                        entry_time=current_time,
                        entry_price=current_price,
                        direction="short",
                        size_usd=self.position_size_usd,
                        size_btc=self.position_size_usd / current_price,
                        entry_funding_rate=funding_rate
                    )

                elif funding_rate <= self.funding_long_threshold:
                    # Go long
                    entry_signals["long"] += 1
                    self.position = Position(
                        entry_time=current_time,
                        entry_price=current_price,
                        direction="long",
                        size_usd=self.position_size_usd,
                        size_btc=self.position_size_usd / current_price,
                        entry_funding_rate=funding_rate
                    )

            last_funding_time = current_time

        # Close any remaining position at end
        if self.position:
            last_row = self.funding_df.iloc[-1]
            current_price = last_row["price"]
            price_pnl = self._calculate_price_pnl(self.position, current_price)
            total_pnl = price_pnl + self.position.funding_collected

            trade = Trade(
                entry_time=self.position.entry_time,
                exit_time=last_row["timestamp"],
                direction=self.position.direction,
                entry_price=self.position.entry_price,
                exit_price=current_price,
                size_usd=self.position.size_usd,
                funding_collected=self.position.funding_collected,
                funding_payments=self.position.funding_payments,
                price_pnl=price_pnl,
                total_pnl=total_pnl,
                exit_reason="backtest_end",
                duration_hours=(last_row["timestamp"] - self.position.entry_time).total_seconds() / 3600
            )
            self.trades.append(trade)

        return self._generate_report(entry_signals)

    def _calculate_price_pnl(self, position: Position, exit_price: float) -> float:
        """Calculate PnL from price movement"""
        if position.direction == "long":
            return position.size_usd * (exit_price - position.entry_price) / position.entry_price
        else:  # short
            return position.size_usd * (position.entry_price - exit_price) / position.entry_price

    def _generate_report(self, entry_signals: Dict) -> Dict:
        """Generate backtest report"""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)

        # Entry signals
        total_signals = entry_signals["long"] + entry_signals["short"]
        print(f"\n📊 ENTRY SIGNALS:")
        print(f"   Total triggers: {total_signals}")
        print(f"   Long signals (funding < -0.04%): {entry_signals['long']}")
        print(f"   Short signals (funding > +0.08%): {entry_signals['short']}")

        if not self.trades:
            print("\n❌ No trades executed")
            return {"signals": entry_signals, "trades": 0}

        # Trade statistics
        print(f"\n📊 TRADE STATISTICS:")
        print(f"   Total trades: {len(self.trades)}")

        wins = [t for t in self.trades if t.total_pnl > 0]
        losses = [t for t in self.trades if t.total_pnl <= 0]
        print(f"   Winners: {len(wins)} ({len(wins)/len(self.trades)*100:.1f}%)")
        print(f"   Losers: {len(losses)} ({len(losses)/len(self.trades)*100:.1f}%)")

        # Exit reasons
        funding_exits = [t for t in self.trades if t.exit_reason == "funding_normalized"]
        stop_losses = [t for t in self.trades if t.exit_reason == "stop_loss"]
        print(f"\n   Exit reasons:")
        print(f"   - Funding normalized: {len(funding_exits)}")
        print(f"   - Stop loss: {len(stop_losses)}")

        # Funding collected
        total_funding = sum(t.funding_collected for t in self.trades)
        total_price_pnl = sum(t.price_pnl for t in self.trades)
        total_pnl = sum(t.total_pnl for t in self.trades)

        print(f"\n💰 PROFIT/LOSS:")
        print(f"   Total Funding Collected: ${total_funding:+.2f}")
        print(f"   Total Price P&L: ${total_price_pnl:+.2f}")
        print(f"   Net Total P&L: ${total_pnl:+.2f}")

        # Average per trade
        avg_funding = total_funding / len(self.trades)
        avg_price_pnl = total_price_pnl / len(self.trades)
        avg_total = total_pnl / len(self.trades)

        print(f"\n   Per Trade Average:")
        print(f"   - Funding: ${avg_funding:+.2f}")
        print(f"   - Price P&L: ${avg_price_pnl:+.2f}")
        print(f"   - Total: ${avg_total:+.2f}")

        # Duration
        avg_duration = sum(t.duration_hours for t in self.trades) / len(self.trades)
        avg_funding_payments = sum(t.funding_payments for t in self.trades) / len(self.trades)

        print(f"\n⏱️  TIMING:")
        print(f"   Avg trade duration: {avg_duration:.1f} hours")
        print(f"   Avg funding payments per trade: {avg_funding_payments:.1f}")

        # By direction
        long_trades = [t for t in self.trades if t.direction == "long"]
        short_trades = [t for t in self.trades if t.direction == "short"]

        print(f"\n📈 BY DIRECTION:")
        if long_trades:
            long_pnl = sum(t.total_pnl for t in long_trades)
            long_funding = sum(t.funding_collected for t in long_trades)
            print(f"   LONG ({len(long_trades)} trades):")
            print(f"   - Funding: ${long_funding:+.2f}")
            print(f"   - Total P&L: ${long_pnl:+.2f}")

        if short_trades:
            short_pnl = sum(t.total_pnl for t in short_trades)
            short_funding = sum(t.funding_collected for t in short_trades)
            print(f"   SHORT ({len(short_trades)} trades):")
            print(f"   - Funding: ${short_funding:+.2f}")
            print(f"   - Total P&L: ${short_pnl:+.2f}")

        # Show recent trades
        print(f"\n📋 RECENT TRADES (last 10):")
        print(f"   {'Entry':<12} {'Exit':<12} {'Dir':<6} {'Funding':>10} {'Price PnL':>10} {'Total':>10} {'Exit Reason':<18}")
        print(f"   {'-'*12} {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*18}")

        for t in self.trades[-10:]:
            entry_str = t.entry_time.strftime("%Y-%m-%d")
            exit_str = t.exit_time.strftime("%Y-%m-%d")
            print(f"   {entry_str:<12} {exit_str:<12} {t.direction:<6} "
                  f"${t.funding_collected:>9.2f} ${t.price_pnl:>9.2f} ${t.total_pnl:>9.2f} {t.exit_reason:<18}")

        # ROI
        capital = self.position_size_usd
        roi = (total_pnl / capital) * 100

        print(f"\n📊 ANNUAL RETURN:")
        print(f"   Capital: ${capital:.2f}")
        print(f"   Total Return: ${total_pnl:+.2f} ({roi:+.1f}%)")

        print("\n" + "=" * 70)

        return {
            "signals": entry_signals,
            "trades": len(self.trades),
            "winners": len(wins),
            "losers": len(losses),
            "funding_exits": len(funding_exits),
            "stop_losses": len(stop_losses),
            "total_funding": total_funding,
            "total_price_pnl": total_price_pnl,
            "total_pnl": total_pnl,
            "avg_duration_hours": avg_duration,
            "roi_pct": roi
        }


def main():
    """Run the backtest"""
    backtest = CarryBacktest()
    backtest.load_data(days=365)
    results = backtest.run()


if __name__ == "__main__":
    main()
