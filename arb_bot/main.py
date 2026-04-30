#!/usr/bin/env python3
"""
Cross-Exchange Arbitrage Bot - Main Entry Point

Monitors BTC/USDT prices across Binance, Bybit, and OKX
Executes arbitrage trades when spread exceeds threshold
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Optional

from monitor import PriceMonitor
from arbitrage import ArbitrageCalculator, ArbitrageOpportunity, OpportunityType
from executor import TradeExecutor
from risk import RiskManager, RiskAction
from logger import DatabaseLogger
from dashboard import TerminalDashboard
from notifications import TelegramNotifier
import config


class ArbitrageBot:
    """Main arbitrage bot class"""

    def __init__(self):
        self.monitor = PriceMonitor()
        self.calculator = ArbitrageCalculator()
        self.executor: Optional[TradeExecutor] = None
        self.risk_manager = RiskManager()
        self.logger = DatabaseLogger()
        self.notifier = TelegramNotifier()
        self.dashboard = TerminalDashboard(self.logger, self.risk_manager)

        self.running = False
        self.iteration = 0
        self.last_dashboard_update = datetime.now()
        self.dashboard_update_interval = 2  # seconds

    async def initialize(self):
        """Initialize all components"""
        print("Initializing Arbitrage Bot...")

        await self.monitor.initialize()
        self.executor = TradeExecutor(self.monitor.exchanges)

        # Send startup notification
        await self.notifier.notify_startup()

        print("Initialization complete")

    async def shutdown(self, reason: str = "Manual"):
        """Shutdown the bot"""
        print(f"\nShutting down: {reason}")
        self.running = False

        # Close all open positions
        if self.executor:
            for position_id in list(self.executor.open_positions.keys()):
                await self.executor.close_position(position_id, "shutdown")

        # Send shutdown notification
        await self.notifier.notify_shutdown(reason)

        # Close exchange connections
        await self.monitor.close()

        print("Shutdown complete")

    async def run_iteration(self):
        """Run a single iteration of the main loop"""
        self.iteration += 1

        # 1. Fetch prices from all exchanges
        prices = await self.monitor.fetch_all_prices()

        if len(prices) < 2:
            return  # Need at least 2 exchanges

        # Update dashboard prices
        self.dashboard.update_prices(prices)

        # 2. Find arbitrage opportunities
        opportunities = self.calculator.find_opportunities(prices)

        # 3. Log all opportunities
        for opp in opportunities:
            self.logger.log_opportunity(opp)

        # 4. Check risk limits
        can_trade, reason = self.risk_manager.can_trade()

        # 5. Execute tradeable opportunities
        if can_trade and config.LIVE_TRADING:
            best_opp = self.calculator.get_best_opportunity(prices)

            if best_opp and best_opp.opportunity_type == OpportunityType.TRADEABLE:
                position = await self.executor.execute_arbitrage(best_opp)

                if position:
                    self.logger.log_trade(position)
                    self.logger.log_opportunity(best_opp, was_traded=True)
                    await self.notifier.notify_trade_executed(position)
                    self.risk_manager.register_position(
                        position.position_id,
                        best_opp.net_spread_pct
                    )

        # 6. Monitor and close open positions
        if self.executor:
            await self.monitor_positions(prices)

        # 7. Update dashboard periodically
        now = datetime.now()
        if (now - self.last_dashboard_update).total_seconds() >= self.dashboard_update_interval:
            self.dashboard.render()
            self.last_dashboard_update = now

    async def monitor_positions(self, prices):
        """Monitor open positions for exit conditions"""
        if not self.executor:
            return

        for position_id, position in list(self.executor.open_positions.items()):
            # Check circuit breaker
            action = self.risk_manager.check_circuit_breaker(position, prices)
            if action == RiskAction.CLOSE_POSITION:
                await self.executor.close_position(position_id, "circuit_breaker")
                await self.notifier.notify_risk_alert("Circuit Breaker", f"Position {position_id} closed")
                continue

            # Check unhedged timeout
            action = self.risk_manager.check_unhedged_timeout(position)
            if action == RiskAction.CLOSE_POSITION:
                await self.executor.close_position(position_id, "timeout")
                await self.notifier.notify_risk_alert("Timeout", f"Position {position_id} closed")
                continue

    async def run(self):
        """Main run loop"""
        print("\n" + "=" * 60)
        print("CROSS-EXCHANGE ARBITRAGE BOT")
        print("=" * 60)
        print(f"Mode: {'OBSERVATION' if not config.LIVE_TRADING else 'LIVE TRADING'}")
        print(f"Trade Size: ${config.TRADE_SIZE_USD}")
        print(f"Min Spread: {config.MIN_SPREAD_TO_TRADE*100:.2f}%")
        print(f"Poll Interval: {config.POLL_INTERVAL_MS}ms")
        print("=" * 60 + "\n")

        await self.initialize()

        self.running = True
        poll_interval = config.POLL_INTERVAL_MS / 1000  # Convert to seconds

        try:
            while self.running:
                try:
                    await self.run_iteration()
                except Exception as e:
                    print(f"Error in iteration: {e}")

                await asyncio.sleep(poll_interval)

        except KeyboardInterrupt:
            pass
        finally:
            await self.shutdown()


async def main():
    """Main entry point"""
    bot = ArbitrageBot()

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        bot.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
