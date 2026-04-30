"""
Dashboard - Terminal-based live dashboard showing spreads, PnL, and opportunities
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

from monitor import PriceData
from arbitrage import ArbitrageOpportunity, ArbitrageCalculator
from logger import DatabaseLogger
from risk import RiskManager
import config


class TerminalDashboard:
    """Terminal-based dashboard for the arbitrage bot"""

    def __init__(self, logger: DatabaseLogger, risk_manager: RiskManager):
        self.logger = logger
        self.risk_manager = risk_manager
        self.calculator = ArbitrageCalculator()
        self.last_prices: Dict[str, PriceData] = {}
        self.last_opportunities: List[ArbitrageOpportunity] = []
        self.start_time = datetime.now()

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def update_prices(self, prices: Dict[str, PriceData]):
        """Update current prices"""
        self.last_prices = prices

    def update_opportunities(self, opportunities: List[ArbitrageOpportunity]):
        """Update current opportunities"""
        self.last_opportunities = opportunities

    def render(self):
        """Render the dashboard"""
        self.clear_screen()

        # Header
        print("=" * 80)
        print("   CROSS-EXCHANGE ARBITRAGE BOT - LIVE DASHBOARD")
        mode = "OBSERVATION" if not config.LIVE_TRADING else "LIVE TRADING"
        print(f"   Mode: {mode} | Trade Size: ${config.TRADE_SIZE_USD} | Min Spread: {config.MIN_SPREAD_TO_TRADE*100:.2f}%")
        print("=" * 80)

        # Current time and uptime
        now = datetime.now()
        uptime = now - self.start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n  Time: {now.strftime('%Y-%m-%d %H:%M:%S')} | Uptime: {hours}h {minutes}m {seconds}s")

        # Exchange Prices
        print("\n" + "-" * 80)
        print("  EXCHANGE PRICES (BTC/USDT)")
        print("-" * 80)

        if self.last_prices:
            print(f"  {'Exchange':<12} {'Bid':>14} {'Ask':>14} {'Spread':>10} {'Latency':>10}")
            print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*10} {'-'*10}")

            for exchange in config.EXCHANGES:
                if exchange in self.last_prices:
                    price = self.last_prices[exchange]
                    spread = ((price.ask - price.bid) / price.mid) * 100
                    print(f"  {exchange:<12} ${price.bid:>12,.2f} ${price.ask:>12,.2f} "
                          f"{spread:>9.4f}% {price.latency_ms:>8.0f}ms")
        else:
            print("  Waiting for price data...")

        # Current Spreads
        print("\n" + "-" * 80)
        print("  CROSS-EXCHANGE SPREADS")
        print("-" * 80)

        if len(self.last_prices) >= 2:
            opportunities = self.calculator.find_opportunities(self.last_prices)

            if opportunities:
                print(f"  {'Pair':<20} {'Gross':>10} {'Fees':>10} {'Net':>10} {'Status':>12}")
                print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

                for opp in opportunities[:6]:  # Show top 6
                    status = "TRADEABLE" if opp.opportunity_type.value == "tradeable" else "LOG ONLY"
                    color_start = "\033[92m" if status == "TRADEABLE" else "\033[93m"
                    color_end = "\033[0m"

                    print(f"  {opp.pair_name:<20} "
                          f"{opp.gross_spread_pct*100:>9.4f}% "
                          f"{opp.total_fee_pct*100:>9.4f}% "
                          f"{opp.net_spread_pct*100:>9.4f}% "
                          f"{color_start}{status:>12}{color_end}")
            else:
                print("  No profitable spreads detected")
        else:
            print("  Waiting for prices from multiple exchanges...")

        # Today's Statistics
        print("\n" + "-" * 80)
        print("  TODAY'S STATISTICS")
        print("-" * 80)

        stats = self.logger.get_today_stats()
        opp_per_hour = self.logger.get_opportunity_count_last_hour()

        print(f"  Opportunities Seen:      {stats['opportunities_seen']:>10}")
        print(f"  Tradeable Opportunities: {stats['tradeable_opportunities']:>10}")
        print(f"  Opportunities/Hour:      {opp_per_hour:>10}")
        print(f"  Avg Net Spread:          {stats['avg_spread_pct']:>9.4f}%")
        print(f"  Trades Executed:         {stats['trades_executed']:>10}")
        print(f"  Today's PnL:             ${stats['total_pnl']:>9.2f}")
        print(f"  Win Rate:                {stats['win_rate']:>9.1f}%")

        # Risk Status
        risk_summary = self.risk_manager.get_risk_summary()
        print(f"\n  Daily Loss Budget:       ${risk_summary['remaining_loss_budget']:>9.2f} remaining")

        if risk_summary['trading_halted']:
            print(f"  \033[91mTRADING HALTED: {risk_summary['halt_reason']}\033[0m")

        # Recent Trades
        print("\n" + "-" * 80)
        print("  LAST 5 TRADES")
        print("-" * 80)

        recent_trades = self.logger.get_recent_trades(5)
        if recent_trades:
            print(f"  {'Time':<20} {'Pair':<18} {'Spread':>10} {'PnL':>10} {'Status':>12}")
            print(f"  {'-'*20} {'-'*18} {'-'*10} {'-'*10} {'-'*12}")

            for trade in recent_trades:
                timestamp = trade['timestamp'][:19] if trade['timestamp'] else "N/A"
                pnl_color = "\033[92m" if trade['pnl'] >= 0 else "\033[91m"
                print(f"  {timestamp:<20} {trade['pair']:<18} "
                      f"{trade['spread_pct']:>9.4f}% "
                      f"{pnl_color}${trade['pnl']:>9.2f}\033[0m "
                      f"{trade['status']:>12}")
        else:
            print("  No trades yet")

        # Estimated Daily Profit (observation mode)
        if not config.LIVE_TRADING and stats['tradeable_opportunities'] > 0:
            print("\n" + "-" * 80)
            print("  ESTIMATED DAILY PROFIT (Observation Mode)")
            print("-" * 80)

            avg_profit_per_trade = config.TRADE_SIZE_USD * (stats['avg_spread_pct'] / 100)
            est_daily_profit = stats['tradeable_opportunities'] * avg_profit_per_trade

            print(f"  Avg Profit per Trade:    ${avg_profit_per_trade:>9.4f}")
            print(f"  Estimated Daily Profit:  ${est_daily_profit:>9.2f}")
            print(f"  (Based on {stats['tradeable_opportunities']} tradeable opportunities)")

        print("\n" + "=" * 80)
        print("  Press Ctrl+C to stop")
        print("=" * 80)

    def render_compact(self):
        """Render a compact single-line status"""
        now = datetime.now().strftime('%H:%M:%S')
        stats = self.logger.get_today_stats()

        # Get best spread
        best_spread = 0
        if len(self.last_prices) >= 2:
            opportunities = self.calculator.find_opportunities(self.last_prices)
            if opportunities:
                best_spread = opportunities[0].net_spread_pct * 100

        mode = "OBS" if not config.LIVE_TRADING else "LIVE"

        print(f"\r[{now}] [{mode}] "
              f"BTC: ${self.last_prices.get('binance', PriceData('', '', 0, 0, 0, datetime.now(), 0)).mid:,.0f} | "
              f"Best Spread: {best_spread:.4f}% | "
              f"Opps: {stats['opportunities_seen']} | "
              f"Trades: {stats['trades_executed']} | "
              f"PnL: ${stats['total_pnl']:.2f}",
              end="", flush=True)


def test_dashboard():
    """Test the dashboard with mock data"""
    from datetime import datetime

    logger = DatabaseLogger()
    risk_manager = RiskManager()
    dashboard = TerminalDashboard(logger, risk_manager)

    # Mock prices
    mock_prices = {
        "binance": PriceData("binance", "BTC/USDT", 95000, 95010, 95005, datetime.now(), 45),
        "bybit": PriceData("bybit", "BTC/USDT", 95050, 95060, 95055, datetime.now(), 62),
        "okx": PriceData("okx", "BTC/USDT", 95020, 95030, 95025, datetime.now(), 55),
    }

    dashboard.update_prices(mock_prices)
    dashboard.render()


if __name__ == "__main__":
    test_dashboard()
