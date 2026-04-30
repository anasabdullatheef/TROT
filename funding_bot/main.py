#!/usr/bin/env python3
"""
Funding Rate Arbitrage Bot - Main Entry Point
Scheduler that runs the strategy every hour
"""

import time
import schedule
from datetime import datetime
from typing import Optional

import config
from funding import get_current_funding_rate, get_next_funding_time
from strategy import check_entry_conditions, check_exit_conditions
from regime import calculate_adx
from risk import RiskManager
from exchange import ExchangeClient, open_long_position, open_short_position, close_position
from logger import TradeLogger


class FundingBot:
    """
    Main bot class that orchestrates the funding rate arbitrage strategy
    """

    def __init__(self, live_mode: bool = False, testnet: bool = True):
        """
        Initialize the bot

        Args:
            live_mode: If False, only logs signals without executing trades
            testnet: If True and live_mode is True, use Binance testnet
        """
        self.live_mode = live_mode
        self.exchange = ExchangeClient(testnet=testnet) if live_mode else None
        self.risk_manager = RiskManager()
        self.logger = TradeLogger()
        self.current_position: Optional[dict] = None

        print("=" * 60)
        print("🤖 FUNDING RATE ARBITRAGE BOT")
        print("=" * 60)
        print(f"Mode: {'LIVE' if live_mode else 'PAPER'} {'(Testnet)' if testnet and live_mode else ''}")
        print(f"Symbol: {config.SYMBOL}")
        print(f"Check interval: {config.CHECK_INTERVAL_MINUTES} minutes")
        print(f"Funding thresholds: >{config.FUNDING_LONG_THRESHOLD*100:.2f}% (short), "
              f"<{config.FUNDING_SHORT_THRESHOLD*100:.2f}% (long)")
        print(f"Risk per trade: {config.RISK_PER_TRADE*100}%")
        print("=" * 60)

    def get_market_data(self) -> dict:
        """
        Fetch current market data including price, funding rate, and indicators
        """
        # Get current funding rate
        funding_rate = get_current_funding_rate()
        if funding_rate is None:
            return None

        # Get current price and calculate indicators
        if self.exchange:
            ticker = self.exchange.get_ticker()
            current_price = ticker.get("last", 0)

            # Fetch recent candles for indicators
            candles = self.exchange.fetch_ohlcv(limit=50)
            if not candles:
                return None

            import pandas as pd
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            # Paper mode - fetch from public API
            import requests
            url = f"{config.BINANCE_BASE_URL}/fapi/v1/klines"
            params = {
                "symbol": config.BINANCE_SYMBOL,
                "interval": "1h",
                "limit": 50
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            import pandas as pd
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            current_price = float(df.iloc[-1]["close"])

        # Calculate indicators
        from ta.momentum import RSIIndicator
        from ta.trend import ADXIndicator

        rsi_indicator = RSIIndicator(close=df["close"], window=config.RSI_PERIOD)
        df["rsi"] = rsi_indicator.rsi()

        adx_indicator = ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"],
            window=config.ADX_PERIOD
        )
        df["adx"] = adx_indicator.adx()

        latest = df.iloc[-1]

        return {
            "price": current_price,
            "funding_rate": funding_rate,
            "rsi": latest["rsi"],
            "adx": latest["adx"],
            "timestamp": datetime.now()
        }

    def check_and_trade(self):
        """
        Main trading logic - called on schedule
        """
        print(f"\n[{datetime.now()}] Checking market conditions...")

        # Get market data
        data = self.get_market_data()
        if data is None:
            self.logger.log_error("Failed to fetch market data")
            return

        price = data["price"]
        funding_rate = data["funding_rate"]
        rsi = data["rsi"]
        adx = data["adx"]

        print(f"  Price: ${price:,.2f}")
        print(f"  Funding: {funding_rate*100:.4f}%")
        print(f"  RSI: {rsi:.1f}")
        print(f"  ADX: {adx:.1f}")

        # Check if we have an open position
        if self.current_position is not None:
            # Check exit conditions
            exit_check = check_exit_conditions(
                entry_price=self.current_position["entry_price"],
                current_price=price,
                position_type=self.current_position["type"],
                funding_rate=funding_rate
            )

            if exit_check["exit"]:
                self._close_position(price, exit_check["reason"])
                return

            print(f"  Position: {self.current_position['type'].upper()} @ ${self.current_position['entry_price']:,.2f}")
            print(f"  Unrealized PnL: {exit_check['pnl_pct']*100:+.2f}%")

        else:
            # Check if we can open a new position
            can_open = self.risk_manager.can_open_position()

            if not can_open["allowed"]:
                self.logger.log_skip(datetime.now(), "; ".join(can_open["reasons"]))
                return

            # Check entry conditions
            entry_check = check_entry_conditions(
                funding_rate=funding_rate,
                rsi=rsi,
                adx=adx
            )

            if entry_check["signal"]:
                self._open_position(
                    signal=entry_check["signal"],
                    price=price,
                    funding_rate=funding_rate,
                    rsi=rsi,
                    adx=adx
                )
            else:
                print("  No signal - conditions not met")

    def _open_position(self, signal: str, price: float, funding_rate: float, rsi: float, adx: float):
        """Open a new position"""
        # Calculate position size
        sizing = self.risk_manager.calculate_position_size(price)

        self.logger.log_signal(
            timestamp=datetime.now(),
            direction=signal,
            price=price,
            funding_rate=funding_rate,
            rsi=rsi,
            adx=adx,
            reason=f"Funding extreme, RSI confirms, ADX low"
        )

        if self.live_mode and self.exchange:
            # Execute real trade
            if signal == "long":
                result = open_long_position(self.exchange, sizing["position_size"], price)
            else:
                result = open_short_position(self.exchange, sizing["position_size"], price)

            if result is None:
                self.logger.log_error(f"Failed to open {signal} position")
                return

            actual_price = result["entry"]["price"] or price
        else:
            actual_price = price

        # Record position
        self.current_position = {
            "type": signal,
            "entry_price": actual_price,
            "size": sizing["position_size"],
            "entry_time": datetime.now()
        }

        self.risk_manager.open_position(
            position_type=signal,
            entry_price=actual_price,
            size=sizing["position_size"],
            timestamp=datetime.now()
        )

        self.logger.log_entry(
            timestamp=datetime.now(),
            direction=signal,
            price=actual_price,
            size=sizing["position_size"],
            funding_rate=funding_rate,
            capital=self.risk_manager.current_capital,
            reason=f"Signal confirmed"
        )

    def _close_position(self, price: float, reason: str):
        """Close the current position"""
        if self.current_position is None:
            return

        if self.live_mode and self.exchange:
            # Execute real close
            result = close_position(self.exchange)
            if result:
                price = result.get("price", price)

        # Record the trade
        trade = self.risk_manager.close_position(
            exit_price=price,
            exit_reason=reason,
            timestamp=datetime.now()
        )

        if trade:
            self.logger.log_exit(
                timestamp=datetime.now(),
                direction=self.current_position["type"],
                price=price,
                size=self.current_position["size"],
                pnl=trade["pnl"],
                pnl_pct=trade["pnl_pct"],
                capital=self.risk_manager.current_capital,
                reason=reason
            )

        self.current_position = None

    def run(self):
        """Start the bot scheduler"""
        print(f"\n🚀 Starting bot... checking every {config.CHECK_INTERVAL_MINUTES} minutes")

        # Schedule the check
        schedule.every(config.CHECK_INTERVAL_MINUTES).minutes.do(self.check_and_trade)

        # Also run immediately
        self.check_and_trade()

        # Keep running
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⛔ Bot stopped by user")
            self._shutdown()

    def _shutdown(self):
        """Clean shutdown"""
        # Close any open position
        if self.current_position:
            data = self.get_market_data()
            if data:
                self._close_position(data["price"], "Bot shutdown")

        # Print final stats
        stats = self.risk_manager.get_statistics()
        self.logger.log_stats(stats)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Funding Rate Arbitrage Bot")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode (requires API keys)"
    )
    parser.add_argument(
        "--mainnet",
        action="store_true",
        help="Use mainnet instead of testnet (only with --live)"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest instead of live/paper trading"
    )

    args = parser.parse_args()

    if args.backtest:
        # Run backtest
        from backtest import main as run_backtest
        run_backtest()
    else:
        # Run live/paper bot
        bot = FundingBot(
            live_mode=args.live,
            testnet=not args.mainnet
        )
        bot.run()


if __name__ == "__main__":
    main()
