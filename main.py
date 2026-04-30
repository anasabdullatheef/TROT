#!/usr/bin/env python3
"""
TROT - Trading Robot for Optimal Trading
Main entry point and scheduler for the autonomous trading bot.

Usage:
    python main.py              # Run live trading bot
    python main.py --backtest   # Run backtest only
    python main.py --once       # Run one check cycle and exit
"""
import sys
import time
import argparse
import schedule
from datetime import datetime

import config
from strategy import strategy
from risk import risk_manager
from exchange import exchange
from logger import logger
from backtest import Backtester


class TradingBot:
    """Main trading bot controller."""

    def __init__(self):
        self.is_running = False
        self.check_count = 0

    def check_and_trade(self):
        """Main trading loop - check for signals and execute trades."""
        self.check_count += 1
        logger.log_info(f"=== Check #{self.check_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

        try:
            # Reset daily stats if needed
            risk_manager.reset_daily_stats()

            # Check if trading is killed
            if risk_manager.is_killed:
                logger.log_risk_event("TRADING_PAUSED", risk_manager.kill_reason)
                return

            # Check existing positions for SL/TP
            self._check_exits()

            # Get market data
            df = exchange.fetch_ohlcv(
                symbol=config.SYMBOL,
                timeframe=config.TIMEFRAME,
                limit=100
            )

            current_price = df['close'].iloc[-1]

            # Get current signal
            signal, indicators = strategy.get_current_signal(df)

            if signal != 0:
                logger.log_signal(config.SYMBOL, "BUY" if signal == 1 else "SELL", current_price, indicators)

            # Process buy signal
            if signal == 1:
                self._execute_buy(current_price)

            # Process sell signal (for open positions)
            elif signal == -1:
                self._execute_signal_sell(current_price)

            # Log current status
            self._log_status()

        except Exception as e:
            logger.log_error("Error in check_and_trade", str(e))

    def _check_exits(self):
        """Check all positions for stop-loss or take-profit."""
        to_close = exchange.check_stop_loss_take_profit(config.SYMBOL)

        for symbol, reason, price in to_close:
            position = exchange.positions.get(symbol)
            if position:
                entry_price = position['entry_price']
                quantity = position['quantity']

                # Calculate PnL
                pnl = (price - entry_price) * quantity
                pnl_pct = ((price - entry_price) / entry_price) * 100

                # Close position
                order = exchange.close_position(symbol, reason)

                if order:
                    # Update risk manager
                    risk_manager.update_capital(pnl, pnl > 0)

                    logger.log_trade(
                        action=f"CLOSE_{reason.upper()}",
                        symbol=symbol,
                        side="sell",
                        price=price,
                        quantity=quantity,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        balance=risk_manager.current_capital,
                        reason=reason
                    )

    def _execute_buy(self, current_price: float):
        """Execute a buy order."""
        # Check if we can open a new position
        can_open, reason = risk_manager.can_open_position(exchange.get_position_count())

        if not can_open:
            logger.log_risk_event("BUY_BLOCKED", reason)
            return

        # Check if we already have a position in this symbol
        if config.SYMBOL in exchange.positions:
            logger.log_info(f"Already have position in {config.SYMBOL}")
            return

        # Calculate stop-loss and take-profit
        stop_loss = strategy.get_stop_loss_price(current_price)
        take_profit = strategy.get_take_profit_price(current_price)

        # Calculate position size
        quantity, position_value = risk_manager.calculate_position_size(current_price, stop_loss)

        if quantity <= 0:
            logger.log_risk_event("INVALID_POSITION_SIZE", f"qty={quantity}")
            return

        # Validate risk parameters
        if not risk_manager.validate_stop_loss(current_price, stop_loss):
            return

        # Execute buy order
        order = exchange.place_market_order(
            symbol=config.SYMBOL,
            side="buy",
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        if order:
            logger.log_trade(
                action="BUY",
                symbol=config.SYMBOL,
                side="buy",
                price=current_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                balance=risk_manager.current_capital
            )

    def _execute_signal_sell(self, current_price: float):
        """Execute sell based on strategy signal."""
        if config.SYMBOL not in exchange.positions:
            return

        position = exchange.positions[config.SYMBOL]
        entry_price = position['entry_price']
        quantity = position['quantity']

        # Calculate PnL
        pnl = (current_price - entry_price) * quantity
        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        # Close position
        order = exchange.close_position(config.SYMBOL, "signal_exit")

        if order:
            risk_manager.update_capital(pnl, pnl > 0)

            logger.log_trade(
                action="SELL",
                symbol=config.SYMBOL,
                side="sell",
                price=current_price,
                quantity=quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                balance=risk_manager.current_capital,
                reason="EMA crossdown"
            )

    def _log_status(self):
        """Log current bot status."""
        metrics = risk_manager.get_risk_metrics()
        positions = exchange.get_position_count()

        logger.log_info(
            f"Status: Capital=${metrics['current_capital']:.2f} | "
            f"Positions={positions}/{config.MAX_OPEN_POSITIONS} | "
            f"Daily PnL=${metrics['daily_pnl']:.2f} ({metrics['daily_drawdown_pct']:.1f}% DD)"
        )

    def run(self, once: bool = False):
        """Start the trading bot."""
        self.is_running = True

        logger.log_success("="*50)
        logger.log_success("  TROT - Trading Robot Starting")
        logger.log_success(f"  Mode: {config.TRADING_MODE.upper()}")
        logger.log_success(f"  Symbol: {config.SYMBOL}")
        logger.log_success(f"  Capital: ${config.INITIAL_CAPITAL:,.2f}")
        logger.log_success("="*50)

        # Run once if requested
        if once:
            self.check_and_trade()
            return

        # Schedule regular checks
        schedule.every(config.CHECK_INTERVAL_MINUTES).minutes.do(self.check_and_trade)

        # Run initial check
        self.check_and_trade()

        # Main loop
        logger.log_info(f"Scheduler running (checking every {config.CHECK_INTERVAL_MINUTES} minutes)")

        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.log_info("Shutting down...")
            self.stop()

    def stop(self):
        """Stop the trading bot."""
        self.is_running = False
        logger.log_info("Bot stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='TROT - Trading Robot for Optimal Trading')
    parser.add_argument('--backtest', action='store_true', help='Run backtest only')
    parser.add_argument('--once', action='store_true', help='Run one check cycle and exit')
    parser.add_argument('--days', type=int, default=180, help='Days of data for backtest')
    args = parser.parse_args()

    if args.backtest:
        # Run backtest
        logger.log_info("Starting backtest...")
        backtester = Backtester()
        results = backtester.run_backtest(show_plot=False)
        backtester.save_results()
        return

    # Run live trading bot
    bot = TradingBot()
    bot.run(once=args.once)


if __name__ == "__main__":
    main()
