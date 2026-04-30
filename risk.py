"""
Risk management module.
Handles position sizing, limits, and drawdown protection.
"""
from datetime import datetime, date
from typing import Tuple, Optional
import config
from logger import logger


class RiskManager:
    """
    Risk management for the trading bot.

    Rules:
        - Max 2% of capital per trade
        - 1.5% stop-loss, 3% take-profit
        - Max 3 open positions
        - 5% daily drawdown kill switch
    """

    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.current_capital = self.initial_capital
        self.daily_start_capital = self.initial_capital
        self.last_reset_date = date.today()
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.is_killed = False
        self.kill_reason = ""

        # Track trades for statistics
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0

    def reset_daily_stats(self):
        """Reset daily statistics at the start of a new day."""
        today = date.today()
        if today > self.last_reset_date:
            # Log previous day's summary if there were trades
            if self.trades_today > 0:
                logger.log_daily_summary(
                    date=self.last_reset_date.isoformat(),
                    trades=self.trades_today,
                    wins=self.wins_today,
                    losses=self.losses_today,
                    total_pnl=self.daily_pnl,
                    balance=self.current_capital
                )

            # Reset for new day
            self.daily_start_capital = self.current_capital
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.wins_today = 0
            self.losses_today = 0
            self.last_reset_date = today
            self.is_killed = False
            self.kill_reason = ""

            logger.log_info(f"Daily stats reset. Starting capital: ${self.current_capital:.2f}")

    def calculate_position_size(
        self,
        current_price: float,
        stop_loss_price: float
    ) -> Tuple[float, float]:
        """
        Calculate position size based on risk parameters.

        Args:
            current_price: Current asset price
            stop_loss_price: Stop-loss price level

        Returns:
            Tuple of (quantity, position_value)
        """
        # Max amount to risk on this trade (2% of capital)
        max_risk = self.current_capital * config.RISK_PER_TRADE

        # Risk per unit (difference between entry and stop-loss)
        risk_per_unit = abs(current_price - stop_loss_price)

        if risk_per_unit <= 0:
            logger.log_error("Invalid stop-loss calculation")
            return 0.0, 0.0

        # Calculate quantity based on risk
        quantity = max_risk / risk_per_unit

        # Calculate total position value
        position_value = quantity * current_price

        # Ensure we don't exceed available capital
        max_position_value = self.current_capital * 0.95  # Keep 5% buffer
        if position_value > max_position_value:
            quantity = max_position_value / current_price
            position_value = quantity * current_price

        return quantity, position_value

    def can_open_position(self, open_positions: int) -> Tuple[bool, str]:
        """
        Check if we can open a new position.

        Args:
            open_positions: Current number of open positions

        Returns:
            Tuple of (can_open, reason)
        """
        self.reset_daily_stats()

        # Check kill switch
        if self.is_killed:
            return False, f"Trading killed: {self.kill_reason}"

        # Check max positions
        if open_positions >= config.MAX_OPEN_POSITIONS:
            return False, f"Max positions reached ({config.MAX_OPEN_POSITIONS})"

        # Check daily drawdown
        daily_drawdown = (self.daily_start_capital - self.current_capital) / self.daily_start_capital
        if daily_drawdown >= config.DAILY_DRAWDOWN_LIMIT:
            self.is_killed = True
            self.kill_reason = f"Daily drawdown limit ({config.DAILY_DRAWDOWN_LIMIT*100}%) exceeded"
            logger.log_risk_event("KILL_SWITCH_ACTIVATED", self.kill_reason)
            return False, self.kill_reason

        # Check if we have enough capital
        min_position = 10.0  # Minimum USDT to trade
        if self.current_capital < min_position:
            return False, f"Insufficient capital (${self.current_capital:.2f})"

        return True, "OK"

    def update_capital(self, pnl: float, is_win: bool):
        """
        Update capital after a trade closes.

        Args:
            pnl: Profit/loss from the trade
            is_win: Whether the trade was profitable
        """
        self.current_capital += pnl
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.trades_today += 1

        if is_win:
            self.wins_today += 1
        else:
            self.losses_today += 1

        logger.log_info(
            f"Capital updated: ${self.current_capital:.2f} | "
            f"Daily PnL: ${self.daily_pnl:.2f} | "
            f"Total PnL: ${self.total_pnl:.2f}"
        )

        # Check for kill switch after update
        self._check_drawdown()

    def _check_drawdown(self):
        """Check if daily drawdown limit is hit."""
        if self.daily_start_capital > 0:
            daily_drawdown = (self.daily_start_capital - self.current_capital) / self.daily_start_capital

            if daily_drawdown >= config.DAILY_DRAWDOWN_LIMIT:
                self.is_killed = True
                self.kill_reason = f"Daily drawdown ({daily_drawdown*100:.1f}%) exceeded limit"
                logger.log_risk_event("KILL_SWITCH_ACTIVATED", self.kill_reason)

    def get_risk_metrics(self) -> dict:
        """Get current risk metrics."""
        daily_drawdown = 0.0
        if self.daily_start_capital > 0:
            daily_drawdown = (self.daily_start_capital - self.current_capital) / self.daily_start_capital

        total_drawdown = 0.0
        if self.initial_capital > 0:
            total_drawdown = (self.initial_capital - self.current_capital) / self.initial_capital

        return {
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'daily_drawdown_pct': daily_drawdown * 100,
            'total_drawdown_pct': total_drawdown * 100,
            'trades_today': self.trades_today,
            'win_rate_today': (self.wins_today / self.trades_today * 100) if self.trades_today > 0 else 0,
            'is_killed': self.is_killed,
            'kill_reason': self.kill_reason
        }

    def validate_stop_loss(self, entry_price: float, stop_loss: float) -> bool:
        """Validate that stop-loss is within acceptable range."""
        sl_pct = abs(entry_price - stop_loss) / entry_price

        # Stop-loss should be close to configured percentage
        if sl_pct > config.STOP_LOSS_PCT * 1.5:
            logger.log_risk_event(
                "STOP_LOSS_VALIDATION_FAILED",
                f"SL {sl_pct*100:.1f}% exceeds limit {config.STOP_LOSS_PCT*100}%"
            )
            return False

        return True

    def validate_take_profit(self, entry_price: float, take_profit: float) -> bool:
        """Validate that take-profit is within acceptable range."""
        tp_pct = abs(take_profit - entry_price) / entry_price

        # Take-profit should be at least the configured percentage
        if tp_pct < config.TAKE_PROFIT_PCT * 0.5:
            logger.log_risk_event(
                "TAKE_PROFIT_VALIDATION_WARNING",
                f"TP {tp_pct*100:.1f}% below recommended {config.TAKE_PROFIT_PCT*100}%"
            )
            # Don't fail, just warn

        return True


# Global risk manager instance
risk_manager = RiskManager()
