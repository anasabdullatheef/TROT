"""
Risk Manager - Leg protection, daily loss limits, circuit breakers
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum

from monitor import PriceData
from executor import ArbitragePosition, PositionStatus
import config


class RiskAction(Enum):
    NONE = "none"
    CLOSE_POSITION = "close_position"
    HALT_TRADING = "halt_trading"


@dataclass
class RiskState:
    """Current risk state"""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_losses: int = 0
    trading_halted: bool = False
    halt_reason: Optional[str] = None
    last_reset_date: date = field(default_factory=date.today)


class RiskManager:
    """Manages all risk controls for the arbitrage bot"""

    def __init__(self):
        self.state = RiskState()
        self.position_entry_spreads: Dict[str, float] = {}

    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.state = RiskState()
        print("Daily risk stats reset")

    def check_daily_reset(self):
        """Check if we need to reset daily stats"""
        today = date.today()
        if self.state.last_reset_date != today:
            self.reset_daily_stats()
            self.state.last_reset_date = today

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed"""
        self.check_daily_reset()

        if self.state.trading_halted:
            return False, self.state.halt_reason or "Trading halted"

        if self.state.daily_pnl <= -config.MAX_DAILY_LOSS_USD:
            self.state.trading_halted = True
            self.state.halt_reason = f"Daily loss limit hit: ${self.state.daily_pnl:.2f}"
            return False, self.state.halt_reason

        return True, "OK"

    def record_trade_result(self, pnl: float):
        """Record the result of a trade"""
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1

        if pnl < 0:
            self.state.daily_losses += 1

        # Check daily loss limit
        if self.state.daily_pnl <= -config.MAX_DAILY_LOSS_USD:
            self.state.trading_halted = True
            self.state.halt_reason = f"Daily loss limit hit: ${self.state.daily_pnl:.2f}"
            print(f"RISK: Daily loss limit reached. Trading halted.")

    def check_single_trade_loss(self, pnl: float) -> bool:
        """Check if a single trade exceeds max loss"""
        if pnl <= -config.MAX_SINGLE_TRADE_LOSS_USD:
            print(f"RISK: Single trade loss ${pnl:.2f} exceeds limit ${config.MAX_SINGLE_TRADE_LOSS_USD}")
            return True
        return False

    def register_position(self, position_id: str, entry_spread: float):
        """Register a new position's entry spread for circuit breaker monitoring"""
        self.position_entry_spreads[position_id] = entry_spread

    def unregister_position(self, position_id: str):
        """Unregister a closed position"""
        if position_id in self.position_entry_spreads:
            del self.position_entry_spreads[position_id]

    def check_circuit_breaker(
        self,
        position: ArbitragePosition,
        current_prices: Dict[str, PriceData]
    ) -> RiskAction:
        """
        Check if circuit breaker should trigger for a position.

        If spread moves against us by more than 0.1%, exit immediately.
        """
        buy_ex = position.opportunity.buy_exchange
        sell_ex = position.opportunity.sell_exchange

        if buy_ex not in current_prices or sell_ex not in current_prices:
            return RiskAction.NONE

        # Current spread (for closing: we would sell what we bought, buy what we sold)
        # To close: sell on buy_exchange (at bid), buy on sell_exchange (at ask)
        current_sell_price = current_prices[buy_ex].bid
        current_buy_price = current_prices[sell_ex].ask

        # Calculate current closing P&L
        entry_buy_price = position.opportunity.buy_price
        entry_sell_price = position.opportunity.sell_price

        # Entry: bought at entry_buy_price, sold at entry_sell_price
        # Exit: sell at current_sell_price, buy at current_buy_price
        # P&L = (entry_sell_price - entry_buy_price) + (current_sell_price - current_buy_price)

        entry_spread = (entry_sell_price - entry_buy_price) / entry_buy_price
        exit_spread = (current_sell_price - current_buy_price) / current_buy_price

        # Adverse move = entry_spread - exit_spread (if this is positive, spread converged which is good)
        # If this is negative and > 0.1%, that's bad
        adverse_move = -(exit_spread)  # Simplified: if exit spread is very negative

        # Actually, let's think about this more carefully:
        # Entry: we locked in entry_spread profit
        # Exit: we need to pay exit costs
        # If exit_spread is very negative (we have to sell low, buy high), that's bad

        if exit_spread < -config.CIRCUIT_BREAKER_SPREAD_MOVE:
            print(f"CIRCUIT BREAKER: Position {position.position_id} - "
                  f"Exit spread {exit_spread*100:.3f}% exceeded limit")
            return RiskAction.CLOSE_POSITION

        return RiskAction.NONE

    def check_unhedged_timeout(
        self,
        position: ArbitragePosition
    ) -> RiskAction:
        """
        Check if position has been open too long without being hedged.
        Max 30 seconds for unhedged positions.
        """
        if position.status != PositionStatus.OPEN:
            return RiskAction.NONE

        elapsed = (datetime.now() - position.entry_time).total_seconds()

        if elapsed > config.MAX_UNHEDGED_SECONDS:
            print(f"RISK: Position {position.position_id} exceeded max hold time "
                  f"({elapsed:.1f}s > {config.MAX_UNHEDGED_SECONDS}s)")
            return RiskAction.CLOSE_POSITION

        return RiskAction.NONE

    def get_risk_summary(self) -> Dict:
        """Get current risk state summary"""
        return {
            "daily_pnl": self.state.daily_pnl,
            "daily_trades": self.state.daily_trades,
            "daily_losses": self.state.daily_losses,
            "trading_halted": self.state.trading_halted,
            "halt_reason": self.state.halt_reason,
            "max_daily_loss": config.MAX_DAILY_LOSS_USD,
            "remaining_loss_budget": config.MAX_DAILY_LOSS_USD + self.state.daily_pnl
        }

    def force_halt(self, reason: str):
        """Force halt all trading"""
        self.state.trading_halted = True
        self.state.halt_reason = reason
        print(f"RISK: Trading force halted - {reason}")

    def resume_trading(self):
        """Resume trading (manual override)"""
        self.state.trading_halted = False
        self.state.halt_reason = None
        print("RISK: Trading resumed")
