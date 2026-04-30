"""
Trade Executor - Places simultaneous orders on both exchanges
Handles leg risk protection
"""

import asyncio
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

import ccxt.async_support as ccxt

from arbitrage import ArbitrageOpportunity
import config


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    EMERGENCY_CLOSED = "emergency_closed"


@dataclass
class Order:
    """Represents a single order"""
    order_id: str
    exchange: str
    symbol: str
    side: str  # "buy" or "sell"
    amount: float  # in base currency (BTC)
    price: float  # expected price
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_amount: Optional[float] = None
    exchange_order_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


@dataclass
class ArbitragePosition:
    """Represents a complete arbitrage position (buy leg + sell leg)"""
    position_id: str
    opportunity: ArbitrageOpportunity
    buy_order: Order
    sell_order: Order
    status: PositionStatus = PositionStatus.OPEN
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    notes: str = ""


class TradeExecutor:
    """Executes arbitrage trades across exchanges"""

    def __init__(self, exchanges: Dict[str, ccxt.Exchange]):
        self.exchanges = exchanges
        self.open_positions: Dict[str, ArbitragePosition] = {}
        self.closed_positions: list = []

    async def execute_arbitrage(
        self,
        opportunity: ArbitrageOpportunity
    ) -> Optional[ArbitragePosition]:
        """
        Execute an arbitrage trade.

        Places buy and sell orders simultaneously.
        If one leg fails, closes the other immediately.
        """
        if not config.LIVE_TRADING:
            # Observation mode - don't execute
            return None

        if len(self.open_positions) >= config.MAX_SIMULTANEOUS_POSITIONS:
            print(f"Max positions reached ({config.MAX_SIMULTANEOUS_POSITIONS})")
            return None

        position_id = str(uuid.uuid4())[:8]

        # Calculate order size in BTC
        btc_amount = config.TRADE_SIZE_USD / opportunity.buy_price

        # Create orders
        buy_order = Order(
            order_id=f"{position_id}-buy",
            exchange=opportunity.buy_exchange,
            symbol=config.SYMBOL,
            side="buy",
            amount=btc_amount,
            price=opportunity.buy_price
        )

        sell_order = Order(
            order_id=f"{position_id}-sell",
            exchange=opportunity.sell_exchange,
            symbol=config.SYMBOL,
            side="sell",
            amount=btc_amount,
            price=opportunity.sell_price
        )

        # Create position
        position = ArbitragePosition(
            position_id=position_id,
            opportunity=opportunity,
            buy_order=buy_order,
            sell_order=sell_order
        )

        print(f"Executing arbitrage {position_id}: "
              f"Buy {btc_amount:.6f} BTC on {opportunity.buy_exchange} @ ${opportunity.buy_price:,.2f}, "
              f"Sell on {opportunity.sell_exchange} @ ${opportunity.sell_price:,.2f}")

        # Execute both legs simultaneously
        buy_task = self._place_order(buy_order)
        sell_task = self._place_order(sell_order)

        results = await asyncio.gather(buy_task, sell_task, return_exceptions=True)

        buy_success = isinstance(results[0], Order) and results[0].status == OrderStatus.FILLED
        sell_success = isinstance(results[1], Order) and results[1].status == OrderStatus.FILLED

        if buy_success and sell_success:
            # Both legs filled successfully
            position.status = PositionStatus.OPEN
            position.buy_order = results[0]
            position.sell_order = results[1]
            self.open_positions[position_id] = position
            print(f"Position {position_id} opened successfully")
            return position

        elif buy_success and not sell_success:
            # Buy filled but sell failed - EMERGENCY: close buy immediately
            print(f"LEG RISK: Sell failed, closing buy leg immediately")
            await self._emergency_close(buy_order)
            position.status = PositionStatus.EMERGENCY_CLOSED
            position.notes = "Sell leg failed, buy leg emergency closed"
            self.closed_positions.append(position)
            return position

        elif sell_success and not buy_success:
            # Sell filled but buy failed - EMERGENCY: close sell immediately
            print(f"LEG RISK: Buy failed, closing sell leg immediately")
            await self._emergency_close(sell_order)
            position.status = PositionStatus.EMERGENCY_CLOSED
            position.notes = "Buy leg failed, sell leg emergency closed"
            self.closed_positions.append(position)
            return position

        else:
            # Both failed
            position.status = PositionStatus.CLOSED
            position.notes = "Both legs failed"
            self.closed_positions.append(position)
            return position

    async def _place_order(self, order: Order) -> Order:
        """Place a single order on an exchange"""
        if order.exchange not in self.exchanges:
            order.status = OrderStatus.FAILED
            order.error = f"Exchange {order.exchange} not connected"
            return order

        exchange = self.exchanges[order.exchange]

        try:
            # Place market order
            result = await exchange.create_order(
                symbol=order.symbol,
                type="market",
                side=order.side,
                amount=order.amount
            )

            order.exchange_order_id = result.get("id")
            order.filled_price = result.get("average") or result.get("price")
            order.filled_amount = result.get("filled")
            order.status = OrderStatus.FILLED

            print(f"Order {order.order_id} filled: {order.side} {order.filled_amount:.6f} @ ${order.filled_price:,.2f}")

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error = str(e)
            print(f"Order {order.order_id} failed: {e}")

        return order

    async def _emergency_close(self, order: Order) -> bool:
        """Emergency close an order - reverse the position at market"""
        if order.exchange not in self.exchanges:
            return False

        exchange = self.exchanges[order.exchange]

        try:
            # Reverse the order
            reverse_side = "sell" if order.side == "buy" else "buy"
            amount = order.filled_amount or order.amount

            result = await exchange.create_order(
                symbol=order.symbol,
                type="market",
                side=reverse_side,
                amount=amount
            )

            print(f"Emergency close executed: {reverse_side} {amount:.6f}")
            return True

        except Exception as e:
            print(f"Emergency close FAILED: {e}")
            return False

    async def close_position(
        self,
        position_id: str,
        reason: str = "manual"
    ) -> bool:
        """Close an open arbitrage position"""
        if position_id not in self.open_positions:
            return False

        position = self.open_positions[position_id]

        # Close both legs
        # For the buy leg: sell
        # For the sell leg: buy back

        buy_exchange = self.exchanges.get(position.buy_order.exchange)
        sell_exchange = self.exchanges.get(position.sell_order.exchange)

        if not buy_exchange or not sell_exchange:
            return False

        try:
            # Close buy leg (sell)
            close_buy = buy_exchange.create_order(
                symbol=config.SYMBOL,
                type="market",
                side="sell",
                amount=position.buy_order.filled_amount or position.buy_order.amount
            )

            # Close sell leg (buy back)
            close_sell = sell_exchange.create_order(
                symbol=config.SYMBOL,
                type="market",
                side="buy",
                amount=position.sell_order.filled_amount or position.sell_order.amount
            )

            results = await asyncio.gather(close_buy, close_sell, return_exceptions=True)

            # Calculate PnL
            # Buy leg PnL: sell_close_price - buy_entry_price
            # Sell leg PnL: sell_entry_price - buy_close_price

            position.status = PositionStatus.CLOSED
            position.exit_time = datetime.now()
            position.notes = f"Closed: {reason}"

            del self.open_positions[position_id]
            self.closed_positions.append(position)

            print(f"Position {position_id} closed: {reason}")
            return True

        except Exception as e:
            print(f"Error closing position {position_id}: {e}")
            return False

    def get_open_positions(self) -> Dict[str, ArbitragePosition]:
        """Get all open positions"""
        return self.open_positions.copy()

    def get_position_count(self) -> int:
        """Get number of open positions"""
        return len(self.open_positions)
