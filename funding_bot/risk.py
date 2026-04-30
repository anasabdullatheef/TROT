"""
Risk Management Module
Handles position sizing, daily loss limits, and trade validation
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, Optional, List
import config


class RiskManager:
    """
    Manages risk for the trading strategy

    Rules:
    - Max 2% of capital per trade
    - Max 1 open position at a time
    - Skip new signals if daily PnL is already down 3%
    """

    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.current_capital = self.initial_capital
        self.daily_pnl: Dict[date, float] = {}  # Track daily PnL
        self.open_positions: List[Dict] = []
        self.closed_trades: List[Dict] = []

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_pct: float = None
    ) -> Dict:
        """
        Calculate position size based on risk per trade

        Uses fixed percentage risk model:
        - Risk amount = Capital * RISK_PER_TRADE
        - Position size = Risk amount / Stop loss distance

        Returns:
            Dict with position_size, risk_amount, position_value
        """
        if stop_loss_pct is None:
            stop_loss_pct = config.STOP_LOSS_PCT

        risk_amount = self.current_capital * config.RISK_PER_TRADE
        stop_loss_distance = entry_price * stop_loss_pct

        # Position size in base currency (BTC)
        position_size = risk_amount / stop_loss_distance

        # Position value in quote currency (USDT)
        position_value = position_size * entry_price

        return {
            "position_size": position_size,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_pct": config.RISK_PER_TRADE * 100,
            "leverage_required": position_value / self.current_capital
        }

    def can_open_position(self) -> Dict:
        """
        Check if we're allowed to open a new position

        Returns:
            Dict with allowed (bool), reasons (list)
        """
        result = {
            "allowed": True,
            "reasons": []
        }

        # Check max positions
        if len(self.open_positions) >= config.MAX_POSITIONS:
            result["allowed"] = False
            result["reasons"].append(
                f"Max positions reached ({len(self.open_positions)}/{config.MAX_POSITIONS})"
            )

        # Check daily loss limit
        today = date.today()
        daily_pnl = self.get_daily_pnl(today)
        daily_pnl_pct = daily_pnl / self.initial_capital if self.initial_capital > 0 else 0

        if daily_pnl_pct <= -config.DAILY_LOSS_LIMIT:
            result["allowed"] = False
            result["reasons"].append(
                f"Daily loss limit hit: {daily_pnl_pct*100:.2f}% <= -{config.DAILY_LOSS_LIMIT*100}%"
            )

        if result["allowed"]:
            result["reasons"].append("All risk checks passed")

        return result

    def get_daily_pnl(self, target_date: date = None) -> float:
        """Get total PnL for a specific date"""
        if target_date is None:
            target_date = date.today()

        return self.daily_pnl.get(target_date, 0.0)

    def record_trade_pnl(self, pnl: float, trade_date: date = None):
        """Record PnL from a closed trade"""
        if trade_date is None:
            trade_date = date.today()

        if trade_date not in self.daily_pnl:
            self.daily_pnl[trade_date] = 0.0

        self.daily_pnl[trade_date] += pnl

    def open_position(
        self,
        position_type: str,
        entry_price: float,
        size: float,
        timestamp: datetime
    ) -> Dict:
        """
        Register a new open position

        Args:
            position_type: 'long' or 'short'
            entry_price: Entry price
            size: Position size in base currency
            timestamp: Entry timestamp

        Returns:
            Position dict
        """
        position = {
            "type": position_type,
            "entry_price": entry_price,
            "size": size,
            "entry_time": timestamp,
            "value": size * entry_price
        }
        self.open_positions.append(position)
        return position

    def close_position(
        self,
        exit_price: float,
        exit_reason: str,
        timestamp: datetime
    ) -> Optional[Dict]:
        """
        Close the current open position

        Returns:
            Trade result dict or None if no position
        """
        if not self.open_positions:
            return None

        position = self.open_positions.pop(0)

        if position["type"] == "long":
            pnl = (exit_price - position["entry_price"]) * position["size"]
            pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"]
        else:  # short
            pnl = (position["entry_price"] - exit_price) * position["size"]
            pnl_pct = (position["entry_price"] - exit_price) / position["entry_price"]

        trade = {
            "type": position["type"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "size": position["size"],
            "entry_time": position["entry_time"],
            "exit_time": timestamp,
            "exit_reason": exit_reason,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "win": pnl > 0
        }

        self.closed_trades.append(trade)
        self.current_capital += pnl
        self.record_trade_pnl(pnl, timestamp.date())

        return trade

    def get_current_position(self) -> Optional[Dict]:
        """Get the current open position if any"""
        if self.open_positions:
            return self.open_positions[0]
        return None

    def get_statistics(self) -> Dict:
        """Calculate overall trading statistics"""
        if not self.closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "avg_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "current_capital": self.current_capital,
                "return_pct": 0
            }

        trades_df = pd.DataFrame(self.closed_trades)

        total_trades = len(trades_df)
        winning = trades_df[trades_df["win"]]
        losing = trades_df[~trades_df["win"]]

        total_wins = winning["pnl"].sum() if len(winning) > 0 else 0
        total_losses = abs(losing["pnl"].sum()) if len(losing) > 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / total_trades * 100 if total_trades > 0 else 0,
            "total_pnl": trades_df["pnl"].sum(),
            "total_pnl_pct": (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            "avg_pnl": trades_df["pnl"].mean(),
            "avg_win": winning["pnl"].mean() if len(winning) > 0 else 0,
            "avg_loss": losing["pnl"].mean() if len(losing) > 0 else 0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
            "current_capital": self.current_capital,
            "return_pct": (self.current_capital - self.initial_capital) / self.initial_capital * 100
        }

    def reset(self):
        """Reset the risk manager to initial state"""
        self.current_capital = self.initial_capital
        self.daily_pnl = {}
        self.open_positions = []
        self.closed_trades = []


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown from an equity curve

    Args:
        equity_curve: Series of portfolio values over time

    Returns:
        Maximum drawdown as a decimal (e.g., 0.15 = 15%)
    """
    if len(equity_curve) == 0:
        return 0.0

    # Calculate running maximum
    running_max = equity_curve.cummax()

    # Calculate drawdown at each point
    drawdown = (equity_curve - running_max) / running_max

    # Return the maximum drawdown (most negative value)
    return abs(drawdown.min())


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760  # Hourly data: 24 * 365
) -> float:
    """
    Calculate annualized Sharpe ratio

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of periods in a year (8760 for hourly)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    return (excess_returns.mean() / excess_returns.std()) * (periods_per_year ** 0.5)


if __name__ == "__main__":
    # Test the module
    print("Testing risk management module...")

    rm = RiskManager(initial_capital=10000)

    # Test position sizing
    sizing = rm.calculate_position_size(entry_price=40000)
    print(f"\nPosition Sizing at $40,000:")
    print(f"  Position size: {sizing['position_size']:.6f} BTC")
    print(f"  Position value: ${sizing['position_value']:.2f}")
    print(f"  Risk amount: ${sizing['risk_amount']:.2f}")
    print(f"  Leverage required: {sizing['leverage_required']:.2f}x")

    # Test can open position
    can_open = rm.can_open_position()
    print(f"\nCan open position: {can_open}")

    # Simulate some trades
    print("\n--- Simulating trades ---")
    from datetime import datetime, timedelta

    # Open a long position
    pos = rm.open_position(
        position_type="long",
        entry_price=40000,
        size=0.0133,  # ~$533 position
        timestamp=datetime.now()
    )
    print(f"Opened long: {pos}")

    # Check if we can open another
    can_open = rm.can_open_position()
    print(f"Can open another: {can_open}")

    # Close the position with profit
    trade = rm.close_position(
        exit_price=40850,  # +2.125%
        exit_reason="Take profit",
        timestamp=datetime.now() + timedelta(hours=5)
    )
    print(f"Closed trade: PnL=${trade['pnl']:.2f} ({trade['pnl_pct']*100:.2f}%)")

    # Get statistics
    stats = rm.get_statistics()
    print(f"\nStatistics: {stats}")

    # Test drawdown calculation
    equity = pd.Series([10000, 10200, 10500, 10300, 9800, 10100, 10400])
    max_dd = calculate_max_drawdown(equity)
    print(f"\nMax Drawdown test: {max_dd*100:.2f}%")

    # Test Sharpe ratio
    returns = pd.Series([0.001, -0.002, 0.003, 0.001, -0.001, 0.002])
    sharpe = calculate_sharpe_ratio(returns)
    print(f"Sharpe ratio test: {sharpe:.2f}")
