"""
Risk Management Module
- Kelly Criterion position sizing (capped at quarter Kelly)
- Dynamic TP/SL based on ATR
- Daily drawdown kill switch
- Macro event blackout periods
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List, Tuple

import config


class RiskManager:
    """
    Advanced risk management with Kelly sizing and dynamic stops
    """

    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.current_capital = self.initial_capital
        self.daily_pnl: Dict[date, float] = {}
        self.open_positions: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.kill_switch_active = False
        self.kill_switch_date: Optional[date] = None

        # Historical win rate for Kelly calculation
        self.win_rate = 0.55  # Default assumption
        self.avg_win = 0.02   # 2% average win
        self.avg_loss = 0.015  # 1.5% average loss

    def calculate_kelly_fraction(
        self,
        win_rate: float = None,
        avg_win: float = None,
        avg_loss: float = None
    ) -> float:
        """
        Calculate Kelly Criterion fraction

        Kelly = (p * b - q) / b
        where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = ratio of average win to average loss
        """
        p = win_rate or self.win_rate
        q = 1 - p
        avg_w = avg_win or self.avg_win
        avg_l = avg_loss or self.avg_loss

        if avg_l == 0:
            return 0

        b = avg_w / avg_l
        kelly = (p * b - q) / b

        # Cap at quarter Kelly for safety
        kelly = max(0, min(kelly, 1)) * config.KELLY_FRACTION

        return kelly

    def calculate_position_size(
        self,
        entry_price: float,
        atr: float,
        confidence: float = 0.5
    ) -> Dict:
        """
        Calculate position size using Kelly Criterion with ATR-based stops

        Args:
            entry_price: Entry price
            atr: Current ATR for stop calculation
            confidence: Model confidence (used to scale Kelly)

        Returns:
            Dict with position_size, risk_amount, sl_price, tp_price
        """
        # Calculate dynamic stop/take profit
        sl_distance = atr * config.ATR_STOP_LOSS_MULTIPLIER
        tp_distance = atr * config.ATR_TAKE_PROFIT_MULTIPLIER

        sl_pct = sl_distance / entry_price
        tp_pct = tp_distance / entry_price

        # Update Kelly inputs based on TP/SL ratio
        avg_win_estimate = tp_pct
        avg_loss_estimate = sl_pct

        # Calculate Kelly fraction
        kelly = self.calculate_kelly_fraction(
            win_rate=max(0.4, min(0.7, confidence)),  # Use confidence as win rate proxy
            avg_win=avg_win_estimate,
            avg_loss=avg_loss_estimate
        )

        # Scale Kelly by confidence
        kelly_scaled = kelly * confidence

        # Cap at max capital per trade
        position_pct = min(kelly_scaled, config.MAX_CAPITAL_PER_TRADE)

        # Calculate actual position size
        risk_amount = self.current_capital * position_pct
        position_value = risk_amount / sl_pct  # Size based on risk
        position_size = position_value / entry_price

        return {
            "position_size": position_size,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_pct": position_pct * 100,
            "kelly_fraction": kelly,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "sl_pct": sl_pct * 100,
            "tp_pct": tp_pct * 100
        }

    def check_daily_drawdown(self, current_date: date = None) -> Tuple[bool, float]:
        """
        Check if daily drawdown limit is breached

        Returns:
            (kill_switch_triggered, daily_pnl_pct)
        """
        if current_date is None:
            current_date = date.today()

        daily_pnl = self.daily_pnl.get(current_date, 0)
        daily_pnl_pct = daily_pnl / self.initial_capital

        if daily_pnl_pct <= -config.DAILY_DRAWDOWN_LIMIT:
            self.kill_switch_active = True
            self.kill_switch_date = current_date
            return True, daily_pnl_pct

        # Reset kill switch on new day
        if self.kill_switch_date and current_date > self.kill_switch_date:
            self.kill_switch_active = False
            self.kill_switch_date = None

        return False, daily_pnl_pct

    def is_macro_blackout(self, current_time: datetime = None) -> Tuple[bool, str]:
        """
        Check if we're in a macro event blackout period

        Returns:
            (is_blackout, event_description)
        """
        if current_time is None:
            current_time = datetime.now()

        for event in config.MACRO_EVENTS:
            event_month, event_day, event_hour = event
            event_time = current_time.replace(
                month=event_month,
                day=event_day,
                hour=event_hour,
                minute=0,
                second=0
            )

            # Check if within blackout window
            blackout_start = event_time - timedelta(hours=config.MACRO_BLACKOUT_HOURS)
            blackout_end = event_time + timedelta(hours=config.MACRO_BLACKOUT_HOURS)

            if blackout_start <= current_time <= blackout_end:
                return True, f"Macro event at {event_time}"

        return False, ""

    def can_open_position(self, current_time: datetime = None) -> Dict:
        """
        Check if we can open a new position

        Returns:
            Dict with allowed (bool), reasons (list)
        """
        if current_time is None:
            current_time = datetime.now()

        result = {
            "allowed": True,
            "reasons": []
        }

        # Check max positions
        if len(self.open_positions) >= config.MAX_POSITIONS:
            result["allowed"] = False
            result["reasons"].append(f"Max positions reached ({len(self.open_positions)}/{config.MAX_POSITIONS})")

        # Check daily drawdown
        kill_switch, daily_pnl = self.check_daily_drawdown(current_time.date())
        if kill_switch:
            result["allowed"] = False
            result["reasons"].append(f"Daily drawdown limit hit: {daily_pnl*100:.2f}%")

        # Check macro blackout
        is_blackout, event = self.is_macro_blackout(current_time)
        if is_blackout:
            result["allowed"] = False
            result["reasons"].append(f"Macro blackout: {event}")

        if result["allowed"]:
            result["reasons"].append("All risk checks passed")

        return result

    def open_position(
        self,
        position_type: str,
        entry_price: float,
        size: float,
        timestamp: datetime,
        atr: float,
        confidence: float,
        regime: int
    ) -> Dict:
        """Register a new position"""
        # Calculate TP/SL prices
        if position_type == "long":
            sl_price = entry_price - (atr * config.ATR_STOP_LOSS_MULTIPLIER)
            tp_price = entry_price + (atr * config.ATR_TAKE_PROFIT_MULTIPLIER)
        else:
            sl_price = entry_price + (atr * config.ATR_STOP_LOSS_MULTIPLIER)
            tp_price = entry_price - (atr * config.ATR_TAKE_PROFIT_MULTIPLIER)

        position = {
            "type": position_type,
            "entry_price": entry_price,
            "size": size,
            "entry_time": timestamp,
            "value": size * entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "atr": atr,
            "confidence": confidence,
            "regime": regime
        }

        self.open_positions.append(position)
        return position

    def close_position(
        self,
        exit_price: float,
        exit_reason: str,
        timestamp: datetime
    ) -> Optional[Dict]:
        """Close the current position"""
        if not self.open_positions:
            return None

        position = self.open_positions.pop(0)

        if position["type"] == "long":
            pnl = (exit_price - position["entry_price"]) * position["size"]
            pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"]
        else:
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
            "win": pnl > 0,
            "confidence": position["confidence"],
            "regime": position["regime"]
        }

        self.closed_trades.append(trade)
        self.current_capital += pnl

        # Update daily PnL
        trade_date = timestamp.date()
        if trade_date not in self.daily_pnl:
            self.daily_pnl[trade_date] = 0
        self.daily_pnl[trade_date] += pnl

        # Update win rate for Kelly calculation
        self._update_kelly_inputs()

        return trade

    def _update_kelly_inputs(self):
        """Update win rate and average win/loss from trade history"""
        if len(self.closed_trades) < 10:
            return

        trades_df = pd.DataFrame(self.closed_trades)
        winning = trades_df[trades_df["win"]]
        losing = trades_df[~trades_df["win"]]

        if len(trades_df) > 0:
            self.win_rate = len(winning) / len(trades_df)

        if len(winning) > 0:
            self.avg_win = winning["pnl_pct"].mean()

        if len(losing) > 0:
            self.avg_loss = abs(losing["pnl_pct"].mean())

    def get_current_position(self) -> Optional[Dict]:
        """Get current open position"""
        return self.open_positions[0] if self.open_positions else None

    def get_statistics(self) -> Dict:
        """Calculate overall statistics"""
        if not self.closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
                "profit_factor": 0,
                "current_capital": self.current_capital,
                "return_pct": 0,
                "current_kelly": self.calculate_kelly_fraction()
            }

        trades_df = pd.DataFrame(self.closed_trades)
        winning = trades_df[trades_df["win"]]
        losing = trades_df[~trades_df["win"]]

        total_wins = winning["pnl"].sum() if len(winning) > 0 else 0
        total_losses = abs(losing["pnl"].sum()) if len(losing) > 0 else 0

        return {
            "total_trades": len(trades_df),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(trades_df) * 100,
            "total_pnl": trades_df["pnl"].sum(),
            "avg_pnl": trades_df["pnl"].mean(),
            "avg_win": winning["pnl"].mean() if len(winning) > 0 else 0,
            "avg_loss": losing["pnl"].mean() if len(losing) > 0 else 0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
            "current_capital": self.current_capital,
            "return_pct": (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            "current_kelly": self.calculate_kelly_fraction()
        }

    def reset(self):
        """Reset to initial state"""
        self.current_capital = self.initial_capital
        self.daily_pnl = {}
        self.open_positions = []
        self.closed_trades = []
        self.kill_switch_active = False
        self.kill_switch_date = None
        self.win_rate = 0.55
        self.avg_win = 0.02
        self.avg_loss = 0.015


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from equity curve"""
    if len(equity_curve) == 0:
        return 0.0

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return abs(drawdown.min())


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 8760) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    return (returns.mean() / returns.std()) * (periods_per_year ** 0.5)


def calculate_sortino_ratio(returns: pd.Series, periods_per_year: int = 8760) -> float:
    """Calculate annualized Sortino ratio (downside deviation only)"""
    if len(returns) == 0:
        return 0.0

    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float("inf") if returns.mean() > 0 else 0.0

    downside_std = downside_returns.std()
    return (returns.mean() / downside_std) * (periods_per_year ** 0.5)


if __name__ == "__main__":
    print("Testing risk module...")

    rm = RiskManager(initial_capital=10000)

    # Test Kelly calculation
    kelly = rm.calculate_kelly_fraction(win_rate=0.55, avg_win=0.02, avg_loss=0.015)
    print(f"Kelly fraction (55% WR, 2%W/1.5%L): {kelly:.4f} ({kelly*100:.2f}%)")

    # Test position sizing with ATR
    sizing = rm.calculate_position_size(
        entry_price=50000,
        atr=1000,
        confidence=0.7
    )
    print(f"\nPosition sizing at $50k, ATR=$1000, conf=70%:")
    print(f"  Position size: {sizing['position_size']:.6f} BTC")
    print(f"  Position value: ${sizing['position_value']:.2f}")
    print(f"  Risk amount: ${sizing['risk_amount']:.2f} ({sizing['risk_pct']:.2f}%)")
    print(f"  SL distance: ${sizing['sl_distance']:.0f} ({sizing['sl_pct']:.2f}%)")
    print(f"  TP distance: ${sizing['tp_distance']:.0f} ({sizing['tp_pct']:.2f}%)")

    # Test can open position
    can_open = rm.can_open_position()
    print(f"\nCan open position: {can_open}")

    # Simulate trades
    from datetime import datetime

    pos = rm.open_position(
        position_type="long",
        entry_price=50000,
        size=0.01,
        timestamp=datetime.now(),
        atr=1000,
        confidence=0.7,
        regime=1
    )
    print(f"\nOpened position: SL=${pos['sl_price']:.0f}, TP=${pos['tp_price']:.0f}")

    trade = rm.close_position(
        exit_price=51500,
        exit_reason="Take profit",
        timestamp=datetime.now()
    )
    print(f"Closed: PnL=${trade['pnl']:.2f} ({trade['pnl_pct']*100:.2f}%)")

    stats = rm.get_statistics()
    print(f"\nStats: {stats}")

    print("\n✅ Risk module test passed")
