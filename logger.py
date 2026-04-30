"""
Trade logging module.
Logs trades to CSV and console with timestamps.
"""
import os
import csv
from datetime import datetime
from typing import Optional
import config


class TradeLogger:
    """Handles all trade logging to CSV and console."""

    def __init__(self, log_file: str = None):
        self.log_file = log_file or config.LOG_FILE_PATH
        self.console_log = config.CONSOLE_LOG
        self._ensure_log_directory()
        self._initialize_csv()

    def _ensure_log_directory(self):
        """Create logs directory if it doesn't exist."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'action',
                    'symbol',
                    'side',
                    'price',
                    'quantity',
                    'value',
                    'stop_loss',
                    'take_profit',
                    'pnl',
                    'pnl_pct',
                    'balance',
                    'reason'
                ])

    def _timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _console(self, message: str, level: str = "INFO"):
        """Print to console if enabled."""
        if self.console_log:
            timestamp = self._timestamp()
            color = {
                "INFO": "\033[97m",      # White
                "BUY": "\033[92m",       # Green
                "SELL": "\033[91m",      # Red
                "WARN": "\033[93m",      # Yellow
                "ERROR": "\033[91m",     # Red
                "SUCCESS": "\033[96m",   # Cyan
            }.get(level, "\033[97m")
            reset = "\033[0m"
            print(f"{color}[{timestamp}] [{level}] {message}{reset}")

    def log_trade(
        self,
        action: str,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        balance: Optional[float] = None,
        reason: str = ""
    ):
        """Log a trade to CSV and console."""
        value = price * quantity

        # Log to CSV
        if config.LOG_TO_FILE:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self._timestamp(),
                    action,
                    symbol,
                    side,
                    f"{price:.2f}",
                    f"{quantity:.6f}",
                    f"{value:.2f}",
                    f"{stop_loss:.2f}" if stop_loss else "",
                    f"{take_profit:.2f}" if take_profit else "",
                    f"{pnl:.2f}" if pnl else "",
                    f"{pnl_pct:.2f}" if pnl_pct else "",
                    f"{balance:.2f}" if balance else "",
                    reason
                ])

        # Log to console
        level = "BUY" if side == "buy" else "SELL" if side == "sell" else "INFO"
        msg = f"{action} | {symbol} | {side.upper()} | Price: ${price:.2f} | Qty: {quantity:.6f} | Value: ${value:.2f}"
        if pnl is not None:
            msg += f" | PnL: ${pnl:.2f} ({pnl_pct:.2f}%)"
        if reason:
            msg += f" | Reason: {reason}"
        self._console(msg, level)

    def log_signal(self, symbol: str, signal: str, price: float, indicators: dict):
        """Log a trading signal detection."""
        ind_str = " | ".join([f"{k}: {v:.2f}" for k, v in indicators.items()])
        self._console(f"SIGNAL: {signal} on {symbol} @ ${price:.2f} | {ind_str}", "INFO")

    def log_risk_event(self, event: str, details: str = ""):
        """Log risk management events."""
        msg = f"RISK: {event}"
        if details:
            msg += f" | {details}"
        self._console(msg, "WARN")

    def log_error(self, error: str, details: str = ""):
        """Log errors."""
        msg = f"ERROR: {error}"
        if details:
            msg += f" | {details}"
        self._console(msg, "ERROR")

    def log_info(self, message: str):
        """Log general info."""
        self._console(message, "INFO")

    def log_success(self, message: str):
        """Log success messages."""
        self._console(message, "SUCCESS")

    def log_daily_summary(
        self,
        date: str,
        trades: int,
        wins: int,
        losses: int,
        total_pnl: float,
        balance: float
    ):
        """Log daily trading summary."""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        msg = (
            f"DAILY SUMMARY [{date}] | "
            f"Trades: {trades} | Wins: {wins} | Losses: {losses} | "
            f"Win Rate: {win_rate:.1f}% | PnL: ${total_pnl:.2f} | Balance: ${balance:.2f}"
        )
        self._console(msg, "SUCCESS" if total_pnl >= 0 else "WARN")


# Global logger instance
logger = TradeLogger()
