"""
Logger Module
Handles logging to CSV files and console output
"""

import csv
import os
from datetime import datetime
from typing import Dict, Optional, List
import config


class TradeLogger:
    """
    Logs trades and signals to CSV files and console
    """

    def __init__(self, log_file: str = None):
        self.log_file = log_file or config.LOG_FILE

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            "timestamp",
            "action",        # signal, entry, exit
            "direction",     # long, short, None
            "price",
            "size",
            "funding_rate",
            "rsi",
            "adx",
            "pnl",
            "pnl_pct",
            "capital",
            "reason"
        ]
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_signal(
        self,
        timestamp: datetime,
        direction: str,
        price: float,
        funding_rate: float,
        rsi: float,
        adx: float,
        reason: str
    ):
        """Log a trading signal"""
        self._write_row({
            "timestamp": timestamp,
            "action": "signal",
            "direction": direction,
            "price": price,
            "funding_rate": funding_rate,
            "rsi": rsi,
            "adx": adx,
            "reason": reason
        })

        self._console_log(
            f"📊 SIGNAL: {direction.upper()} @ ${price:,.2f} | "
            f"Funding: {funding_rate*100:.4f}% | RSI: {rsi:.1f} | ADX: {adx:.1f}"
        )

    def log_entry(
        self,
        timestamp: datetime,
        direction: str,
        price: float,
        size: float,
        funding_rate: float,
        capital: float,
        reason: str
    ):
        """Log a trade entry"""
        self._write_row({
            "timestamp": timestamp,
            "action": "entry",
            "direction": direction,
            "price": price,
            "size": size,
            "funding_rate": funding_rate,
            "capital": capital,
            "reason": reason
        })

        self._console_log(
            f"🟢 ENTRY: {direction.upper()} {size:.6f} BTC @ ${price:,.2f} | "
            f"Capital: ${capital:,.2f}"
        )

    def log_exit(
        self,
        timestamp: datetime,
        direction: str,
        price: float,
        size: float,
        pnl: float,
        pnl_pct: float,
        capital: float,
        reason: str
    ):
        """Log a trade exit"""
        self._write_row({
            "timestamp": timestamp,
            "action": "exit",
            "direction": direction,
            "price": price,
            "size": size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "capital": capital,
            "reason": reason
        })

        emoji = "🟢" if pnl > 0 else "🔴"
        self._console_log(
            f"{emoji} EXIT: {direction.upper()} @ ${price:,.2f} | "
            f"PnL: ${pnl:,.2f} ({pnl_pct*100:+.2f}%) | "
            f"Capital: ${capital:,.2f} | {reason}"
        )

    def log_skip(
        self,
        timestamp: datetime,
        reason: str
    ):
        """Log when a signal is skipped"""
        self._write_row({
            "timestamp": timestamp,
            "action": "skip",
            "reason": reason
        })

        self._console_log(f"⏭️  SKIP: {reason}")

    def log_info(self, message: str):
        """Log an info message to console"""
        self._console_log(f"ℹ️  {message}")

    def log_error(self, message: str):
        """Log an error message"""
        self._console_log(f"❌ ERROR: {message}")
        self._write_row({
            "timestamp": datetime.now(),
            "action": "error",
            "reason": message
        })

    def log_stats(self, stats: Dict):
        """Log trading statistics summary"""
        self._console_log("\n" + "=" * 60)
        self._console_log("📈 TRADING STATISTICS")
        self._console_log("=" * 60)
        self._console_log(f"Total Trades:     {stats.get('total_trades', 0)}")
        self._console_log(f"Winning Trades:   {stats.get('winning_trades', 0)}")
        self._console_log(f"Losing Trades:    {stats.get('losing_trades', 0)}")
        self._console_log(f"Win Rate:         {stats.get('win_rate', 0):.1f}%")
        self._console_log(f"Total PnL:        ${stats.get('total_pnl', 0):,.2f}")
        self._console_log(f"Return:           {stats.get('return_pct', 0):+.2f}%")
        self._console_log(f"Profit Factor:    {stats.get('profit_factor', 0):.2f}")
        self._console_log(f"Current Capital:  ${stats.get('current_capital', 0):,.2f}")
        self._console_log("=" * 60 + "\n")

    def _write_row(self, data: Dict):
        """Write a row to CSV file"""
        row = [
            data.get("timestamp", ""),
            data.get("action", ""),
            data.get("direction", ""),
            data.get("price", ""),
            data.get("size", ""),
            data.get("funding_rate", ""),
            data.get("rsi", ""),
            data.get("adx", ""),
            data.get("pnl", ""),
            data.get("pnl_pct", ""),
            data.get("capital", ""),
            data.get("reason", "")
        ]

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _console_log(self, message: str):
        """Print message to console with timestamp"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {message}")


class BacktestLogger:
    """
    Specialized logger for backtesting that collects data for analysis
    """

    def __init__(self):
        self.trades: List[Dict] = []
        self.signals: List[Dict] = []
        self.equity_curve: List[Dict] = []

    def log_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        direction: str,
        entry_price: float,
        exit_price: float,
        size: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        funding_rate_entry: float,
        rsi_entry: float,
        adx_entry: float
    ):
        """Log a completed trade"""
        self.trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "funding_rate": funding_rate_entry,
            "rsi": rsi_entry,
            "adx": adx_entry,
            "win": pnl > 0
        })

    def log_equity(self, timestamp: datetime, capital: float, drawdown: float = 0):
        """Log equity curve point"""
        self.equity_curve.append({
            "timestamp": timestamp,
            "capital": capital,
            "drawdown": drawdown
        })

    def log_signal(
        self,
        timestamp: datetime,
        direction: str,
        price: float,
        funding_rate: float,
        rsi: float,
        adx: float,
        executed: bool
    ):
        """Log a signal (whether executed or not)"""
        self.signals.append({
            "timestamp": timestamp,
            "direction": direction,
            "price": price,
            "funding_rate": funding_rate,
            "rsi": rsi,
            "adx": adx,
            "executed": executed
        })

    def get_trades_df(self):
        """Return trades as DataFrame"""
        import pandas as pd
        return pd.DataFrame(self.trades)

    def get_equity_df(self):
        """Return equity curve as DataFrame"""
        import pandas as pd
        return pd.DataFrame(self.equity_curve)

    def reset(self):
        """Reset all logged data"""
        self.trades = []
        self.signals = []
        self.equity_curve = []


if __name__ == "__main__":
    # Test the logger
    print("Testing logger module...")

    logger = TradeLogger("data/test_trades.csv")

    logger.log_signal(
        timestamp=datetime.now(),
        direction="short",
        price=42500,
        funding_rate=0.0008,
        rsi=72,
        adx=18,
        reason="Funding extreme positive, RSI overbought"
    )

    logger.log_entry(
        timestamp=datetime.now(),
        direction="short",
        price=42500,
        size=0.0133,
        funding_rate=0.0008,
        capital=10000,
        reason="All conditions met"
    )

    logger.log_exit(
        timestamp=datetime.now(),
        direction="short",
        price=41650,
        size=0.0133,
        pnl=11.32,
        pnl_pct=0.02,
        capital=10011.32,
        reason="Take profit hit"
    )

    logger.log_stats({
        "total_trades": 1,
        "winning_trades": 1,
        "losing_trades": 0,
        "win_rate": 100,
        "total_pnl": 11.32,
        "return_pct": 0.11,
        "profit_factor": float("inf"),
        "current_capital": 10011.32
    })

    print(f"\nLog file created: {logger.log_file}")

    # Cleanup test file
    import os
    os.remove("data/test_trades.csv")
    print("Test file cleaned up")
