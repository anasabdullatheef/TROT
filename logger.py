"""
Logger Module
Handles logging to CSV files and console output
"""

import csv
import os
from datetime import datetime
from typing import Dict, List
import config


class TradeLogger:
    """Logs trades and signals to CSV and console"""

    def __init__(self, log_file: str = None):
        self.log_file = log_file or config.LOG_FILE
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        if not os.path.exists(self.log_file):
            self._init_csv()

    def _init_csv(self):
        headers = [
            "timestamp", "action", "direction", "price", "size",
            "regime", "confidence", "funding_boost", "sentiment",
            "pnl", "pnl_pct", "capital", "reason"
        ]
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_entry(
        self,
        timestamp: datetime,
        direction: str,
        price: float,
        size: float,
        regime: int,
        confidence: float,
        funding_boost: bool,
        sentiment: float,
        capital: float,
        reason: str
    ):
        self._write_row({
            "timestamp": timestamp,
            "action": "entry",
            "direction": direction,
            "price": price,
            "size": size,
            "regime": regime,
            "confidence": confidence,
            "funding_boost": funding_boost,
            "sentiment": sentiment,
            "capital": capital,
            "reason": reason
        })
        boost = " [+FUNDING]" if funding_boost else ""
        print(f"🔵 {timestamp}: ENTRY {direction.upper()} @ ${price:,.0f} | "
              f"Regime {regime} | Conf: {confidence*100:.1f}%{boost}")

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
        print(f"{emoji} {timestamp}: EXIT @ ${price:,.0f} | "
              f"PnL: ${pnl:,.2f} ({pnl_pct*100:+.2f}%) | {reason}")

    def log_halt(self, timestamp: datetime, reason: str):
        self._write_row({
            "timestamp": timestamp,
            "action": "halt",
            "reason": reason
        })
        print(f"⛔ {timestamp}: HALT - {reason}")

    def _write_row(self, data: Dict):
        row = [
            data.get("timestamp", ""),
            data.get("action", ""),
            data.get("direction", ""),
            data.get("price", ""),
            data.get("size", ""),
            data.get("regime", ""),
            data.get("confidence", ""),
            data.get("funding_boost", ""),
            data.get("sentiment", ""),
            data.get("pnl", ""),
            data.get("pnl_pct", ""),
            data.get("capital", ""),
            data.get("reason", "")
        ]
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


class BacktestLogger:
    """Collects backtest data for analysis"""

    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

    def log_trade(self, trade: Dict):
        self.trades.append(trade)

    def log_equity(self, timestamp: datetime, capital: float, drawdown: float = 0):
        self.equity_curve.append({
            "timestamp": timestamp,
            "capital": capital,
            "drawdown": drawdown
        })

    def reset(self):
        self.trades = []
        self.equity_curve = []
