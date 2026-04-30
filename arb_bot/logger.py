"""
Logger - Logs every opportunity, trade, and PnL to SQLite
"""

import os
import sqlite3
from datetime import datetime, date
from typing import Dict, List, Optional
from pathlib import Path

from arbitrage import ArbitrageOpportunity, OpportunityType
from executor import ArbitragePosition, PositionStatus
import config


class DatabaseLogger:
    """Logs all trading activity to SQLite database"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DATABASE_PATH
        self._ensure_db_dir()
        self._init_db()

    def _ensure_db_dir(self):
        """Ensure database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Opportunities table - logs every opportunity seen
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                buy_exchange TEXT,
                sell_exchange TEXT,
                buy_price REAL,
                sell_price REAL,
                gross_spread_pct REAL,
                total_fee_pct REAL,
                net_spread_pct REAL,
                opportunity_type TEXT,
                estimated_profit_usd REAL,
                was_traded INTEGER DEFAULT 0
            )
        """)

        # Trades table - logs executed trades
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT UNIQUE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                buy_exchange TEXT,
                sell_exchange TEXT,
                buy_price REAL,
                sell_price REAL,
                amount_btc REAL,
                amount_usd REAL,
                entry_spread_pct REAL,
                status TEXT,
                exit_time DATETIME,
                realized_pnl REAL,
                notes TEXT
            )
        """)

        # Daily summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE,
                opportunities_seen INTEGER DEFAULT 0,
                tradeable_opportunities INTEGER DEFAULT 0,
                trades_executed INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_spread_pct REAL DEFAULT 0
            )
        """)

        # Price snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                binance_bid REAL,
                binance_ask REAL,
                bybit_bid REAL,
                bybit_ask REAL,
                okx_bid REAL,
                okx_ask REAL
            )
        """)

        conn.commit()
        conn.close()

    def log_opportunity(self, opportunity: ArbitrageOpportunity, was_traded: bool = False):
        """Log an arbitrage opportunity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO opportunities (
                timestamp, buy_exchange, sell_exchange, buy_price, sell_price,
                gross_spread_pct, total_fee_pct, net_spread_pct,
                opportunity_type, estimated_profit_usd, was_traded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            opportunity.timestamp.isoformat(),
            opportunity.buy_exchange,
            opportunity.sell_exchange,
            opportunity.buy_price,
            opportunity.sell_price,
            opportunity.gross_spread_pct,
            opportunity.total_fee_pct,
            opportunity.net_spread_pct,
            opportunity.opportunity_type.value,
            opportunity.estimated_profit_usd,
            1 if was_traded else 0
        ))

        conn.commit()
        conn.close()

    def log_trade(self, position: ArbitragePosition):
        """Log an executed trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        amount_btc = config.TRADE_SIZE_USD / position.opportunity.buy_price

        cursor.execute("""
            INSERT OR REPLACE INTO trades (
                position_id, timestamp, buy_exchange, sell_exchange,
                buy_price, sell_price, amount_btc, amount_usd,
                entry_spread_pct, status, exit_time, realized_pnl, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position.position_id,
            position.entry_time.isoformat(),
            position.opportunity.buy_exchange,
            position.opportunity.sell_exchange,
            position.opportunity.buy_price,
            position.opportunity.sell_price,
            amount_btc,
            config.TRADE_SIZE_USD,
            position.opportunity.net_spread_pct,
            position.status.value,
            position.exit_time.isoformat() if position.exit_time else None,
            position.realized_pnl,
            position.notes
        ))

        conn.commit()
        conn.close()

    def log_prices(self, prices: Dict):
        """Log current prices from all exchanges"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO price_snapshots (
                timestamp, binance_bid, binance_ask,
                bybit_bid, bybit_ask, okx_bid, okx_ask
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            prices.get("binance", {}).get("bid", 0),
            prices.get("binance", {}).get("ask", 0),
            prices.get("bybit", {}).get("bid", 0),
            prices.get("bybit", {}).get("ask", 0),
            prices.get("okx", {}).get("bid", 0),
            prices.get("okx", {}).get("ask", 0)
        ))

        conn.commit()
        conn.close()

    def get_today_stats(self) -> Dict:
        """Get today's statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = date.today().isoformat()

        # Opportunities seen today
        cursor.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN opportunity_type = 'tradeable' THEN 1 ELSE 0 END),
                   AVG(net_spread_pct)
            FROM opportunities
            WHERE DATE(timestamp) = ?
        """, (today,))
        opp_result = cursor.fetchone()

        # Trades today
        cursor.execute("""
            SELECT COUNT(*), SUM(realized_pnl),
                   SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)
            FROM trades
            WHERE DATE(timestamp) = ?
        """, (today,))
        trade_result = cursor.fetchone()

        conn.close()

        opportunities_seen = opp_result[0] or 0
        tradeable_opportunities = opp_result[1] or 0
        avg_spread = opp_result[2] or 0

        trades_executed = trade_result[0] or 0
        total_pnl = trade_result[1] or 0
        winning_trades = trade_result[2] or 0
        win_rate = (winning_trades / trades_executed * 100) if trades_executed > 0 else 0

        return {
            "date": today,
            "opportunities_seen": opportunities_seen,
            "tradeable_opportunities": tradeable_opportunities,
            "trades_executed": trades_executed,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_spread_pct": avg_spread * 100
        }

    def get_recent_trades(self, limit: int = 5) -> List[Dict]:
        """Get recent trades"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT position_id, timestamp, buy_exchange, sell_exchange,
                   entry_spread_pct, status, realized_pnl
            FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        results = cursor.fetchall()
        conn.close()

        trades = []
        for row in results:
            trades.append({
                "position_id": row[0],
                "timestamp": row[1],
                "pair": f"{row[2]}->{row[3]}",
                "spread_pct": row[4] * 100 if row[4] else 0,
                "status": row[5],
                "pnl": row[6] or 0
            })

        return trades

    def get_recent_opportunities(self, limit: int = 10) -> List[Dict]:
        """Get recent opportunities"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, buy_exchange, sell_exchange,
                   net_spread_pct, opportunity_type, was_traded
            FROM opportunities
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        results = cursor.fetchall()
        conn.close()

        opportunities = []
        for row in results:
            opportunities.append({
                "timestamp": row[0],
                "pair": f"{row[1]}->{row[2]}",
                "net_spread_pct": row[3] * 100 if row[3] else 0,
                "type": row[4],
                "traded": row[5] == 1
            })

        return opportunities

    def get_opportunity_count_last_hour(self) -> int:
        """Get number of opportunities in the last hour"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM opportunities
            WHERE timestamp > datetime('now', '-1 hour')
        """)

        result = cursor.fetchone()
        conn.close()

        return result[0] or 0
