"""
Notifications - Telegram alerts for trades and daily summaries
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Optional

from executor import ArbitragePosition
from logger import DatabaseLogger
import config


class TelegramNotifier:
    """Sends notifications via Telegram"""

    def __init__(self):
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            print("Telegram notifications disabled (no token/chat_id)")

    async def send_message(self, message: str) -> bool:
        """Send a message via Telegram"""
        if not self.enabled:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    return response.status == 200
        except Exception as e:
            print(f"Telegram error: {e}")
            return False

    async def notify_trade_executed(self, position: ArbitragePosition):
        """Notify about an executed trade"""
        message = (
            f"<b>ARB TRADE EXECUTED</b>\n\n"
            f"Position ID: <code>{position.position_id}</code>\n"
            f"Pair: {position.opportunity.buy_exchange} -> {position.opportunity.sell_exchange}\n"
            f"Buy: ${position.opportunity.buy_price:,.2f}\n"
            f"Sell: ${position.opportunity.sell_price:,.2f}\n"
            f"Net Spread: {position.opportunity.net_spread_pct*100:.4f}%\n"
            f"Est. Profit: ${position.opportunity.estimated_profit_usd:.2f}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        await self.send_message(message)

    async def notify_trade_closed(self, position: ArbitragePosition):
        """Notify about a closed trade"""
        pnl_emoji = "+" if position.realized_pnl >= 0 else ""

        message = (
            f"<b>ARB TRADE CLOSED</b>\n\n"
            f"Position ID: <code>{position.position_id}</code>\n"
            f"Status: {position.status.value}\n"
            f"PnL: {pnl_emoji}${position.realized_pnl:.2f}\n"
            f"Notes: {position.notes}\n"
            f"Duration: {(position.exit_time - position.entry_time).total_seconds():.1f}s"
        )

        await self.send_message(message)

    async def notify_risk_alert(self, alert_type: str, message: str):
        """Notify about a risk event"""
        msg = (
            f"<b>RISK ALERT</b>\n\n"
            f"Type: {alert_type}\n"
            f"Message: {message}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        await self.send_message(msg)

    async def send_daily_summary(self, logger: DatabaseLogger):
        """Send daily summary"""
        stats = logger.get_today_stats()

        message = (
            f"<b>DAILY SUMMARY - {stats['date']}</b>\n\n"
            f"Opportunities Seen: {stats['opportunities_seen']}\n"
            f"Tradeable: {stats['tradeable_opportunities']}\n"
            f"Trades Executed: {stats['trades_executed']}\n"
            f"Win Rate: {stats['win_rate']:.1f}%\n"
            f"Total PnL: ${stats['total_pnl']:.2f}\n"
            f"Avg Spread: {stats['avg_spread_pct']:.4f}%"
        )

        await self.send_message(message)

    async def notify_startup(self):
        """Notify bot startup"""
        mode = "OBSERVATION" if not config.LIVE_TRADING else "LIVE TRADING"

        message = (
            f"<b>ARB BOT STARTED</b>\n\n"
            f"Mode: {mode}\n"
            f"Trade Size: ${config.TRADE_SIZE_USD}\n"
            f"Min Spread: {config.MIN_SPREAD_TO_TRADE*100:.2f}%\n"
            f"Max Daily Loss: ${config.MAX_DAILY_LOSS_USD}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        await self.send_message(message)

    async def notify_shutdown(self, reason: str = "Manual"):
        """Notify bot shutdown"""
        message = (
            f"<b>ARB BOT STOPPED</b>\n\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        await self.send_message(message)
