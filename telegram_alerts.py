"""
Telegram Alerts Module
Sends trade notifications and reports via Telegram
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional
import config

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("Warning: python-telegram-bot not installed, Telegram alerts disabled")


class TelegramAlerts:
    """Handles Telegram notifications"""

    def __init__(self):
        self.enabled = False
        self.bot = None

        if TELEGRAM_AVAILABLE and config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
            try:
                self.bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
                self.chat_id = config.TELEGRAM_CHAT_ID
                self.enabled = True
                print("✅ Telegram alerts enabled")
            except Exception as e:
                print(f"⚠️ Telegram setup failed: {e}")
        else:
            print("⚠️ Telegram alerts disabled (missing credentials)")

    def _send_message(self, message: str):
        """Send a message synchronously"""
        if not self.enabled:
            print(f"[TELEGRAM DISABLED] {message}")
            return False

        try:
            asyncio.get_event_loop().run_until_complete(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode="HTML"
                )
            )
            return True
        except Exception as e:
            print(f"Telegram send error: {e}")
            return False

    def send_trade_entry(
        self,
        direction: str,
        price: float,
        size: float,
        regime: int,
        confidence: float,
        funding_boost: bool,
        sentiment: float
    ):
        """Send trade entry notification"""
        emoji = "🟢" if direction == "long" else "🔴"
        boost = "📈 +20% funding boost" if funding_boost else ""

        message = f"""
{emoji} <b>TRADE ENTRY</b>

<b>Direction:</b> {direction.upper()}
<b>Price:</b> ${price:,.2f}
<b>Size:</b> {size:.6f} BTC
<b>Regime:</b> {regime}
<b>Confidence:</b> {confidence*100:.1f}%
<b>Sentiment:</b> {sentiment:.2f}
{boost}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message)

    def send_trade_exit(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str
    ):
        """Send trade exit notification"""
        emoji = "💰" if pnl > 0 else "💸"

        message = f"""
{emoji} <b>TRADE EXIT</b>

<b>Direction:</b> {direction.upper()}
<b>Entry:</b> ${entry_price:,.2f}
<b>Exit:</b> ${exit_price:,.2f}
<b>PnL:</b> ${pnl:,.2f} ({pnl_pct*100:+.2f}%)
<b>Reason:</b> {exit_reason}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message)

    def send_sentiment_alert(self, alert_type: str, details: str):
        """Send sentiment-based alert"""
        message = f"""
⚠️ <b>SENTIMENT ALERT</b>

<b>Type:</b> {alert_type}
<b>Details:</b> {details}
<b>Action:</b> Trading halted

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message)

    def send_kill_switch_alert(self, reason: str, daily_pnl: float):
        """Send kill switch activation alert"""
        message = f"""
🛑 <b>KILL SWITCH ACTIVATED</b>

<b>Reason:</b> {reason}
<b>Daily PnL:</b> ${daily_pnl:,.2f}
<b>Action:</b> All trading halted for today

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message)

    def send_retrain_report(
        self,
        model_type: str,
        old_accuracy: float,
        new_accuracy: float,
        replaced: bool,
        details: str
    ):
        """Send model retraining report"""
        emoji = "✅" if replaced else "❌"
        action = "Model replaced" if replaced else "Old model kept"

        message = f"""
🔄 <b>MODEL RETRAIN REPORT</b>

<b>Model:</b> {model_type}
<b>Old Accuracy:</b> {old_accuracy*100:.2f}%
<b>New Accuracy:</b> {new_accuracy*100:.2f}%
{emoji} <b>{action}</b>

<b>Details:</b>
{details}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message)

    def send_daily_summary(
        self,
        trades: int,
        pnl: float,
        win_rate: float,
        capital: float,
        regime_distribution: Dict
    ):
        """Send daily trading summary"""
        regime_text = "\n".join([f"  Regime {k}: {v} trades" for k, v in regime_distribution.items()])

        message = f"""
📊 <b>DAILY SUMMARY</b>

<b>Trades:</b> {trades}
<b>PnL:</b> ${pnl:,.2f}
<b>Win Rate:</b> {win_rate:.1f}%
<b>Capital:</b> ${capital:,.2f}

<b>Regime Distribution:</b>
{regime_text}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message)

    def send_error(self, error_type: str, error_message: str):
        """Send error notification"""
        message = f"""
❌ <b>ERROR</b>

<b>Type:</b> {error_type}
<b>Message:</b> {error_message}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message)

    def send_startup(self, mode: str, capital: float):
        """Send bot startup notification"""
        message = f"""
🚀 <b>ATLAS BOT STARTED</b>

<b>Mode:</b> {mode}
<b>Capital:</b> ${capital:,.2f}
<b>Symbol:</b> {config.SYMBOL}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message)


# Global instance
alerts = TelegramAlerts()


if __name__ == "__main__":
    # Test alerts (will print to console if no credentials)
    alerts.send_startup("BACKTEST", 10000)
    alerts.send_trade_entry(
        direction="long",
        price=50000,
        size=0.01,
        regime=2,
        confidence=0.72,
        funding_boost=True,
        sentiment=0.15
    )
    alerts.send_trade_exit(
        direction="long",
        entry_price=50000,
        exit_price=51000,
        pnl=100,
        pnl_pct=0.02,
        exit_reason="Take Profit"
    )
