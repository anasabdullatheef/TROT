#!/usr/bin/env python3
"""
Atlas Bot - Main Entry Point
Advanced ML-driven trading bot with regime detection and sentiment analysis
"""

import argparse
import time
import schedule
from datetime import datetime
import os

import config
from exchange import ExchangeClient, open_long_position, open_short_position, close_position
from funding import get_current_funding_rate
from features import compute_all_features
from regime import RegimeDetector
from models.lstm_model import RegimeLSTMEnsemble
from strategy import TradingStrategy
from risk import RiskManager
from logger import TradeLogger
from database import Database
from telegram_alerts import alerts
from retrain import run_retraining


class AtlasBot:
    """Main bot class"""

    def __init__(self, live_mode: bool = False, testnet: bool = True):
        self.live_mode = live_mode
        self.exchange = ExchangeClient(testnet=testnet) if live_mode else None
        self.risk_manager = RiskManager()
        self.logger = TradeLogger()
        self.db = Database()

        # Load models
        self.regime_detector = None
        self.lstm_ensemble = None
        self._load_models()

        # Initialize strategy
        self.strategy = TradingStrategy(self.regime_detector, self.lstm_ensemble)

        # State
        self.current_position = None
        self.finbert_history = []

        print("=" * 60)
        print("🤖 ATLAS BOT")
        print("=" * 60)
        mode = "LIVE" if live_mode else "PAPER"
        net = "(Testnet)" if testnet and live_mode else ""
        print(f"Mode: {mode} {net}")
        print(f"Symbol: {config.SYMBOL}")
        print(f"Models loaded: HMM={self.regime_detector is not None}, LSTM={len(self.lstm_ensemble.models) if self.lstm_ensemble else 0}")
        print("=" * 60)

        alerts.send_startup(mode, config.INITIAL_CAPITAL)

    def _load_models(self):
        """Load trained models"""
        if os.path.exists(config.HMM_MODEL_PATH):
            try:
                self.regime_detector = RegimeDetector()
                self.regime_detector.load()
            except Exception as e:
                print(f"⚠️ Could not load HMM: {e}")

        self.lstm_ensemble = RegimeLSTMEnsemble()
        self.lstm_ensemble.load_all()

    def _fetch_current_data(self):
        """Fetch current market data"""
        import pandas as pd
        import requests
        from ta.volatility import AverageTrueRange
        from ta.momentum import RSIIndicator

        # Fetch recent candles
        url = f"{config.BINANCE_BASE_URL}/fapi/v1/klines"
        params = {
            "symbol": config.BINANCE_SYMBOL,
            "interval": "1h",
            "limit": 100
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # Compute basic features
        df = compute_all_features(df, include_sentiment=False)

        # Add funding rate
        funding_rate = get_current_funding_rate()
        df["funding_rate"] = funding_rate

        return df

    def check_and_trade(self):
        """Main trading loop"""
        print(f"\n[{datetime.now()}] Checking market conditions...")

        try:
            df = self._fetch_current_data()
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            alerts.send_error("Data Fetch", str(e))
            return

        if df.empty:
            return

        latest = df.iloc[-1]
        price = latest["close"]
        atr = latest.get("atr", price * 0.02) or price * 0.02
        funding_rate = latest.get("funding_rate", 0) or 0

        print(f"  Price: ${price:,.2f}")
        print(f"  ATR: ${atr:,.0f}")
        print(f"  Funding: {funding_rate*100:.4f}%")

        # Check exit if in position
        if self.current_position:
            exit_check = self.strategy.check_exit_conditions(
                entry_price=self.current_position["entry_price"],
                current_price=price,
                position_type=self.current_position["type"],
                atr=atr,
                funding_rate=funding_rate,
                finbert_history=self.finbert_history
            )

            if exit_check["exit"]:
                self._close_position(price, exit_check["reason"])
                return

            # Show position status
            print(f"  Position: {self.current_position['type'].upper()} @ ${self.current_position['entry_price']:,.2f}")
            print(f"  Unrealized: {exit_check['pnl_pct']*100:+.2f}%")
            return

        # Check entry
        can_open = self.risk_manager.can_open_position()

        if not can_open["allowed"]:
            print(f"  ⛔ Cannot trade: {can_open['reasons']}")
            return

        # Add regime labels
        if self.regime_detector:
            from regime import add_regime_to_dataframe
            df = add_regime_to_dataframe(df, self.regime_detector)

        signal = self.strategy.get_signal(df)

        if signal["halt"]:
            print(f"  ⛔ HALT: {signal['halt_reason']}")
            self.logger.log_halt(datetime.now(), signal["halt_reason"])
            alerts.send_sentiment_alert("Halt", signal["halt_reason"])
            return

        print(f"  Regime: {signal['regime_name']}")
        print(f"  Confidence: {signal['confidence']*100:.1f}%")

        if signal["signal"] == "enter" and signal["direction"]:
            self._open_position(
                direction=signal["direction"],
                price=price,
                atr=atr,
                confidence=signal["confidence"],
                regime=signal["regime"],
                funding_boost=signal["funding_boost"],
                sentiment=signal["sentiment_score"]
            )
        else:
            print("  No signal - waiting")

    def _open_position(
        self,
        direction: str,
        price: float,
        atr: float,
        confidence: float,
        regime: int,
        funding_boost: bool,
        sentiment: float
    ):
        """Open a new position"""
        sizing = self.risk_manager.calculate_position_size(price, atr, confidence)

        if self.live_mode and self.exchange:
            if direction == "long":
                result = open_long_position(self.exchange, sizing["position_size"], price, atr)
            else:
                result = open_short_position(self.exchange, sizing["position_size"], price, atr)

            if not result:
                alerts.send_error("Order Failed", f"Could not open {direction} position")
                return

        self.current_position = self.risk_manager.open_position(
            position_type=direction,
            entry_price=price,
            size=sizing["position_size"],
            timestamp=datetime.now(),
            atr=atr,
            confidence=confidence,
            regime=regime
        )

        self.logger.log_entry(
            timestamp=datetime.now(),
            direction=direction,
            price=price,
            size=sizing["position_size"],
            regime=regime,
            confidence=confidence,
            funding_boost=funding_boost,
            sentiment=sentiment,
            capital=self.risk_manager.current_capital,
            reason="Signal confirmed"
        )

        alerts.send_trade_entry(
            direction=direction,
            price=price,
            size=sizing["position_size"],
            regime=regime,
            confidence=confidence,
            funding_boost=funding_boost,
            sentiment=sentiment
        )

    def _close_position(self, price: float, reason: str):
        """Close current position"""
        if not self.current_position:
            return

        if self.live_mode and self.exchange:
            close_position(self.exchange)

        trade = self.risk_manager.close_position(
            exit_price=price,
            exit_reason=reason,
            timestamp=datetime.now()
        )

        if trade:
            self.logger.log_exit(
                timestamp=datetime.now(),
                direction=trade["type"],
                price=price,
                size=trade["size"],
                pnl=trade["pnl"],
                pnl_pct=trade["pnl_pct"],
                capital=self.risk_manager.current_capital,
                reason=reason
            )

            alerts.send_trade_exit(
                direction=trade["type"],
                entry_price=trade["entry_price"],
                exit_price=price,
                pnl=trade["pnl"],
                pnl_pct=trade["pnl_pct"],
                exit_reason=reason
            )

            # Check kill switch
            kill_switch, daily_pnl = self.risk_manager.check_daily_drawdown()
            if kill_switch:
                alerts.send_kill_switch_alert(
                    reason="Daily drawdown limit reached",
                    daily_pnl=daily_pnl * self.risk_manager.initial_capital
                )

        self.current_position = None

    def run(self):
        """Start the bot"""
        print(f"\n🚀 Starting bot... checking every {config.CHECK_INTERVAL_MINUTES} minutes")

        # Schedule trading checks
        schedule.every(config.CHECK_INTERVAL_MINUTES).minutes.do(self.check_and_trade)

        # Schedule weekly retraining
        schedule.every(config.RETRAIN_INTERVAL_DAYS).days.do(run_retraining)

        # Run immediately
        self.check_and_trade()

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⛔ Bot stopped by user")
            self._shutdown()

    def _shutdown(self):
        """Clean shutdown"""
        if self.current_position:
            df = self._fetch_current_data()
            if not df.empty:
                self._close_position(df.iloc[-1]["close"], "Bot shutdown")

        stats = self.risk_manager.get_statistics()
        print(f"\nFinal Stats: {stats}")

        self.db.close()


def main():
    parser = argparse.ArgumentParser(description="Atlas Trading Bot")
    parser.add_argument("--live", action="store_true", help="Run in live mode")
    parser.add_argument("--mainnet", action="store_true", help="Use mainnet")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--retrain", action="store_true", help="Retrain models")

    args = parser.parse_args()

    if args.backtest:
        from backtest import main as run_backtest
        run_backtest()
    elif args.retrain:
        run_retraining()
    else:
        bot = AtlasBot(live_mode=args.live, testnet=not args.mainnet)
        bot.run()


if __name__ == "__main__":
    main()
