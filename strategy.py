"""
Strategy Module
Combines regime detection, LSTM predictions, sentiment, and funding signals
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

import config
from regime import RegimeDetector
from models.lstm_model import RegimeLSTMEnsemble
from funding import classify_funding_signal


class TradingStrategy:
    """
    Atlas Trading Strategy

    Signal Logic:
    1. Detect current regime with HMM
    2. If sentiment spike > 2 std: halt all new trades
    3. If 3 consecutive FinBERT headlines highly negative: close all longs
    4. If funding rate signal triggers (Config B): boost confidence by 20%
    5. Load regime-specific LSTM, only trade if confidence > 65%
    """

    def __init__(
        self,
        regime_detector: RegimeDetector = None,
        lstm_ensemble: RegimeLSTMEnsemble = None
    ):
        self.regime_detector = regime_detector
        self.lstm_ensemble = lstm_ensemble

        # Sentiment tracking
        self.sentiment_history: List[float] = []
        self.consecutive_negative_headlines = 0

        # State
        self.halted = False
        self.halt_reason = ""

    def check_sentiment_halt(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check if trading should be halted due to sentiment spike

        Returns:
            (should_halt, reason)
        """
        if df.empty or "finbert_1h" not in df.columns:
            return False, ""

        recent_sentiment = df["finbert_1h"].tail(24).dropna()

        if len(recent_sentiment) < 10:
            return False, ""

        mean_sentiment = recent_sentiment.mean()
        std_sentiment = recent_sentiment.std()

        if std_sentiment == 0:
            return False, ""

        current_sentiment = df["finbert_1h"].iloc[-1]
        z_score = abs(current_sentiment - mean_sentiment) / std_sentiment

        if z_score > config.SENTIMENT_SPIKE_THRESHOLD:
            return True, f"Sentiment spike: z-score {z_score:.2f} > {config.SENTIMENT_SPIKE_THRESHOLD}"

        return False, ""

    def check_consecutive_negative(self, headlines_sentiment: List[float]) -> bool:
        """
        Check if last 3 headlines are highly negative

        Returns:
            True if should close longs
        """
        if len(headlines_sentiment) < config.CONSECUTIVE_NEGATIVE_THRESHOLD:
            return False

        recent = headlines_sentiment[-config.CONSECUTIVE_NEGATIVE_THRESHOLD:]

        # Check if all are negative (< -0.3)
        return all(s < -0.3 for s in recent)

    def get_funding_boost(self, funding_rate: float, direction: str) -> float:
        """
        Check if funding signal aligns with trade direction

        Returns:
            Confidence boost (0 or FUNDING_BOOST_WEIGHT)
        """
        funding_signal = classify_funding_signal(funding_rate)

        if funding_signal is None:
            return 0.0

        # Funding signal should align with our direction
        if funding_signal == direction:
            return config.FUNDING_BOOST_WEIGHT

        return 0.0

    def get_signal(
        self,
        df: pd.DataFrame,
        current_position: Optional[str] = None
    ) -> Dict:
        """
        Generate trading signal

        Args:
            df: DataFrame with all features and regime labels
            current_position: 'long', 'short', or None

        Returns:
            Dict with signal info
        """
        result = {
            "signal": None,
            "direction": None,
            "confidence": 0.0,
            "regime": None,
            "regime_name": None,
            "funding_boost": False,
            "sentiment_score": 0.0,
            "halt": False,
            "halt_reason": "",
            "close_longs": False,
            "reasons": []
        }

        if df.empty or len(df) < config.LSTM_SEQUENCE_LENGTH:
            result["reasons"].append("Insufficient data")
            return result

        # Get current regime
        if self.regime_detector is not None:
            regime_id, regime_name, regime_stats = self.regime_detector.get_current_regime(df)
            result["regime"] = regime_id
            result["regime_name"] = regime_name
            result["reasons"].append(f"Regime: {regime_name}")
        else:
            regime_id = 0
            result["regime"] = 0
            result["regime_name"] = "Unknown"

        # Check sentiment halt
        should_halt, halt_reason = self.check_sentiment_halt(df)
        if should_halt:
            result["halt"] = True
            result["halt_reason"] = halt_reason
            result["reasons"].append(f"HALT: {halt_reason}")
            return result

        # Get current values
        latest = df.iloc[-1]
        funding_rate = latest.get("funding_rate", 0) or 0
        finbert = latest.get("finbert_1h", 0) or 0
        fng = latest.get("fng_value", 0.5) or 0.5

        result["sentiment_score"] = finbert

        # Get LSTM prediction
        if self.lstm_ensemble is not None and regime_id in self.lstm_ensemble.models:
            confidence = self.lstm_ensemble.predict(df, regime_id)
        else:
            # Without LSTM, use simple logic based on regime stats
            confidence = 0.5

        result["confidence"] = confidence

        # Determine direction based on confidence
        if confidence > 0.5:
            potential_direction = "long"
        else:
            potential_direction = "short"
            confidence = 1 - confidence  # Invert for short confidence

        # Check funding boost
        funding_boost = self.get_funding_boost(funding_rate, potential_direction)
        if funding_boost > 0:
            confidence += funding_boost
            result["funding_boost"] = True
            result["reasons"].append(f"Funding boost: +{funding_boost*100:.0f}%")

        result["confidence"] = min(confidence, 1.0)

        # Check if confidence meets threshold
        if result["confidence"] >= config.LSTM_CONFIDENCE_THRESHOLD:
            # Don't enter if we already have a position
            if current_position is None:
                result["signal"] = "enter"
                result["direction"] = potential_direction
                result["reasons"].append(
                    f"Confidence {result['confidence']*100:.1f}% >= {config.LSTM_CONFIDENCE_THRESHOLD*100:.0f}%"
                )
            else:
                result["reasons"].append(f"Already in {current_position} position")
        else:
            result["reasons"].append(
                f"Confidence {result['confidence']*100:.1f}% < {config.LSTM_CONFIDENCE_THRESHOLD*100:.0f}%"
            )

        return result

    def check_exit_conditions(
        self,
        entry_price: float,
        current_price: float,
        position_type: str,
        atr: float,
        funding_rate: float = 0,
        finbert_history: List[float] = None
    ) -> Dict:
        """
        Check if position should be closed

        Exit conditions:
        - Dynamic TP at 3x ATR
        - Dynamic SL at 1.5x ATR
        - 3 consecutive highly negative headlines (for longs)
        """
        result = {
            "exit": False,
            "reason": "",
            "pnl_pct": 0.0
        }

        # Calculate PnL
        if position_type == "long":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        result["pnl_pct"] = pnl_pct

        # Dynamic TP/SL based on ATR
        tp_distance = atr * config.ATR_TAKE_PROFIT_MULTIPLIER
        sl_distance = atr * config.ATR_STOP_LOSS_MULTIPLIER

        tp_pct = tp_distance / entry_price
        sl_pct = sl_distance / entry_price

        # Check take profit
        if pnl_pct >= tp_pct:
            result["exit"] = True
            result["reason"] = f"Take profit: {pnl_pct*100:.2f}% >= {tp_pct*100:.2f}% (3x ATR)"
            return result

        # Check stop loss
        if pnl_pct <= -sl_pct:
            result["exit"] = True
            result["reason"] = f"Stop loss: {pnl_pct*100:.2f}% <= -{sl_pct*100:.2f}% (1.5x ATR)"
            return result

        # Check consecutive negative headlines for longs
        if position_type == "long" and finbert_history:
            if self.check_consecutive_negative(finbert_history):
                result["exit"] = True
                result["reason"] = "3 consecutive highly negative headlines"
                return result

        return result


class BacktestStrategy:
    """
    Backtesting wrapper for the trading strategy
    """

    def __init__(
        self,
        regime_detector: RegimeDetector = None,
        lstm_ensemble: RegimeLSTMEnsemble = None
    ):
        self.strategy = TradingStrategy(regime_detector, lstm_ensemble)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for entire dataframe (for backtesting)
        """
        df = df.copy()

        # Initialize signal columns
        df["signal"] = None
        df["signal_direction"] = None
        df["signal_confidence"] = 0.0
        df["signal_halt"] = False

        # Process each row (after LSTM sequence length)
        for i in range(config.LSTM_SEQUENCE_LENGTH, len(df)):
            # Get data up to current row
            current_df = df.iloc[:i + 1].copy()

            # Get signal
            signal = self.strategy.get_signal(current_df)

            df.loc[df.index[i], "signal"] = signal["signal"]
            df.loc[df.index[i], "signal_direction"] = signal["direction"]
            df.loc[df.index[i], "signal_confidence"] = signal["confidence"]
            df.loc[df.index[i], "signal_halt"] = signal["halt"]

        return df


if __name__ == "__main__":
    print("Testing strategy module...")

    # Create sample data
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=n, freq="1h"),
        "close": 40000 + np.cumsum(np.random.randn(n) * 100),
        "atr": 500 + np.random.rand(n) * 200,
        "funding_rate": np.random.randn(n) * 0.0002,
        "finbert_1h": np.random.randn(n) * 0.2,
        "fng_value": 0.3 + np.random.rand(n) * 0.4,
        "regime": np.random.randint(0, 4, n)
    })

    # Test strategy without models
    strategy = TradingStrategy()

    # Test signal generation
    signal = strategy.get_signal(df)
    print(f"\nSignal result:")
    for key, value in signal.items():
        print(f"  {key}: {value}")

    # Test exit conditions
    exit_result = strategy.check_exit_conditions(
        entry_price=40000,
        current_price=41500,
        position_type="long",
        atr=500,
        funding_rate=0.0003
    )
    print(f"\nExit check: {exit_result}")

    # Test sentiment halt
    df_spike = df.copy()
    df_spike.loc[df_spike.index[-1], "finbert_1h"] = 2.0  # Spike
    should_halt, reason = strategy.check_sentiment_halt(df_spike)
    print(f"\nSentiment halt test: {should_halt}, {reason}")

    print("\n✅ Strategy module test passed")
