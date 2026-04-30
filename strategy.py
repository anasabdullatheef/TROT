"""
Trading strategy module.
Implements momentum-based signal logic with EMA, RSI, and volume confirmation.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import ta
import config
from logger import logger


class MomentumStrategy:
    """
    Momentum-based trading strategy.

    Buy Signal:
        - Price breaks above 20-period EMA
        - RSI above 50
        - Volume above average

    Sell Signal:
        - Price drops below EMA
        - OR stop-loss hit
        - OR take-profit hit
    """

    def __init__(self):
        self.ema_period = config.EMA_PERIOD
        self.rsi_period = config.RSI_PERIOD
        self.rsi_threshold = config.RSI_THRESHOLD
        self.volume_ma_period = config.VOLUME_MA_PERIOD
        self.volume_multiplier = config.VOLUME_MULTIPLIER

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()

        # EMA
        df['ema'] = ta.trend.ema_indicator(df['close'], window=self.ema_period)

        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)

        # Volume MA
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Price position relative to EMA
        df['above_ema'] = df['close'] > df['ema']
        prev_above = df['above_ema'].shift(1)
        df['prev_above_ema'] = prev_above.where(prev_above.notna(), False)

        # EMA crossover detection (handle NaN safely)
        df['ema_crossover_up'] = (df['above_ema']) & (~df['prev_above_ema'].astype(bool))
        df['ema_crossover_down'] = (~df['above_ema'].astype(bool)) & (df['prev_above_ema'])

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with 'signal' column (1 = buy, -1 = sell, 0 = hold)
        """
        df = self.calculate_indicators(df)

        # Buy conditions
        buy_condition = (
            (df['above_ema']) &                                    # Price above EMA
            (df['rsi'] > self.rsi_threshold) &                    # RSI above threshold
            (df['volume_ratio'] >= self.volume_multiplier) &      # Volume confirmation
            (df['ema_crossover_up'] | df['above_ema'].shift(1).isna())  # Crossover or first signal
        )

        # Sell conditions (price drops below EMA)
        sell_condition = df['ema_crossover_down']

        # Create signal column
        df['signal'] = 0
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1

        return df

    def get_current_signal(self, df: pd.DataFrame) -> Tuple[int, dict]:
        """
        Get the current trading signal.

        Args:
            df: DataFrame with recent OHLCV data (at least EMA period + RSI period candles)

        Returns:
            Tuple of (signal, indicators_dict)
            signal: 1 = buy, -1 = sell, 0 = hold
        """
        if len(df) < max(self.ema_period, self.rsi_period, self.volume_ma_period) + 1:
            return 0, {}

        df = self.generate_signals(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        indicators = {
            'close': latest['close'],
            'ema': latest['ema'],
            'rsi': latest['rsi'],
            'volume_ratio': latest['volume_ratio']
        }

        # Check for buy signal
        if (latest['above_ema'] and
            latest['rsi'] > self.rsi_threshold and
            latest['volume_ratio'] >= self.volume_multiplier and
            (not prev['above_ema'] or pd.isna(prev['above_ema']))):
            return 1, indicators

        # Check for sell signal (price dropped below EMA)
        if not latest['above_ema'] and prev['above_ema']:
            return -1, indicators

        return 0, indicators

    def should_exit(self, df: pd.DataFrame, entry_price: float) -> Tuple[bool, str]:
        """
        Check if we should exit a position.

        Args:
            df: DataFrame with recent OHLCV data
            entry_price: The price at which position was entered

        Returns:
            Tuple of (should_exit, reason)
        """
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        current_price = latest['close']

        # Check if price dropped below EMA
        if not latest['above_ema']:
            return True, "price_below_ema"

        # Check stop-loss
        stop_loss_price = entry_price * (1 - config.STOP_LOSS_PCT)
        if current_price <= stop_loss_price:
            return True, "stop_loss"

        # Check take-profit
        take_profit_price = entry_price * (1 + config.TAKE_PROFIT_PCT)
        if current_price >= take_profit_price:
            return True, "take_profit"

        return False, ""

    def get_stop_loss_price(self, entry_price: float) -> float:
        """Calculate stop-loss price for a given entry."""
        return entry_price * (1 - config.STOP_LOSS_PCT)

    def get_take_profit_price(self, entry_price: float) -> float:
        """Calculate take-profit price for a given entry."""
        return entry_price * (1 + config.TAKE_PROFIT_PCT)


# Global strategy instance
strategy = MomentumStrategy()
