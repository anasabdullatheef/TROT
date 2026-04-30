"""
Market Regime Module
Uses ADX (Average Directional Index) to determine if market is trending or ranging
We only trade funding rate reversions in ranging markets (ADX < threshold)
"""

import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
from typing import Tuple, Dict
import config


def calculate_adx(df: pd.DataFrame, period: int = None) -> pd.DataFrame:
    """
    Calculate ADX indicator and add to DataFrame

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ADX period (default from config)

    Returns:
        DataFrame with additional columns: adx, adx_pos, adx_neg
    """
    if period is None:
        period = config.ADX_PERIOD

    df = df.copy()

    # Ensure we have required columns
    required = ["high", "low", "close"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"DataFrame must have columns: {required}")

    # Calculate ADX using ta library (fillna=False to preserve NaN during warmup)
    adx_indicator = ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=period,
        fillna=False
    )

    df["adx"] = adx_indicator.adx()
    df["adx_pos"] = adx_indicator.adx_pos()  # +DI
    df["adx_neg"] = adx_indicator.adx_neg()  # -DI

    return df


def is_ranging_market(adx_value: float, threshold: float = None) -> bool:
    """
    Check if market is in a ranging (non-trending) state

    ADX interpretation:
    - ADX < 20: Weak trend / ranging
    - ADX 20-25: Weak trend starting
    - ADX 25-50: Strong trend
    - ADX > 50: Very strong trend

    For funding rate mean reversion, we want ranging markets (ADX < 25)
    """
    if threshold is None:
        threshold = config.ADX_THRESHOLD

    return adx_value < threshold


def classify_regime(adx: float, adx_pos: float, adx_neg: float) -> Dict:
    """
    Classify the current market regime based on ADX values

    Returns dict with:
        - regime: 'ranging', 'weak_trend', 'strong_trend', 'very_strong_trend'
        - direction: 'bullish', 'bearish', 'neutral'
        - tradeable: bool (True if regime allows our strategy)
    """
    # Determine trend strength
    if adx < 20:
        regime = "ranging"
    elif adx < 25:
        regime = "weak_trend"
    elif adx < 50:
        regime = "strong_trend"
    else:
        regime = "very_strong_trend"

    # Determine trend direction using +DI and -DI
    if adx_pos > adx_neg + 5:  # Bullish if +DI significantly higher
        direction = "bullish"
    elif adx_neg > adx_pos + 5:  # Bearish if -DI significantly higher
        direction = "bearish"
    else:
        direction = "neutral"

    # For funding rate strategy, we only trade in ranging or weak trend markets
    tradeable = adx < config.ADX_THRESHOLD

    return {
        "regime": regime,
        "direction": direction,
        "tradeable": tradeable,
        "adx": adx,
        "adx_pos": adx_pos,
        "adx_neg": adx_neg
    }


def get_regime_from_df(df: pd.DataFrame) -> Dict:
    """
    Get the current market regime from the latest row of DataFrame
    DataFrame should already have ADX columns calculated
    """
    if df.empty:
        return {
            "regime": "unknown",
            "direction": "unknown",
            "tradeable": False,
            "adx": np.nan,
            "adx_pos": np.nan,
            "adx_neg": np.nan
        }

    latest = df.iloc[-1]

    return classify_regime(
        adx=latest["adx"],
        adx_pos=latest["adx_pos"],
        adx_neg=latest["adx_neg"]
    )


def calculate_volatility_regime(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Additional volatility regime indicator using ATR
    Higher volatility can amplify funding rate reversions
    """
    df = df.copy()

    # Calculate True Range
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )

    # ATR
    df["atr"] = df["tr"].rolling(window=lookback).mean()

    # ATR as percentage of price
    df["atr_pct"] = df["atr"] / df["close"] * 100

    # Volatility regime classification
    df["volatility_regime"] = pd.cut(
        df["atr_pct"],
        bins=[0, 1, 2, 4, float("inf")],
        labels=["low", "normal", "high", "extreme"]
    )

    return df


if __name__ == "__main__":
    # Test the module with sample data
    print("Testing regime module...")

    # Create sample data
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    close = 40000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.random.rand(n) * 200
    low = close - np.random.rand(n) * 200

    df = pd.DataFrame({
        "timestamp": dates,
        "open": close - np.random.randn(n) * 50,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.rand(n) * 1000000
    })

    # Calculate ADX
    df = calculate_adx(df)
    print(f"ADX calculated, latest value: {df['adx'].iloc[-1]:.2f}")

    # Get regime
    regime = get_regime_from_df(df)
    print(f"\nMarket Regime:")
    print(f"  Regime: {regime['regime']}")
    print(f"  Direction: {regime['direction']}")
    print(f"  Tradeable: {regime['tradeable']}")
    print(f"  ADX: {regime['adx']:.2f}")
    print(f"  +DI: {regime['adx_pos']:.2f}")
    print(f"  -DI: {regime['adx_neg']:.2f}")

    # Test ranging check
    print(f"\nIs ranging market: {is_ranging_market(regime['adx'])}")
