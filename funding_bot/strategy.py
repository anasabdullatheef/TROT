"""
Strategy Module
Combines funding rate signal with RSI and ADX confirmations
Entry conditions:
- Funding rate extreme (>+0.05% for short, <-0.03% for long)
- RSI confirms momentum exhaustion (>65 for short, <35 for long)
- ADX < 25 (ranging market, not strong trend)
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from typing import Dict, Optional, Tuple
import config
from funding import classify_funding_signal
from regime import calculate_adx, is_ranging_market


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all required indicators: RSI and ADX
    """
    df = df.copy()

    # Calculate RSI (fillna=False to preserve NaN during warmup)
    rsi_indicator = RSIIndicator(close=df["close"], window=config.RSI_PERIOD, fillna=False)
    df["rsi"] = rsi_indicator.rsi()

    # Calculate ADX (from regime module)
    df = calculate_adx(df, period=config.ADX_PERIOD)

    return df


def check_entry_conditions(
    funding_rate: float,
    rsi: float,
    adx: float,
    current_position: str = None
) -> Dict:
    """
    Check if all entry conditions are met

    Args:
        funding_rate: Current funding rate (decimal, e.g., 0.0005 = 0.05%)
        rsi: Current RSI value (0-100)
        adx: Current ADX value
        current_position: 'long', 'short', or None

    Returns:
        Dict with signal info:
            - signal: 'long', 'short', or None
            - funding_signal: bool
            - rsi_signal: bool
            - adx_signal: bool
            - reasons: list of strings explaining the decision
    """
    result = {
        "signal": None,
        "funding_signal": False,
        "rsi_signal": False,
        "adx_signal": False,
        "reasons": []
    }

    # Don't enter if we already have a position
    if current_position is not None:
        result["reasons"].append(f"Already in {current_position} position")
        return result

    # Check ADX first (market regime filter)
    adx_ok = is_ranging_market(adx)
    result["adx_signal"] = adx_ok
    if not adx_ok:
        result["reasons"].append(f"ADX={adx:.1f} > {config.ADX_THRESHOLD} (trending market, skip)")
        return result
    result["reasons"].append(f"ADX={adx:.1f} < {config.ADX_THRESHOLD} (ranging market, ok)")

    # Check funding rate signal
    funding_signal = classify_funding_signal(funding_rate)

    if funding_signal == "short":
        # Funding is extremely positive (longs overleveraged)
        result["funding_signal"] = True
        result["reasons"].append(
            f"Funding={funding_rate*100:.4f}% > {config.FUNDING_LONG_THRESHOLD*100:.2f}% (overleveraged long)"
        )

        # Check RSI for short confirmation (overbought)
        if rsi >= config.RSI_OVERBOUGHT:
            result["rsi_signal"] = True
            result["signal"] = "short"
            result["reasons"].append(f"RSI={rsi:.1f} >= {config.RSI_OVERBOUGHT} (overbought, confirm short)")
        else:
            result["reasons"].append(f"RSI={rsi:.1f} < {config.RSI_OVERBOUGHT} (not overbought, wait)")

    elif funding_signal == "long":
        # Funding is extremely negative (shorts overleveraged)
        result["funding_signal"] = True
        result["reasons"].append(
            f"Funding={funding_rate*100:.4f}% < {config.FUNDING_SHORT_THRESHOLD*100:.2f}% (overleveraged short)"
        )

        # Check RSI for long confirmation (oversold)
        if rsi <= config.RSI_OVERSOLD:
            result["rsi_signal"] = True
            result["signal"] = "long"
            result["reasons"].append(f"RSI={rsi:.1f} <= {config.RSI_OVERSOLD} (oversold, confirm long)")
        else:
            result["reasons"].append(f"RSI={rsi:.1f} > {config.RSI_OVERSOLD} (not oversold, wait)")

    else:
        result["reasons"].append(
            f"Funding={funding_rate*100:.4f}% in normal range "
            f"[{config.FUNDING_SHORT_THRESHOLD*100:.2f}%, {config.FUNDING_LONG_THRESHOLD*100:.2f}%]"
        )

    return result


def check_exit_conditions(
    entry_price: float,
    current_price: float,
    position_type: str,
    funding_rate: float
) -> Dict:
    """
    Check if exit conditions are met for an open position

    Exit conditions:
    1. Take profit hit (2% in signal direction)
    2. Stop loss hit (1.5%)
    3. Funding rate normalized (between -0.01% and +0.01%)

    Returns:
        Dict with:
            - exit: bool
            - reason: str
            - pnl_pct: float
    """
    if position_type == "long":
        pnl_pct = (current_price - entry_price) / entry_price
    else:  # short
        pnl_pct = (entry_price - current_price) / entry_price

    result = {
        "exit": False,
        "reason": None,
        "pnl_pct": pnl_pct
    }

    # Check take profit
    if pnl_pct >= config.TAKE_PROFIT_PCT:
        result["exit"] = True
        result["reason"] = f"Take profit hit: {pnl_pct*100:.2f}% >= {config.TAKE_PROFIT_PCT*100}%"
        return result

    # Check stop loss
    if pnl_pct <= -config.STOP_LOSS_PCT:
        result["exit"] = True
        result["reason"] = f"Stop loss hit: {pnl_pct*100:.2f}% <= -{config.STOP_LOSS_PCT*100}%"
        return result

    # Check if funding rate has normalized
    funding_signal = classify_funding_signal(funding_rate)
    if funding_signal == "exit":
        result["exit"] = True
        result["reason"] = (
            f"Funding normalized: {funding_rate*100:.4f}% "
            f"in [{config.FUNDING_NEUTRAL_LOW*100:.2f}%, {config.FUNDING_NEUTRAL_HIGH*100:.2f}%]"
        )
        return result

    return result


def generate_signals_for_backtest(df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate entry/exit signals for backtesting

    Args:
        df: OHLCV DataFrame with timestamp
        funding_df: Funding rate DataFrame with timestamp, funding_rate

    Returns:
        DataFrame with additional columns: signal, entry_condition
    """
    df = df.copy()

    # Calculate indicators
    df = calculate_indicators(df)

    # Merge funding rates (forward fill since funding is every 8h, candles are 1h)
    df = df.sort_values("timestamp").reset_index(drop=True)
    funding_df = funding_df.sort_values("timestamp").reset_index(drop=True)

    # Merge on timestamp with tolerance (funding rate applies to following candles)
    df = pd.merge_asof(
        df,
        funding_df,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("8h")
    )

    # Fill forward missing funding rates
    df["funding_rate"] = df["funding_rate"].ffill()

    # Initialize signal column
    df["signal"] = None
    df["entry_reason"] = None

    # Generate signals where we have funding data
    for i in range(len(df)):
        if pd.isna(df.loc[i, "funding_rate"]):
            continue

        funding_rate = df.loc[i, "funding_rate"]
        rsi = df.loc[i, "rsi"]
        adx = df.loc[i, "adx"]

        conditions = check_entry_conditions(funding_rate, rsi, adx)

        if conditions["signal"]:
            df.loc[i, "signal"] = conditions["signal"]
            df.loc[i, "entry_reason"] = "; ".join(conditions["reasons"])

    return df


if __name__ == "__main__":
    # Test the module
    print("Testing strategy module...")

    # Test entry conditions
    print("\n--- Entry Condition Tests ---")

    # Test case 1: All conditions met for short
    result = check_entry_conditions(
        funding_rate=0.0008,  # 0.08% (very high)
        rsi=72,               # Overbought
        adx=20                # Ranging
    )
    print(f"\nTest 1 - High funding, high RSI, low ADX:")
    print(f"  Signal: {result['signal']}")
    for reason in result["reasons"]:
        print(f"  - {reason}")

    # Test case 2: Funding high but RSI not overbought
    result = check_entry_conditions(
        funding_rate=0.0008,  # 0.08%
        rsi=55,               # Not overbought
        adx=20                # Ranging
    )
    print(f"\nTest 2 - High funding, normal RSI:")
    print(f"  Signal: {result['signal']}")
    for reason in result["reasons"]:
        print(f"  - {reason}")

    # Test case 3: All conditions met for long
    result = check_entry_conditions(
        funding_rate=-0.0005,  # -0.05% (very negative)
        rsi=28,                # Oversold
        adx=18                 # Ranging
    )
    print(f"\nTest 3 - Negative funding, low RSI, low ADX:")
    print(f"  Signal: {result['signal']}")
    for reason in result["reasons"]:
        print(f"  - {reason}")

    # Test case 4: Strong trend (ADX too high)
    result = check_entry_conditions(
        funding_rate=0.0008,
        rsi=72,
        adx=35  # Strong trend
    )
    print(f"\nTest 4 - High ADX (trending market):")
    print(f"  Signal: {result['signal']}")
    for reason in result["reasons"]:
        print(f"  - {reason}")

    # Test exit conditions
    print("\n--- Exit Condition Tests ---")

    # Take profit
    exit_result = check_exit_conditions(
        entry_price=40000,
        current_price=40850,  # ~2.1% up
        position_type="long",
        funding_rate=0.0003
    )
    print(f"\nTake profit test: {exit_result}")

    # Stop loss
    exit_result = check_exit_conditions(
        entry_price=40000,
        current_price=39350,  # ~1.6% down
        position_type="long",
        funding_rate=0.0003
    )
    print(f"Stop loss test: {exit_result}")

    # Funding normalized
    exit_result = check_exit_conditions(
        entry_price=40000,
        current_price=40200,
        position_type="long",
        funding_rate=0.00005  # Normalized to near zero
    )
    print(f"Funding normalized test: {exit_result}")
