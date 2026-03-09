"""
Quick Backtest - Uses pre-trained models for faster execution
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

import config
from exchange import fetch_historical_ohlcv
from funding import get_historical_funding_rates
from features import compute_all_features, get_feature_columns
from regime import RegimeDetector, add_regime_to_dataframe
from models.lstm_model import RegimeLSTMEnsemble
from backtest import run_backtest, calculate_stats, print_report, save_pnl_chart
from logger import BacktestLogger


def main():
    """Quick backtest with pre-trained models"""
    print("🚀 ATLAS BOT - QUICK BACKTEST (Pre-trained Models)")
    print("=" * 70)

    # Check if models exist
    hmm_path = config.HMM_MODEL_PATH
    if not os.path.exists(hmm_path):
        print("❌ No pre-trained models found. Run full backtest first.")
        return

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.LOOKBACK_DAYS)

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${config.INITIAL_CAPITAL:,.2f}")

    # Fetch data
    print("\n" + "=" * 70)
    print("FETCHING DATA")
    print("=" * 70)

    ohlcv_df = fetch_historical_ohlcv(
        symbol=config.BINANCE_SYMBOL,
        start_date=start_date,
        end_date=end_date
    )
    print(f"✅ OHLCV: {len(ohlcv_df)} candles")

    funding_df = get_historical_funding_rates(
        start_time=start_date,
        end_time=end_date
    )
    print(f"✅ Funding: {len(funding_df)} records")

    if ohlcv_df.empty:
        print("❌ Failed to fetch data")
        return

    # Compute features
    print("\n" + "=" * 70)
    print("COMPUTING FEATURES")
    print("=" * 70)
    df = compute_all_features(ohlcv_df, funding_df, include_sentiment=False)
    print(f"✅ Features computed: {len(get_feature_columns())} features")

    # Load pre-trained models
    print("\n" + "=" * 70)
    print("LOADING PRE-TRAINED MODELS")
    print("=" * 70)

    regime_detector = RegimeDetector(n_states=config.HMM_N_STATES)
    regime_detector.load()
    regime_detector.print_regime_summary()

    lstm_ensemble = RegimeLSTMEnsemble(n_regimes=config.HMM_N_STATES)
    lstm_ensemble.load_all()

    # Run main backtest (verbose=False for speed)
    print("\n" + "=" * 70)
    print("RUNNING BACKTEST")
    print("=" * 70)
    print("Running with verbose=False for speed...")

    stats, equity_df, logger = run_backtest(
        df, regime_detector, lstm_ensemble, verbose=False
    )

    # Print report
    print_report(stats)

    # Save chart
    save_pnl_chart(equity_df, logger)

    print("\n" + "=" * 70)
    print("✅ QUICK BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
