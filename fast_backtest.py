"""
Fast Backtest - Vectorized approach for speed
Pre-computes all predictions then runs trading simulation
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict

import config
from exchange import fetch_historical_ohlcv
from funding import get_historical_funding_rates, classify_funding_signal
from features import compute_all_features, get_feature_columns
from regime import RegimeDetector, add_regime_to_dataframe
from models.lstm_model import RegimeLSTMEnsemble
from risk import RiskManager, calculate_max_drawdown, calculate_sharpe_ratio, calculate_sortino_ratio
from logger import BacktestLogger
from backtest import print_report, save_pnl_chart


def fast_backtest(
    df: pd.DataFrame,
    regime_detector: RegimeDetector,
    lstm_ensemble: RegimeLSTMEnsemble,
    initial_capital: float = None,
    verbose: bool = False
) -> tuple:
    """
    Fast vectorized backtest
    """
    if initial_capital is None:
        initial_capital = config.INITIAL_CAPITAL

    print("Pre-computing all predictions...")

    # Add regimes
    df = add_regime_to_dataframe(df, regime_detector)

    # Pre-compute LSTM predictions for all regimes
    predictions = {}
    for regime_id in range(config.HMM_N_STATES):
        if regime_id in lstm_ensemble.models:
            model = lstm_ensemble.models[regime_id]
            if model.trained:
                preds = model.predict_proba(df)
                predictions[regime_id] = preds
                print(f"  Regime {regime_id}: {len(preds)} predictions")
            else:
                predictions[regime_id] = np.full(len(df), 0.5)
        else:
            predictions[regime_id] = np.full(len(df), 0.5)

    # Create prediction array indexed by row
    df["lstm_pred"] = 0.5
    for i, row in df.iterrows():
        regime = row.get("regime", 0)
        idx = df.index.get_loc(i)
        if regime in predictions and idx < len(predictions[regime]):
            df.loc[i, "lstm_pred"] = predictions[regime][idx]

    print("Running trading simulation...")

    # Now run simplified trading simulation
    risk_mgr = RiskManager(initial_capital)
    logger = BacktestLogger()

    position = None
    equity_curve = []
    trades_count = 0

    for i in range(config.LSTM_SEQUENCE_LENGTH, len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        price = row["close"]
        atr = row.get("atr", price * 0.02) or price * 0.02
        funding_rate = row.get("funding_rate", 0) or 0
        regime = row.get("regime", 0)
        lstm_pred = row.get("lstm_pred", 0.5)

        if pd.isna(atr) or atr < 1:
            continue

        # Check exit conditions if in position
        if position is not None:
            # Calculate PnL
            if position["type"] == "long":
                pnl_pct = (price - position["entry_price"]) / position["entry_price"]
            else:
                pnl_pct = (position["entry_price"] - price) / position["entry_price"]

            # Dynamic TP/SL
            tp_pct = (atr * config.ATR_TAKE_PROFIT_MULTIPLIER) / position["entry_price"]
            sl_pct = (atr * config.ATR_STOP_LOSS_MULTIPLIER) / position["entry_price"]

            exit_reason = None
            if pnl_pct >= tp_pct:
                exit_reason = f"Take profit: {pnl_pct*100:.2f}% >= {tp_pct*100:.2f}%"
            elif pnl_pct <= -sl_pct:
                exit_reason = f"Stop loss: {pnl_pct*100:.2f}% <= -{sl_pct*100:.2f}%"

            if exit_reason:
                trade = risk_mgr.close_position(
                    exit_price=price,
                    exit_reason=exit_reason,
                    timestamp=ts
                )
                if trade:
                    logger.log_trade(trade)
                    trades_count += 1
                    if verbose:
                        emoji = "🟢" if trade["pnl"] > 0 else "🔴"
                        print(f"{emoji} {ts}: EXIT @ ${price:,.0f} | PnL: ${trade['pnl']:.2f}")
                position = None

        # Check entry conditions
        if position is None:
            can_open = risk_mgr.can_open_position(ts)

            if can_open["allowed"]:
                # Determine direction from LSTM
                confidence = lstm_pred
                if confidence > 0.5:
                    direction = "long"
                else:
                    direction = "short"
                    confidence = 1 - confidence

                # Check funding boost
                funding_signal = classify_funding_signal(funding_rate)
                if funding_signal == direction:
                    confidence = min(confidence + config.FUNDING_BOOST_WEIGHT, 1.0)

                # Enter if confident enough
                if confidence >= config.LSTM_CONFIDENCE_THRESHOLD:
                    sizing = risk_mgr.calculate_position_size(
                        entry_price=price,
                        atr=atr,
                        confidence=confidence
                    )

                    position = risk_mgr.open_position(
                        position_type=direction,
                        entry_price=price,
                        size=sizing["position_size"],
                        timestamp=ts,
                        atr=atr,
                        confidence=confidence,
                        regime=regime
                    )

                    if verbose:
                        print(f"🔵 {ts}: ENTRY {direction.upper()} @ ${price:,.0f} | Conf: {confidence*100:.1f}%")

        # Track equity
        unrealized = 0
        if position:
            if position["type"] == "long":
                unrealized = (price - position["entry_price"]) * position["size"]
            else:
                unrealized = (position["entry_price"] - price) * position["size"]

        equity = risk_mgr.current_capital + unrealized
        equity_curve.append({"timestamp": ts, "capital": equity, "regime": regime})

        # Progress indicator
        if i % 2000 == 0:
            print(f"  Progress: {i}/{len(df)} candles, {trades_count} trades, Capital: ${equity:,.0f}")

    # Close remaining position
    if position is not None:
        last_price = df.iloc[-1]["close"]
        trade = risk_mgr.close_position(
            exit_price=last_price,
            exit_reason="End of backtest",
            timestamp=df.iloc[-1]["timestamp"]
        )
        if trade:
            logger.log_trade(trade)

    # Create equity DataFrame
    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        running_max = equity_df["capital"].cummax()
        equity_df["drawdown"] = (equity_df["capital"] - running_max) / running_max
        equity_df["returns"] = equity_df["capital"].pct_change().fillna(0)

    # Calculate stats
    stats = calculate_stats(risk_mgr, equity_df, logger)

    return stats, equity_df, logger


def calculate_stats(risk_mgr: RiskManager, equity_df: pd.DataFrame, logger: BacktestLogger) -> Dict:
    """Calculate comprehensive backtest statistics"""
    base_stats = risk_mgr.get_statistics()

    if equity_df.empty:
        base_stats.update({"sharpe": 0, "sortino": 0, "max_drawdown": 0, "yearly": {}, "regime_breakdown": {}})
        return base_stats

    base_stats["sharpe"] = calculate_sharpe_ratio(equity_df["returns"])
    base_stats["sortino"] = calculate_sortino_ratio(equity_df["returns"])
    base_stats["max_drawdown"] = calculate_max_drawdown(equity_df["capital"]) * 100

    trades_df = pd.DataFrame(logger.trades) if logger.trades else pd.DataFrame()

    if not trades_df.empty:
        trades_df["year"] = pd.to_datetime(trades_df["exit_time"]).dt.year
        yearly = {}
        for year in trades_df["year"].unique():
            year_trades = trades_df[trades_df["year"] == year]
            year_wins = year_trades[year_trades["win"]]
            yearly[int(year)] = {
                "trades": len(year_trades),
                "pnl": year_trades["pnl"].sum(),
                "win_rate": len(year_wins) / len(year_trades) * 100 if len(year_trades) > 0 else 0
            }
        base_stats["yearly"] = yearly

        regime_breakdown = {}
        for regime in trades_df["regime"].unique():
            regime_trades = trades_df[trades_df["regime"] == regime]
            regime_wins = regime_trades[regime_trades["win"]]
            regime_breakdown[int(regime)] = {
                "trades": len(regime_trades),
                "pnl": regime_trades["pnl"].sum(),
                "win_rate": len(regime_wins) / len(regime_trades) * 100 if len(regime_trades) > 0 else 0
            }
        base_stats["regime_breakdown"] = regime_breakdown
    else:
        base_stats["yearly"] = {}
        base_stats["regime_breakdown"] = {}

    return base_stats


def main():
    """Main fast backtest execution"""
    print("🚀 ATLAS BOT - FAST BACKTEST")
    print("=" * 70)

    # Check if models exist
    if not os.path.exists(config.HMM_MODEL_PATH):
        print("❌ No pre-trained models found. Run full backtest.py first.")
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

    lstm_ensemble = RegimeLSTMEnsemble(n_regimes=config.HMM_N_STATES)
    lstm_ensemble.load_all()

    # Run fast backtest
    print("\n" + "=" * 70)
    print("RUNNING FAST BACKTEST")
    print("=" * 70)

    stats, equity_df, logger = fast_backtest(df, regime_detector, lstm_ensemble, verbose=False)

    # Print report
    print_report(stats)

    # Save chart
    save_pnl_chart(equity_df, logger)

    print("\n" + "=" * 70)
    print("✅ FAST BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
