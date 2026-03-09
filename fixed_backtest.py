"""
Fixed Backtest - Addresses all credibility issues:
1. Strict time-based train/test split (18mo train, 6mo test)
2. No lookahead bias in features
3. Fixed 2% position sizing (no Kelly)
4. Realistic costs (0.1% fee + 0.05% slippage = 0.15% per trade)
5. 20% max drawdown circuit breaker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from typing import Dict, Tuple

import config
from exchange import fetch_historical_ohlcv
from funding import get_historical_funding_rates, classify_funding_signal
from features import compute_all_features, get_feature_columns
from regime import RegimeDetector, add_regime_to_dataframe
from models.lstm_model import RegimeLSTMEnsemble
from risk import calculate_max_drawdown, calculate_sharpe_ratio, calculate_sortino_ratio
from logger import BacktestLogger


# Fixed constants
POSITION_SIZE_PCT = 0.02  # 2% of capital per trade
TRADING_FEE = 0.001  # 0.1% taker fee
SLIPPAGE = 0.0005  # 0.05% slippage
TOTAL_COST_PCT = TRADING_FEE + SLIPPAGE  # 0.15% per trade
MAX_DRAWDOWN_LIMIT = 0.20  # 20% circuit breaker
TRAIN_MONTHS = 18
TEST_MONTHS = 6


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict time-based train/test split
    Train: first 18 months
    Test: last 6 months
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    total_days = (df["timestamp"].max() - df["timestamp"].min()).days
    train_days = int(total_days * TRAIN_MONTHS / (TRAIN_MONTHS + TEST_MONTHS))

    cutoff_date = df["timestamp"].min() + timedelta(days=train_days)

    train_df = df[df["timestamp"] < cutoff_date].copy()
    test_df = df[df["timestamp"] >= cutoff_date].copy()

    print(f"\n📅 Train/Test Split:")
    print(f"   Train: {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()} ({len(train_df)} candles)")
    print(f"   Test:  {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()} ({len(test_df)} candles)")
    print(f"   Cutoff: {cutoff_date.date()}")

    return train_df, test_df


def train_models_on_train_set(train_df: pd.DataFrame) -> Tuple[RegimeDetector, RegimeLSTMEnsemble]:
    """Train models ONLY on training data - no lookahead"""
    print("\n" + "=" * 70)
    print("TRAINING MODELS (ON TRAIN SET ONLY)")
    print("=" * 70)

    # Train HMM
    print("\n🔧 Training HMM Regime Detector...")
    regime_detector = RegimeDetector(n_states=config.HMM_N_STATES)
    regime_detector.fit(train_df)

    # Add regime labels to training data
    train_df = add_regime_to_dataframe(train_df, regime_detector)

    # Train LSTM for each regime
    print("\n🔧 Training LSTM Models per Regime...")
    lstm_ensemble = RegimeLSTMEnsemble(n_regimes=config.HMM_N_STATES)
    lstm_results = lstm_ensemble.train_all(train_df, include_sentiment=False)

    print("\n📊 LSTM Training Results (Train Set):")
    for regime, metrics in lstm_results.items():
        print(f"   Regime {regime}: Acc={metrics['accuracy']:.2%}, AUC={metrics['auc']:.2f}, Samples={metrics['samples']}")

    return regime_detector, lstm_ensemble


def run_fixed_backtest(
    test_df: pd.DataFrame,
    regime_detector: RegimeDetector,
    lstm_ensemble: RegimeLSTMEnsemble,
    initial_capital: float = None
) -> Tuple[Dict, pd.DataFrame, BacktestLogger]:
    """
    Run backtest on TEST SET ONLY with all fixes:
    - Fixed 2% position sizing
    - 0.15% costs per trade
    - 20% drawdown circuit breaker
    """
    if initial_capital is None:
        initial_capital = config.INITIAL_CAPITAL

    print("\n" + "=" * 70)
    print("RUNNING FIXED BACKTEST (TEST SET ONLY)")
    print("=" * 70)
    print(f"Position Size: {POSITION_SIZE_PCT*100:.1f}% of capital")
    print(f"Trading Costs: {TOTAL_COST_PCT*100:.2f}% per trade")
    print(f"Max Drawdown Limit: {MAX_DRAWDOWN_LIMIT*100:.0f}%")

    # Add regimes to test data using trained model
    test_df = add_regime_to_dataframe(test_df, regime_detector)

    # Pre-compute LSTM predictions
    print("\nPre-computing predictions...")
    predictions = {}
    for regime_id in range(config.HMM_N_STATES):
        if regime_id in lstm_ensemble.models and lstm_ensemble.models[regime_id].trained:
            preds = lstm_ensemble.models[regime_id].predict_proba(test_df)
            predictions[regime_id] = preds
        else:
            predictions[regime_id] = np.full(len(test_df), 0.5)

    # Map predictions to dataframe
    test_df["lstm_pred"] = 0.5
    for i, (idx, row) in enumerate(test_df.iterrows()):
        regime = row.get("regime", 0)
        if regime in predictions and i < len(predictions[regime]):
            test_df.loc[idx, "lstm_pred"] = predictions[regime][i]

    # Trading simulation
    capital = initial_capital
    peak_capital = initial_capital
    position = None
    equity_curve = []
    logger = BacktestLogger()
    trades_count = 0
    circuit_breaker_triggered = False

    print("\nRunning simulation...")

    for i in range(config.LSTM_SEQUENCE_LENGTH, len(test_df)):
        row = test_df.iloc[i]
        ts = row["timestamp"]
        price = row["close"]
        atr = row.get("atr", price * 0.02) or price * 0.02
        funding_rate = row.get("funding_rate", 0) or 0
        regime = row.get("regime", 0)
        lstm_pred = row.get("lstm_pred", 0.5)

        if pd.isna(atr) or atr < 1:
            continue

        # Check circuit breaker
        current_dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
        if current_dd >= MAX_DRAWDOWN_LIMIT and not circuit_breaker_triggered:
            circuit_breaker_triggered = True
            print(f"\n⛔ CIRCUIT BREAKER TRIGGERED at {ts}")
            print(f"   Drawdown: {current_dd*100:.2f}% >= {MAX_DRAWDOWN_LIMIT*100:.0f}%")

            # Close any open position
            if position is not None:
                exit_price = price * (1 - TOTAL_COST_PCT)  # Cost on exit
                if position["type"] == "long":
                    pnl = (exit_price - position["entry_price"]) * position["size"]
                else:
                    pnl = (position["entry_price"] - exit_price) * position["size"]

                capital += pnl
                logger.log_trade({
                    "entry_time": position["entry_time"],
                    "exit_time": ts,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "position_type": position["type"],
                    "size": position["size"],
                    "pnl": pnl,
                    "pnl_pct": pnl / (position["entry_price"] * position["size"]),
                    "exit_reason": "Circuit breaker",
                    "regime": position["regime"],
                    "win": pnl > 0
                })
                position = None

        if circuit_breaker_triggered:
            equity_curve.append({"timestamp": ts, "capital": capital, "regime": regime})
            continue

        # Check exit conditions if in position
        if position is not None:
            # Calculate PnL (with costs applied at entry, so just calc raw PnL here)
            if position["type"] == "long":
                raw_pnl_pct = (price - position["entry_price"]) / position["entry_price"]
            else:
                raw_pnl_pct = (position["entry_price"] - price) / position["entry_price"]

            # Dynamic TP/SL based on ATR
            tp_pct = (atr * config.ATR_TAKE_PROFIT_MULTIPLIER) / position["entry_price"]
            sl_pct = (atr * config.ATR_STOP_LOSS_MULTIPLIER) / position["entry_price"]

            exit_reason = None
            if raw_pnl_pct >= tp_pct:
                exit_reason = f"TP: {raw_pnl_pct*100:.2f}% >= {tp_pct*100:.2f}%"
            elif raw_pnl_pct <= -sl_pct:
                exit_reason = f"SL: {raw_pnl_pct*100:.2f}% <= -{sl_pct*100:.2f}%"

            if exit_reason:
                # Apply exit costs
                if position["type"] == "long":
                    exit_price = price * (1 - TOTAL_COST_PCT)
                else:
                    exit_price = price * (1 + TOTAL_COST_PCT)

                if position["type"] == "long":
                    pnl = (exit_price - position["entry_price"]) * position["size"]
                else:
                    pnl = (position["entry_price"] - exit_price) * position["size"]

                capital += pnl
                trades_count += 1

                logger.log_trade({
                    "entry_time": position["entry_time"],
                    "exit_time": ts,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "position_type": position["type"],
                    "size": position["size"],
                    "pnl": pnl,
                    "pnl_pct": pnl / (position["entry_price"] * position["size"]),
                    "exit_reason": exit_reason,
                    "regime": position["regime"],
                    "win": pnl > 0
                })
                position = None

        # Check entry conditions
        if position is None and not circuit_breaker_triggered:
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
                # FIXED: 2% position sizing, no Kelly
                position_value = capital * POSITION_SIZE_PCT

                # Apply entry costs
                if direction == "long":
                    entry_price = price * (1 + TOTAL_COST_PCT)
                else:
                    entry_price = price * (1 - TOTAL_COST_PCT)

                position_size = position_value / entry_price

                position = {
                    "type": direction,
                    "entry_price": entry_price,
                    "size": position_size,
                    "entry_time": ts,
                    "regime": regime,
                    "confidence": confidence
                }

        # Track equity
        unrealized = 0
        if position:
            if position["type"] == "long":
                unrealized = (price - position["entry_price"]) * position["size"]
            else:
                unrealized = (position["entry_price"] - price) * position["size"]

        equity = capital + unrealized
        peak_capital = max(peak_capital, equity)
        equity_curve.append({"timestamp": ts, "capital": equity, "regime": regime})

        # Progress
        if i % 1000 == 0:
            dd = (peak_capital - equity) / peak_capital * 100 if peak_capital > 0 else 0
            print(f"  {i}/{len(test_df)} | Trades: {trades_count} | Capital: ${equity:,.0f} | DD: {dd:.1f}%")

    # Close remaining position
    if position is not None:
        last_row = test_df.iloc[-1]
        last_price = last_row["close"]

        if position["type"] == "long":
            exit_price = last_price * (1 - TOTAL_COST_PCT)
            pnl = (exit_price - position["entry_price"]) * position["size"]
        else:
            exit_price = last_price * (1 + TOTAL_COST_PCT)
            pnl = (position["entry_price"] - exit_price) * position["size"]

        capital += pnl
        logger.log_trade({
            "entry_time": position["entry_time"],
            "exit_time": last_row["timestamp"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "position_type": position["type"],
            "size": position["size"],
            "pnl": pnl,
            "pnl_pct": pnl / (position["entry_price"] * position["size"]),
            "exit_reason": "End of backtest",
            "regime": position["regime"],
            "win": pnl > 0
        })

    # Create equity DataFrame
    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        running_max = equity_df["capital"].cummax()
        equity_df["drawdown"] = (equity_df["capital"] - running_max) / running_max
        equity_df["returns"] = equity_df["capital"].pct_change().fillna(0)

    # Calculate stats
    stats = calculate_stats(initial_capital, capital, equity_df, logger, circuit_breaker_triggered)

    return stats, equity_df, logger


def calculate_stats(
    initial_capital: float,
    final_capital: float,
    equity_df: pd.DataFrame,
    logger: BacktestLogger,
    circuit_breaker: bool
) -> Dict:
    """Calculate comprehensive statistics"""
    trades = logger.trades

    total_pnl = final_capital - initial_capital
    return_pct = (final_capital / initial_capital - 1) * 100

    winning_trades = [t for t in trades if t["pnl"] > 0]
    losing_trades = [t for t in trades if t["pnl"] <= 0]

    gross_profit = sum(t["pnl"] for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t["pnl"] for t in losing_trades)) if losing_trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    stats = {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_pnl": total_pnl,
        "return_pct": return_pct,
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": len(winning_trades) / len(trades) * 100 if trades else 0,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "circuit_breaker_triggered": circuit_breaker
    }

    if not equity_df.empty:
        stats["sharpe"] = calculate_sharpe_ratio(equity_df["returns"])
        stats["sortino"] = calculate_sortino_ratio(equity_df["returns"])
        stats["max_drawdown"] = calculate_max_drawdown(equity_df["capital"]) * 100
    else:
        stats["sharpe"] = 0
        stats["sortino"] = 0
        stats["max_drawdown"] = 0

    # Regime breakdown
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    if not trades_df.empty and "regime" in trades_df.columns:
        regime_breakdown = {}
        for regime in trades_df["regime"].unique():
            regime_trades = trades_df[trades_df["regime"] == regime]
            regime_wins = regime_trades[regime_trades["win"]]
            regime_breakdown[int(regime)] = {
                "trades": len(regime_trades),
                "pnl": regime_trades["pnl"].sum(),
                "win_rate": len(regime_wins) / len(regime_trades) * 100 if len(regime_trades) > 0 else 0
            }
        stats["regime_breakdown"] = regime_breakdown
    else:
        stats["regime_breakdown"] = {}

    return stats


def print_report(stats: Dict):
    """Print formatted report"""
    print("\n" + "=" * 70)
    print("📊 FIXED BACKTEST RESULTS (TEST SET ONLY)")
    print("=" * 70)

    # Annualized return (6 month test period)
    annual_return = stats["return_pct"] * 2  # Approximate annualization

    print(f"\n💰 PERFORMANCE (6-Month Test Period)")
    print("-" * 50)
    print(f"Initial Capital:        ${stats['initial_capital']:,.2f}")
    print(f"Final Capital:          ${stats['final_capital']:,.2f}")
    print(f"Total Return:           {stats['return_pct']:+.2f}%")
    print(f"Annualized Return:      {annual_return:+.2f}% (approx)")
    print(f"Total PnL:              ${stats['total_pnl']:,.2f}")

    print(f"\n📈 RISK METRICS")
    print("-" * 50)
    print(f"Sharpe Ratio:           {stats['sharpe']:.2f}")
    print(f"Sortino Ratio:          {stats['sortino']:.2f}")
    print(f"Max Drawdown:           {stats['max_drawdown']:.2f}%")
    print(f"Profit Factor:          {stats['profit_factor']:.2f}")

    if stats.get("circuit_breaker_triggered"):
        print(f"\n⛔ CIRCUIT BREAKER:      TRIGGERED (20% DD limit)")

    print(f"\n🎯 TRADE STATISTICS")
    print("-" * 50)
    print(f"Total Trades:           {stats['total_trades']}")
    print(f"Winning Trades:         {stats['winning_trades']}")
    print(f"Losing Trades:          {stats['losing_trades']}")
    print(f"Win Rate:               {stats['win_rate']:.1f}%")
    print(f"Gross Profit:           ${stats['gross_profit']:,.2f}")
    print(f"Gross Loss:             ${stats['gross_loss']:,.2f}")

    if stats.get("regime_breakdown"):
        print(f"\n🎭 REGIME BREAKDOWN")
        print("-" * 50)
        print(f"{'Regime':<10} {'Trades':<10} {'PnL':<15} {'Win Rate':<10}")
        for regime, data in sorted(stats["regime_breakdown"].items()):
            print(f"R{regime:<9} {data['trades']:<10} ${data['pnl']:>12,.2f}  {data['win_rate']:>8.1f}%")

    # Credibility check
    print(f"\n🔍 CREDIBILITY CHECK")
    print("-" * 50)
    if annual_return > 100:
        print(f"⚠️  WARNING: Annualized return {annual_return:.1f}% > 100%")
        print(f"    This may indicate remaining issues:")
        print(f"    - Check if LSTM has any lookahead in feature computation")
        print(f"    - Verify ATR-based TP/SL aren't too wide")
        print(f"    - Consider if regime detection uses future data")
    else:
        print(f"✅ Annualized return {annual_return:.1f}% is within credible range")


def save_chart(equity_df: pd.DataFrame, stats: Dict):
    """Save equity chart"""
    os.makedirs("data", exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Equity curve
    ax1 = axes[0]
    ax1.plot(equity_df["timestamp"], equity_df["capital"], color="#00B5AD", linewidth=1.5)
    ax1.axhline(y=config.INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5, label="Initial")
    ax1.fill_between(
        equity_df["timestamp"],
        config.INITIAL_CAPITAL,
        equity_df["capital"],
        where=equity_df["capital"] >= config.INITIAL_CAPITAL,
        alpha=0.3, color="#00B5AD"
    )
    ax1.fill_between(
        equity_df["timestamp"],
        config.INITIAL_CAPITAL,
        equity_df["capital"],
        where=equity_df["capital"] < config.INITIAL_CAPITAL,
        alpha=0.3, color="#E0592A"
    )
    ax1.set_title(f"Equity Curve (Test Period) - Return: {stats['return_pct']:.2f}%", fontsize=12)
    ax1.set_ylabel("Capital (USDT)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    ax2.fill_between(
        equity_df["timestamp"],
        0,
        equity_df["drawdown"] * 100,
        color="#E0592A", alpha=0.5
    )
    ax2.axhline(y=-MAX_DRAWDOWN_LIMIT * 100, color="red", linestyle="--", label=f"{MAX_DRAWDOWN_LIMIT*100:.0f}% Limit")
    ax2.set_title("Drawdown", fontsize=12)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/fixed_backtest_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Chart saved to data/fixed_backtest_chart.png")


def main():
    """Main execution"""
    print("🚀 ATLAS BOT - FIXED BACKTEST")
    print("=" * 70)
    print("\nFixes Applied:")
    print("  ✓ Strict time-based train/test split (18mo/6mo)")
    print("  ✓ No lookahead bias - models trained only on past data")
    print("  ✓ Fixed 2% position sizing (no Kelly)")
    print("  ✓ Realistic costs: 0.1% fee + 0.05% slippage = 0.15%/trade")
    print("  ✓ 20% max drawdown circuit breaker")

    # Fetch full data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.LOOKBACK_DAYS)

    print(f"\nFull Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${config.INITIAL_CAPITAL:,.2f}")

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

    # Compute features (no sentiment to avoid leakage)
    print("\n" + "=" * 70)
    print("COMPUTING FEATURES")
    print("=" * 70)
    df = compute_all_features(ohlcv_df, funding_df, include_sentiment=False)
    print(f"✅ Features computed: {len(get_feature_columns())} features")

    # Split train/test
    train_df, test_df = split_train_test(df)

    # Train models on TRAIN SET ONLY
    regime_detector, lstm_ensemble = train_models_on_train_set(train_df)

    # Run backtest on TEST SET ONLY
    stats, equity_df, logger = run_fixed_backtest(test_df, regime_detector, lstm_ensemble)

    # Print report
    print_report(stats)

    # Save chart
    save_chart(equity_df, stats)

    print("\n" + "=" * 70)
    print("✅ FIXED BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
