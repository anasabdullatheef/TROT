"""
Backtesting Module
Full 2-year backtest with comprehensive metrics and comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import os

import config
from exchange import fetch_historical_ohlcv
from funding import get_historical_funding_rates
from features import compute_all_features, get_feature_columns
from regime import RegimeDetector, add_regime_to_dataframe
from models.lstm_model import RegimeLSTMEnsemble
from strategy import TradingStrategy
from risk import RiskManager, calculate_max_drawdown, calculate_sharpe_ratio, calculate_sortino_ratio
from logger import BacktestLogger


def fetch_all_data(
    start_date: datetime = None,
    end_date: datetime = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch OHLCV and funding rate data"""
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=config.LOOKBACK_DAYS)

    print("=" * 70)
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

    return ohlcv_df, funding_df


def train_models(
    df: pd.DataFrame,
    include_sentiment: bool = True
) -> Tuple[RegimeDetector, RegimeLSTMEnsemble]:
    """Train HMM and LSTM models"""
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    # Train HMM regime detector
    print("\n🔧 Training HMM Regime Detector...")
    regime_detector = RegimeDetector(n_states=config.HMM_N_STATES)
    regime_detector.fit(df)
    regime_detector.print_regime_summary()

    # Add regime labels to data
    df = add_regime_to_dataframe(df, regime_detector)

    # Train LSTM for each regime
    print("\n🔧 Training LSTM Models per Regime...")
    lstm_ensemble = RegimeLSTMEnsemble(n_regimes=config.HMM_N_STATES)
    lstm_results = lstm_ensemble.train_all(df, include_sentiment=include_sentiment)

    print("\n📊 LSTM Training Results:")
    for regime, metrics in lstm_results.items():
        print(f"   Regime {regime}: Acc={metrics['accuracy']:.2%}, AUC={metrics['auc']:.2f}, Samples={metrics['samples']}")

    return regime_detector, lstm_ensemble


def run_backtest(
    df: pd.DataFrame,
    regime_detector: RegimeDetector,
    lstm_ensemble: RegimeLSTMEnsemble,
    initial_capital: float = None,
    verbose: bool = True
) -> Tuple[Dict, pd.DataFrame, BacktestLogger]:
    """
    Run full backtest

    Returns:
        (stats, equity_df, logger)
    """
    if initial_capital is None:
        initial_capital = config.INITIAL_CAPITAL

    risk_mgr = RiskManager(initial_capital)
    strategy = TradingStrategy(regime_detector, lstm_ensemble)
    logger = BacktestLogger()

    # Add regime to dataframe
    df = add_regime_to_dataframe(df, regime_detector)

    position = None
    equity_curve = []
    finbert_history = []

    print("\n" + "=" * 70)
    print("RUNNING BACKTEST")
    print("=" * 70)

    for i in range(config.LSTM_SEQUENCE_LENGTH, len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        price = row["close"]
        atr = row.get("atr", price * 0.02) or price * 0.02
        funding_rate = row.get("funding_rate", 0) or 0
        finbert = row.get("finbert_1h", 0) or 0
        regime = row.get("regime", 0)

        # Track finbert history
        finbert_history.append(finbert)
        if len(finbert_history) > 10:
            finbert_history = finbert_history[-10:]

        # Skip rows with missing critical data
        if pd.isna(atr) or atr < 1:
            continue

        # Check exit conditions if in position
        if position is not None:
            exit_check = strategy.check_exit_conditions(
                entry_price=position["entry_price"],
                current_price=price,
                position_type=position["type"],
                atr=atr,
                funding_rate=funding_rate,
                finbert_history=finbert_history
            )

            if exit_check["exit"]:
                trade = risk_mgr.close_position(
                    exit_price=price,
                    exit_reason=exit_check["reason"],
                    timestamp=ts
                )

                if trade:
                    logger.log_trade(trade)
                    if verbose:
                        emoji = "🟢" if trade["pnl"] > 0 else "🔴"
                        print(f"{emoji} {ts}: EXIT @ ${price:,.0f} | "
                              f"PnL: ${trade['pnl']:.2f} ({trade['pnl_pct']*100:+.2f}%) | "
                              f"{exit_check['reason'][:40]}")

                position = None

        # Check entry conditions if no position
        if position is None:
            # Get data up to current point
            current_df = df.iloc[:i + 1].copy()

            # Check if we can open position
            can_open = risk_mgr.can_open_position(ts)

            if can_open["allowed"]:
                signal = strategy.get_signal(current_df)

                if signal["halt"]:
                    if verbose:
                        print(f"⛔ {ts}: HALT - {signal['halt_reason']}")
                    continue

                if signal["signal"] == "enter" and signal["direction"]:
                    # Calculate position size
                    sizing = risk_mgr.calculate_position_size(
                        entry_price=price,
                        atr=atr,
                        confidence=signal["confidence"]
                    )

                    # Open position
                    position = risk_mgr.open_position(
                        position_type=signal["direction"],
                        entry_price=price,
                        size=sizing["position_size"],
                        timestamp=ts,
                        atr=atr,
                        confidence=signal["confidence"],
                        regime=signal["regime"]
                    )

                    if verbose:
                        boost = " [+FUNDING]" if signal["funding_boost"] else ""
                        print(f"🔵 {ts}: ENTRY {signal['direction'].upper()} @ ${price:,.0f} | "
                              f"Regime {signal['regime']} | Conf: {signal['confidence']*100:.1f}%{boost}")

        # Track equity
        unrealized = 0
        if position:
            if position["type"] == "long":
                unrealized = (price - position["entry_price"]) * position["size"]
            else:
                unrealized = (position["entry_price"] - price) * position["size"]

        equity = risk_mgr.current_capital + unrealized
        equity_curve.append({"timestamp": ts, "capital": equity, "regime": regime})

    # Close any remaining position
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


def calculate_stats(
    risk_mgr: RiskManager,
    equity_df: pd.DataFrame,
    logger: BacktestLogger
) -> Dict:
    """Calculate comprehensive backtest statistics"""
    base_stats = risk_mgr.get_statistics()

    if equity_df.empty:
        base_stats["sharpe"] = 0
        base_stats["sortino"] = 0
        base_stats["max_drawdown"] = 0
        base_stats["yearly"] = {}
        base_stats["regime_breakdown"] = {}
        return base_stats

    # Add Sharpe, Sortino, Max DD
    base_stats["sharpe"] = calculate_sharpe_ratio(equity_df["returns"])
    base_stats["sortino"] = calculate_sortino_ratio(equity_df["returns"])
    base_stats["max_drawdown"] = calculate_max_drawdown(equity_df["capital"]) * 100

    # Year by year breakdown
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

        # Regime breakdown
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


def run_comparison_backtest(
    df: pd.DataFrame,
    regime_detector: RegimeDetector,
    lstm_ensemble: RegimeLSTMEnsemble
) -> Dict:
    """
    Run backtest WITH and WITHOUT sentiment features for comparison
    """
    print("\n" + "=" * 70)
    print("COMPARISON: WITH vs WITHOUT SENTIMENT")
    print("=" * 70)

    results = {}

    # WITH sentiment (already trained models)
    print("\n📊 Running WITH sentiment features...")
    stats_with, equity_with, _ = run_backtest(
        df, regime_detector, lstm_ensemble, verbose=False
    )
    results["with_sentiment"] = stats_with

    # WITHOUT sentiment - retrain models
    print("\n📊 Retraining WITHOUT sentiment features...")
    df_no_sent = df.copy()
    df_no_sent["fng_value"] = 0.5
    df_no_sent["finbert_1h"] = 0
    df_no_sent["finbert_4h"] = 0

    # Retrain LSTM without sentiment
    lstm_no_sent = RegimeLSTMEnsemble(n_regimes=config.HMM_N_STATES)
    lstm_no_sent.train_all(df_no_sent, include_sentiment=False)

    print("\n📊 Running WITHOUT sentiment features...")
    stats_without, equity_without, _ = run_backtest(
        df_no_sent, regime_detector, lstm_no_sent, verbose=False
    )
    results["without_sentiment"] = stats_without

    return results


def print_report(stats: Dict, title: str = "BACKTEST RESULTS"):
    """Print formatted backtest report"""
    print("\n" + "=" * 70)
    print(f"📊 {title}")
    print("=" * 70)

    print(f"\n💰 OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"Initial Capital:      ${config.INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:        ${stats.get('current_capital', 0):,.2f}")
    print(f"Total Return:         {stats.get('return_pct', 0):+.2f}%")
    print(f"Total PnL:            ${stats.get('total_pnl', 0):,.2f}")

    print(f"\n📈 RISK METRICS")
    print("-" * 40)
    print(f"Sharpe Ratio:         {stats.get('sharpe', 0):.2f}")
    print(f"Sortino Ratio:        {stats.get('sortino', 0):.2f}")
    print(f"Max Drawdown:         {stats.get('max_drawdown', 0):.2f}%")
    print(f"Profit Factor:        {stats.get('profit_factor', 0):.2f}")

    print(f"\n🎯 TRADE STATISTICS")
    print("-" * 40)
    print(f"Total Trades:         {stats.get('total_trades', 0)}")
    print(f"Winning Trades:       {stats.get('winning_trades', 0)}")
    print(f"Losing Trades:        {stats.get('losing_trades', 0)}")
    print(f"Win Rate:             {stats.get('win_rate', 0):.1f}%")

    # Year by year
    if stats.get("yearly"):
        print(f"\n📅 YEAR-BY-YEAR BREAKDOWN")
        print("-" * 40)
        print(f"{'Year':<8} {'Trades':<10} {'PnL':<15} {'Win Rate':<10}")
        for year, data in sorted(stats["yearly"].items()):
            print(f"{year:<8} {data['trades']:<10} ${data['pnl']:>12,.2f}  {data['win_rate']:>8.1f}%")

    # Regime breakdown
    if stats.get("regime_breakdown"):
        print(f"\n🎭 REGIME BREAKDOWN")
        print("-" * 40)
        print(f"{'Regime':<10} {'Trades':<10} {'PnL':<15} {'Win Rate':<10}")
        for regime, data in sorted(stats["regime_breakdown"].items()):
            print(f"R{regime:<9} {data['trades']:<10} ${data['pnl']:>12,.2f}  {data['win_rate']:>8.1f}%")


def save_pnl_chart(
    equity_df: pd.DataFrame,
    logger: BacktestLogger,
    comparison: Dict = None
):
    """Save PnL chart"""
    os.makedirs("data", exist_ok=True)

    n_plots = 3 if comparison is None else 4
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))

    colors = {"main": "#00B5AD", "profit": "#00B5AD", "loss": "#E0592A"}

    # 1. Equity curve
    ax1 = axes[0]
    ax1.plot(equity_df["timestamp"], equity_df["capital"], color=colors["main"], linewidth=1.5)
    ax1.fill_between(
        equity_df["timestamp"],
        config.INITIAL_CAPITAL,
        equity_df["capital"],
        where=equity_df["capital"] >= config.INITIAL_CAPITAL,
        alpha=0.3, color=colors["profit"]
    )
    ax1.fill_between(
        equity_df["timestamp"],
        config.INITIAL_CAPITAL,
        equity_df["capital"],
        where=equity_df["capital"] < config.INITIAL_CAPITAL,
        alpha=0.3, color=colors["loss"]
    )
    ax1.axhline(y=config.INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Equity Curve", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Capital (USDT)")
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = axes[1]
    ax2.fill_between(
        equity_df["timestamp"],
        0,
        equity_df["drawdown"] * 100,
        color=colors["loss"], alpha=0.5
    )
    ax2.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    # 3. Monthly PnL
    ax3 = axes[2]
    trades_df = pd.DataFrame(logger.trades) if logger.trades else pd.DataFrame()
    if not trades_df.empty:
        trades_df["month"] = pd.to_datetime(trades_df["exit_time"]).dt.to_period("M")
        monthly_pnl = trades_df.groupby("month")["pnl"].sum()
        monthly_colors = [colors["profit"] if x >= 0 else colors["loss"] for x in monthly_pnl.values]
        monthly_dates = [pd.Timestamp(str(m)) for m in monthly_pnl.index]
        ax3.bar(monthly_dates, monthly_pnl.values, color=monthly_colors, width=20, alpha=0.7)
        ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax3.set_title("Monthly PnL", fontsize=14, fontweight="bold")
    ax3.set_ylabel("PnL (USDT)")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Comparison (if provided)
    if comparison is not None:
        ax4 = axes[3]
        metrics = ["return_pct", "sharpe", "win_rate", "total_trades"]
        labels = ["Return %", "Sharpe", "Win Rate %", "Trades"]
        x = np.arange(len(metrics))
        width = 0.35

        with_vals = [comparison["with_sentiment"].get(m, 0) for m in metrics]
        without_vals = [comparison["without_sentiment"].get(m, 0) for m in metrics]

        ax4.bar(x - width/2, with_vals, width, label="With Sentiment", color=colors["main"])
        ax4.bar(x + width/2, without_vals, width, label="Without Sentiment", color="#F0AB00")
        ax4.set_ylabel("Value")
        ax4.set_title("With vs Without Sentiment Features", fontsize=14, fontweight="bold")
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(config.PNL_CHART_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Chart saved to {config.PNL_CHART_FILE}")


def main():
    """Main backtest execution"""
    print("🚀 ATLAS BOT - FULL BACKTEST")
    print("=" * 70)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.LOOKBACK_DAYS)

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${config.INITIAL_CAPITAL:,.2f}")

    # Fetch data
    ohlcv_df, funding_df = fetch_all_data(start_date, end_date)

    if ohlcv_df.empty:
        print("❌ Failed to fetch data")
        return

    # Compute features
    print("\n" + "=" * 70)
    print("COMPUTING FEATURES")
    print("=" * 70)
    df = compute_all_features(ohlcv_df, funding_df, include_sentiment=False)
    print(f"✅ Features computed: {len(get_feature_columns())} features")

    # Train models
    regime_detector, lstm_ensemble = train_models(df, include_sentiment=False)

    # Save models
    regime_detector.save()
    lstm_ensemble.save_all()

    # Run main backtest
    stats, equity_df, logger = run_backtest(df, regime_detector, lstm_ensemble)

    # Print report
    print_report(stats)

    # Run comparison
    comparison = run_comparison_backtest(df, regime_detector, lstm_ensemble)

    print("\n" + "=" * 70)
    print("📊 SENTIMENT COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<20} {'With Sentiment':<18} {'Without Sentiment':<18}")
    print("-" * 56)

    for metric in ["return_pct", "sharpe", "win_rate", "total_trades", "max_drawdown"]:
        with_val = comparison["with_sentiment"].get(metric, 0)
        without_val = comparison["without_sentiment"].get(metric, 0)
        label = metric.replace("_", " ").title()

        if metric in ["return_pct", "win_rate", "max_drawdown"]:
            print(f"{label:<20} {with_val:>16.2f}% {without_val:>17.2f}%")
        else:
            print(f"{label:<20} {with_val:>17.2f} {without_val:>17.2f}")

    # Save chart
    save_pnl_chart(equity_df, logger, comparison)

    print("\n" + "=" * 70)
    print("✅ BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
