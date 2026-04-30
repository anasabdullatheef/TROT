"""
Backtesting Module
Fetches historical data and runs strategy backtest
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

import config
from funding import get_historical_funding_rates, classify_funding_signal
from strategy import calculate_indicators, check_entry_conditions, check_exit_conditions
from risk import RiskManager, calculate_max_drawdown, calculate_sharpe_ratio
from logger import BacktestLogger


def fetch_historical_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_date: datetime = None,
    end_date: datetime = None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance Futures public API

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Candle interval (e.g., "1h")
        start_date: Start datetime
        end_date: End datetime

    Returns:
        DataFrame with OHLCV data
    """
    url = f"{config.BINANCE_BASE_URL}/fapi/v1/klines"

    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=config.LOOKBACK_DAYS)

    all_candles = []
    current_start = start_date

    print(f"Fetching OHLCV data from {start_date} to {end_date}...")

    while current_start < end_date:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(current_start.timestamp() * 1000),
            "endTime": int(end_date.timestamp() * 1000),
            "limit": 1500  # Binance max is 1500
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_candles.extend(data)

            # Move start to after the last candle we received
            last_ts = data[-1][0]
            current_start = datetime.fromtimestamp(last_ts / 1000) + timedelta(hours=1)

            print(f"  Fetched {len(data)} candles, total: {len(all_candles)}")

            time.sleep(config.RATE_LIMIT_DELAY)

        except Exception as e:
            print(f"Error fetching OHLCV: {e}")
            break

    if not all_candles:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def run_backtest(
    ohlcv_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    initial_capital: float = None,
    verbose: bool = True
) -> Tuple[Dict, pd.DataFrame, BacktestLogger]:
    """
    Run the funding rate strategy backtest

    Args:
        ohlcv_df: OHLCV DataFrame with indicators calculated
        funding_df: Funding rate DataFrame
        initial_capital: Starting capital
        verbose: Print trade details

    Returns:
        Tuple of (stats dict, equity DataFrame, BacktestLogger)
    """
    if initial_capital is None:
        initial_capital = config.INITIAL_CAPITAL

    # Initialize
    risk_mgr = RiskManager(initial_capital)
    logger = BacktestLogger()

    # Merge funding rates with OHLCV
    df = ohlcv_df.copy()
    funding_df = funding_df.copy()

    # Calculate indicators if not already done
    if "rsi" not in df.columns:
        df = calculate_indicators(df)

    # Merge funding rates (forward fill)
    df = pd.merge_asof(
        df.sort_values("timestamp"),
        funding_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("8h")
    )
    df["funding_rate"] = df["funding_rate"].ffill()

    # Track position state
    position: Optional[Dict] = None
    equity_curve = []

    # Process each candle
    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        price = row["close"]
        funding_rate = row["funding_rate"]
        rsi = row["rsi"]
        adx = row["adx"]

        # Skip if we don't have funding data or indicators aren't ready
        # ADX < 1 indicates warmup period (should be NaN but check anyway)
        if pd.isna(funding_rate) or pd.isna(rsi) or pd.isna(adx) or adx < 1:
            continue

        # Check exit conditions if we have a position
        if position is not None:
            exit_check = check_exit_conditions(
                entry_price=position["entry_price"],
                current_price=price,
                position_type=position["type"],
                funding_rate=funding_rate
            )

            if exit_check["exit"]:
                # Close position
                trade = risk_mgr.close_position(
                    exit_price=price,
                    exit_reason=exit_check["reason"],
                    timestamp=ts
                )

                if trade and verbose:
                    emoji = "🟢" if trade["pnl"] > 0 else "🔴"
                    print(f"{emoji} {ts}: Close {trade['type'].upper()} @ ${price:,.0f} | "
                          f"PnL: ${trade['pnl']:.2f} ({trade['pnl_pct']*100:+.2f}%) | "
                          f"Reason: {exit_check['reason'][:50]}")

                logger.log_trade(
                    entry_time=position["entry_time"],
                    exit_time=ts,
                    direction=position["type"],
                    entry_price=position["entry_price"],
                    exit_price=price,
                    size=position["size"],
                    pnl=trade["pnl"],
                    pnl_pct=trade["pnl_pct"],
                    exit_reason=exit_check["reason"],
                    funding_rate_entry=position["funding_rate"],
                    rsi_entry=position["rsi"],
                    adx_entry=position["adx"]
                )

                position = None

        # Check entry conditions if we don't have a position
        if position is None:
            # Check if we can open a new position
            can_open = risk_mgr.can_open_position()

            if can_open["allowed"]:
                entry_check = check_entry_conditions(
                    funding_rate=funding_rate,
                    rsi=rsi,
                    adx=adx
                )

                if entry_check["signal"]:
                    # Calculate position size
                    sizing = risk_mgr.calculate_position_size(price)

                    # Open position
                    pos_data = risk_mgr.open_position(
                        position_type=entry_check["signal"],
                        entry_price=price,
                        size=sizing["position_size"],
                        timestamp=ts
                    )

                    position = {
                        "type": entry_check["signal"],
                        "entry_price": price,
                        "entry_time": ts,
                        "size": sizing["position_size"],
                        "funding_rate": funding_rate,
                        "rsi": rsi,
                        "adx": adx
                    }

                    logger.log_signal(
                        timestamp=ts,
                        direction=entry_check["signal"],
                        price=price,
                        funding_rate=funding_rate,
                        rsi=rsi,
                        adx=adx,
                        executed=True
                    )

                    if verbose:
                        print(f"🔵 {ts}: Open {entry_check['signal'].upper()} @ ${price:,.0f} | "
                              f"Funding: {funding_rate*100:.4f}% | RSI: {rsi:.1f} | ADX: {adx:.1f}")

        # Record equity
        unrealized_pnl = 0
        if position is not None:
            if position["type"] == "long":
                unrealized_pnl = (price - position["entry_price"]) * position["size"]
            else:
                unrealized_pnl = (position["entry_price"] - price) * position["size"]

        equity = risk_mgr.current_capital + unrealized_pnl
        equity_curve.append({
            "timestamp": ts,
            "capital": equity
        })

    # Close any remaining position at the end
    if position is not None:
        last_price = df.iloc[-1]["close"]
        trade = risk_mgr.close_position(
            exit_price=last_price,
            exit_reason="Backtest end",
            timestamp=df.iloc[-1]["timestamp"]
        )

    # Create equity DataFrame
    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df["drawdown"] = (equity_df["capital"].cummax() - equity_df["capital"]) / equity_df["capital"].cummax()

    # Calculate statistics
    stats = risk_mgr.get_statistics()

    # Add more stats
    if not equity_df.empty:
        stats["max_drawdown"] = calculate_max_drawdown(equity_df["capital"])

        # Calculate returns for Sharpe
        equity_df["returns"] = equity_df["capital"].pct_change().fillna(0)
        stats["sharpe_ratio"] = calculate_sharpe_ratio(equity_df["returns"])
    else:
        stats["max_drawdown"] = 0
        stats["sharpe_ratio"] = 0

    return stats, equity_df, logger


def analyze_results(
    logger: BacktestLogger,
    equity_df: pd.DataFrame,
    save_chart: bool = True
) -> Dict:
    """
    Analyze backtest results and generate reports

    Returns extended statistics including year-by-year and month-by-month breakdown
    """
    trades_df = logger.get_trades_df()

    if trades_df.empty:
        return {
            "total_trades": 0,
            "yearly_breakdown": {},
            "monthly_breakdown": {}
        }

    # Overall stats
    total_trades = len(trades_df)
    winning = trades_df[trades_df["win"]]
    losing = trades_df[~trades_df["win"]]

    results = {
        "total_trades": total_trades,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": len(winning) / total_trades * 100 if total_trades > 0 else 0,
        "total_pnl": trades_df["pnl"].sum(),
        "avg_pnl_per_trade": trades_df["pnl"].mean(),
        "avg_win": winning["pnl"].mean() if len(winning) > 0 else 0,
        "avg_loss": losing["pnl"].mean() if len(losing) > 0 else 0,
        "profit_factor": winning["pnl"].sum() / abs(losing["pnl"].sum()) if len(losing) > 0 and losing["pnl"].sum() != 0 else float("inf"),
        "max_drawdown": calculate_max_drawdown(equity_df["capital"]) if not equity_df.empty else 0,
        "sharpe_ratio": calculate_sharpe_ratio(equity_df["returns"]) if "returns" in equity_df.columns else 0
    }

    # Year by year breakdown
    trades_df["year"] = pd.to_datetime(trades_df["exit_time"]).dt.year
    yearly = trades_df.groupby("year").agg({
        "pnl": ["sum", "count"],
        "win": "sum"
    }).round(2)
    yearly.columns = ["pnl", "trades", "wins"]
    yearly["win_rate"] = (yearly["wins"] / yearly["trades"] * 100).round(1)

    results["yearly_breakdown"] = yearly.to_dict("index")

    # Monthly breakdown
    trades_df["month"] = pd.to_datetime(trades_df["exit_time"]).dt.to_period("M")
    monthly = trades_df.groupby("month").agg({
        "pnl": "sum"
    }).round(2)

    # Find best and worst months
    if not monthly.empty:
        best_month_idx = monthly["pnl"].idxmax()
        worst_month_idx = monthly["pnl"].idxmin()

        results["best_month"] = {
            "month": str(best_month_idx),
            "pnl": monthly.loc[best_month_idx, "pnl"]
        }
        results["worst_month"] = {
            "month": str(worst_month_idx),
            "pnl": monthly.loc[worst_month_idx, "pnl"]
        }

    # Exit reason breakdown
    exit_reasons = trades_df["exit_reason"].apply(
        lambda x: "Take Profit" if "Take profit" in x
        else "Stop Loss" if "Stop loss" in x
        else "Funding Normalized" if "normalized" in x
        else "Other"
    )
    results["exit_breakdown"] = exit_reasons.value_counts().to_dict()

    # Direction breakdown
    direction_stats = trades_df.groupby("direction").agg({
        "pnl": ["sum", "count", "mean"],
        "win": "sum"
    }).round(2)
    direction_stats.columns = ["total_pnl", "trades", "avg_pnl", "wins"]
    results["direction_breakdown"] = direction_stats.to_dict("index")

    # Save chart
    if save_chart and not equity_df.empty:
        save_pnl_chart(equity_df, trades_df)

    return results


def save_pnl_chart(equity_df: pd.DataFrame, trades_df: pd.DataFrame):
    """Save cumulative PnL chart to file"""
    os.makedirs("data", exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 1. Equity curve
    ax1 = axes[0]
    ax1.plot(equity_df["timestamp"], equity_df["capital"], color="#00B5AD", linewidth=1)
    ax1.fill_between(equity_df["timestamp"], config.INITIAL_CAPITAL, equity_df["capital"],
                     where=equity_df["capital"] >= config.INITIAL_CAPITAL,
                     alpha=0.3, color="#00B5AD")
    ax1.fill_between(equity_df["timestamp"], config.INITIAL_CAPITAL, equity_df["capital"],
                     where=equity_df["capital"] < config.INITIAL_CAPITAL,
                     alpha=0.3, color="#E0592A")
    ax1.axhline(y=config.INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Equity Curve", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Capital (USDT)")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # 2. Drawdown
    ax2 = axes[1]
    ax2.fill_between(equity_df["timestamp"], 0, -equity_df["drawdown"] * 100,
                     color="#E0592A", alpha=0.5)
    ax2.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # 3. Monthly returns
    ax3 = axes[2]
    if not trades_df.empty:
        trades_df["month"] = pd.to_datetime(trades_df["exit_time"]).dt.to_period("M")
        monthly_pnl = trades_df.groupby("month")["pnl"].sum()
        colors = ["#00B5AD" if x >= 0 else "#E0592A" for x in monthly_pnl.values]
        monthly_dates = [pd.Timestamp(str(m)) for m in monthly_pnl.index]
        ax3.bar(monthly_dates, monthly_pnl.values, color=colors, width=20, alpha=0.7)
        ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax3.set_title("Monthly PnL", fontsize=14, fontweight="bold")
        ax3.set_ylabel("PnL (USDT)")
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    plt.savefig(config.PNL_CHART_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Chart saved to {config.PNL_CHART_FILE}")


def print_backtest_report(stats: Dict, results: Dict):
    """Print formatted backtest report"""
    print("\n" + "=" * 70)
    print("🤖 FUNDING RATE ARBITRAGE BOT - BACKTEST RESULTS")
    print("=" * 70)

    print(f"\n📊 OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"Initial Capital:      ${config.INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:        ${stats.get('current_capital', 0):,.2f}")
    print(f"Total Return:         {stats.get('return_pct', 0):+.2f}%")
    print(f"Total PnL:            ${stats.get('total_pnl', 0):,.2f}")
    print(f"Sharpe Ratio:         {stats.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown:         {stats.get('max_drawdown', 0)*100:.2f}%")

    print(f"\n📈 TRADE STATISTICS")
    print("-" * 40)
    print(f"Total Trades:         {stats.get('total_trades', 0)}")
    print(f"Winning Trades:       {stats.get('winning_trades', 0)}")
    print(f"Losing Trades:        {stats.get('losing_trades', 0)}")
    print(f"Win Rate:             {stats.get('win_rate', 0):.1f}%")
    print(f"Profit Factor:        {stats.get('profit_factor', 0):.2f}")
    print(f"Avg Win:              ${results.get('avg_win', 0):.2f}")
    print(f"Avg Loss:             ${results.get('avg_loss', 0):.2f}")

    # Exit breakdown
    if "exit_breakdown" in results:
        print(f"\n🚪 EXIT REASONS")
        print("-" * 40)
        for reason, count in results["exit_breakdown"].items():
            print(f"{reason:20s}: {count}")

    # Direction breakdown
    if "direction_breakdown" in results:
        print(f"\n📍 DIRECTION BREAKDOWN")
        print("-" * 40)
        for direction, data in results["direction_breakdown"].items():
            print(f"{direction.upper():6s}: {data['trades']:.0f} trades, "
                  f"${data['total_pnl']:.2f} PnL, "
                  f"{data['wins']/data['trades']*100:.1f}% win rate")

    # Year by year
    if "yearly_breakdown" in results and results["yearly_breakdown"]:
        print(f"\n📅 YEAR-BY-YEAR BREAKDOWN")
        print("-" * 40)
        print(f"{'Year':6s} {'Trades':>8s} {'PnL':>12s} {'Win Rate':>10s}")
        print("-" * 40)
        for year, data in sorted(results["yearly_breakdown"].items()):
            print(f"{year:<6} {data['trades']:>8.0f} ${data['pnl']:>10,.2f} {data['win_rate']:>9.1f}%")

    # Best/worst months
    if "best_month" in results:
        print(f"\n🏆 BEST MONTH:  {results['best_month']['month']} (${results['best_month']['pnl']:,.2f})")
    if "worst_month" in results:
        print(f"💀 WORST MONTH: {results['worst_month']['month']} (${results['worst_month']['pnl']:,.2f})")

    print("\n" + "=" * 70)


def main():
    """Main backtest execution"""
    print("🚀 Starting Funding Rate Arbitrage Bot Backtest")
    print("=" * 60)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.LOOKBACK_DAYS)

    print(f"\nBacktest period: {start_date.date()} to {end_date.date()}")
    print(f"Initial capital: ${config.INITIAL_CAPITAL:,.2f}")
    print(f"Risk per trade: {config.RISK_PER_TRADE*100}%")
    print(f"Take profit: {config.TAKE_PROFIT_PCT*100}%")
    print(f"Stop loss: {config.STOP_LOSS_PCT*100}%")

    # Fetch OHLCV data
    print("\n" + "-" * 60)
    ohlcv_df = fetch_historical_ohlcv(
        symbol=config.BINANCE_SYMBOL,
        interval=config.CANDLE_INTERVAL,
        start_date=start_date,
        end_date=end_date
    )

    if ohlcv_df.empty:
        print("❌ Failed to fetch OHLCV data")
        return

    print(f"✅ Fetched {len(ohlcv_df)} candles")
    print(f"   Date range: {ohlcv_df['timestamp'].min()} to {ohlcv_df['timestamp'].max()}")

    # Fetch funding rates
    print("\n" + "-" * 60)
    funding_df = get_historical_funding_rates(
        start_time=start_date,
        end_time=end_date
    )

    if funding_df.empty:
        print("❌ Failed to fetch funding rate data")
        return

    print(f"✅ Fetched {len(funding_df)} funding rate records")
    print(f"   Date range: {funding_df['timestamp'].min()} to {funding_df['timestamp'].max()}")

    # Calculate indicators
    print("\n" + "-" * 60)
    print("Calculating indicators...")
    ohlcv_df = calculate_indicators(ohlcv_df)
    print("✅ Indicators calculated (RSI, ADX)")

    # Run backtest
    print("\n" + "-" * 60)
    print("Running backtest...")
    print("-" * 60)

    stats, equity_df, logger = run_backtest(
        ohlcv_df=ohlcv_df,
        funding_df=funding_df,
        verbose=True
    )

    # Analyze results
    results = analyze_results(logger, equity_df, save_chart=True)

    # Print report
    print_backtest_report(stats, results)


if __name__ == "__main__":
    main()
