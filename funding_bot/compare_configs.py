"""
Compare multiple strategy configurations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt

from backtest import fetch_historical_ohlcv, calculate_max_drawdown, calculate_sharpe_ratio
from funding import get_historical_funding_rates
from strategy import calculate_indicators
from risk import RiskManager
from logger import BacktestLogger
import config


# Define the 3 configs to test
CONFIGS = {
    "Config A": {
        "name": "Relaxed Funding",
        "funding_long_threshold": 0.0003,   # 0.03%
        "funding_short_threshold": -0.0001, # -0.01%
        "rsi_overbought": 60,
        "rsi_oversold": 40,
        "adx_threshold": 30,
        "use_rsi": True,
        "use_adx": True,
    },
    "Config B": {
        "name": "Funding Only",
        "funding_long_threshold": 0.0003,   # 0.03%
        "funding_short_threshold": -0.0001, # -0.01%
        "rsi_overbought": None,
        "rsi_oversold": None,
        "adx_threshold": None,
        "use_rsi": False,
        "use_adx": False,
    },
    "Config C": {
        "name": "High Frequency",
        "funding_long_threshold": 0.0001,   # 0.01%
        "funding_short_threshold": -0.00005, # -0.005%
        "rsi_overbought": 55,
        "rsi_oversold": 45,
        "adx_threshold": None,
        "use_rsi": True,
        "use_adx": False,
    },
}


def check_entry_config(
    funding_rate: float,
    rsi: float,
    adx: float,
    cfg: Dict
) -> Optional[str]:
    """
    Check entry conditions based on config
    Returns 'long', 'short', or None
    """
    # Check ADX if required
    if cfg["use_adx"] and cfg["adx_threshold"] is not None:
        if adx >= cfg["adx_threshold"]:
            return None

    # Check for short signal
    if funding_rate >= cfg["funding_long_threshold"]:
        if cfg["use_rsi"]:
            if rsi >= cfg["rsi_overbought"]:
                return "short"
        else:
            return "short"

    # Check for long signal
    if funding_rate <= cfg["funding_short_threshold"]:
        if cfg["use_rsi"]:
            if rsi <= cfg["rsi_oversold"]:
                return "long"
        else:
            return "long"

    return None


def check_exit_config(
    entry_price: float,
    current_price: float,
    position_type: str,
    funding_rate: float
) -> Tuple[bool, str, float]:
    """
    Check exit conditions (same for all configs)
    Returns (should_exit, reason, pnl_pct)
    """
    if position_type == "long":
        pnl_pct = (current_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - current_price) / entry_price

    # Take profit
    if pnl_pct >= config.TAKE_PROFIT_PCT:
        return True, "Take Profit", pnl_pct

    # Stop loss
    if pnl_pct <= -config.STOP_LOSS_PCT:
        return True, "Stop Loss", pnl_pct

    # Funding normalized
    if -0.0001 <= funding_rate <= 0.0001:
        return True, "Funding Normalized", pnl_pct

    return False, "", pnl_pct


def run_backtest_config(
    df: pd.DataFrame,
    cfg: Dict,
    initial_capital: float = 10000,
    verbose: bool = False
) -> Tuple[Dict, pd.DataFrame, list]:
    """
    Run backtest with specific config
    """
    risk_mgr = RiskManager(initial_capital)
    trades = []
    equity_curve = []
    position = None

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        price = row["close"]
        funding_rate = row["funding_rate"]
        rsi = row["rsi"]
        adx = row["adx"]

        # Skip invalid rows
        if pd.isna(funding_rate) or pd.isna(rsi) or pd.isna(adx) or adx < 1:
            continue

        # Check exit if in position
        if position is not None:
            should_exit, reason, pnl_pct = check_exit_config(
                position["entry_price"], price, position["type"], funding_rate
            )

            if should_exit:
                if position["type"] == "long":
                    pnl = (price - position["entry_price"]) * position["size"]
                else:
                    pnl = (position["entry_price"] - price) * position["size"]

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": ts,
                    "direction": position["type"],
                    "entry_price": position["entry_price"],
                    "exit_price": price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": reason,
                    "win": pnl > 0
                })

                risk_mgr.current_capital += pnl
                position = None

                if verbose:
                    emoji = "🟢" if pnl > 0 else "🔴"
                    print(f"{emoji} {ts}: Close @ ${price:,.0f} | PnL: ${pnl:.2f} ({pnl_pct*100:+.2f}%)")

        # Check entry if no position
        if position is None:
            signal = check_entry_config(funding_rate, rsi, adx, cfg)

            if signal:
                sizing = risk_mgr.calculate_position_size(price)
                position = {
                    "type": signal,
                    "entry_price": price,
                    "entry_time": ts,
                    "size": sizing["position_size"]
                }

                if verbose:
                    print(f"🔵 {ts}: Open {signal.upper()} @ ${price:,.0f} | "
                          f"Funding: {funding_rate*100:.4f}% | RSI: {rsi:.1f}")

        # Track equity
        unrealized = 0
        if position:
            if position["type"] == "long":
                unrealized = (price - position["entry_price"]) * position["size"]
            else:
                unrealized = (position["entry_price"] - price) * position["size"]

        equity_curve.append({
            "timestamp": ts,
            "capital": risk_mgr.current_capital + unrealized
        })

    # Close any remaining position
    if position is not None:
        last_price = df.iloc[-1]["close"]
        if position["type"] == "long":
            pnl = (last_price - position["entry_price"]) * position["size"]
            pnl_pct = (last_price - position["entry_price"]) / position["entry_price"]
        else:
            pnl = (position["entry_price"] - last_price) * position["size"]
            pnl_pct = (position["entry_price"] - last_price) / position["entry_price"]

        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": df.iloc[-1]["timestamp"],
            "direction": position["type"],
            "entry_price": position["entry_price"],
            "exit_price": last_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": "End of Backtest",
            "win": pnl > 0
        })
        risk_mgr.current_capital += pnl

    # Calculate stats
    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    stats = calculate_stats(trades_df, equity_df, initial_capital, risk_mgr.current_capital)

    return stats, equity_df, trades


def calculate_stats(trades_df: pd.DataFrame, equity_df: pd.DataFrame,
                   initial_capital: float, final_capital: float) -> Dict:
    """Calculate comprehensive statistics"""

    if trades_df.empty:
        return {
            "total_return": 0,
            "sharpe": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "final_capital": initial_capital,
            "yearly": {}
        }

    total_trades = len(trades_df)
    winning = trades_df[trades_df["win"]]
    losing = trades_df[~trades_df["win"]]

    total_wins = winning["pnl"].sum() if len(winning) > 0 else 0
    total_losses = abs(losing["pnl"].sum()) if len(losing) > 0 else 0

    # Max drawdown
    if not equity_df.empty:
        running_max = equity_df["capital"].cummax()
        drawdown = (equity_df["capital"] - running_max) / running_max
        max_dd = abs(drawdown.min())

        # Sharpe
        equity_df["returns"] = equity_df["capital"].pct_change().fillna(0)
        sharpe = calculate_sharpe_ratio(equity_df["returns"])
    else:
        max_dd = 0
        sharpe = 0

    # Year by year
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

    return {
        "total_return": (final_capital - initial_capital) / initial_capital * 100,
        "sharpe": sharpe,
        "win_rate": len(winning) / total_trades * 100 if total_trades > 0 else 0,
        "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
        "max_drawdown": max_dd * 100,
        "total_trades": total_trades,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "final_capital": final_capital,
        "yearly": yearly
    }


def main():
    print("=" * 80)
    print("🔬 STRATEGY CONFIGURATION COMPARISON")
    print("=" * 80)

    # Fetch data once
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    print(f"\nBacktest period: {start_date.date()} to {end_date.date()}")
    print(f"Initial capital: $10,000")

    print("\n" + "-" * 80)
    print("Fetching data...")

    ohlcv_df = fetch_historical_ohlcv(start_date=start_date, end_date=end_date)
    print(f"✅ OHLCV: {len(ohlcv_df)} candles")

    funding_df = get_historical_funding_rates(start_time=start_date, end_time=end_date)
    print(f"✅ Funding: {len(funding_df)} records")

    # Prepare data
    ohlcv_df = calculate_indicators(ohlcv_df)

    df = pd.merge_asof(
        ohlcv_df.sort_values("timestamp"),
        funding_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("8h")
    )
    df["funding_rate"] = df["funding_rate"].ffill()

    # Run each config
    results = {}
    all_equity = {}

    for config_name, cfg in CONFIGS.items():
        print(f"\n" + "=" * 80)
        print(f"Running {config_name}: {cfg['name']}")
        print("-" * 80)

        desc = []
        desc.append(f"Funding: >{cfg['funding_long_threshold']*100:.2f}% (short), <{cfg['funding_short_threshold']*100:.3f}% (long)")
        if cfg["use_rsi"]:
            desc.append(f"RSI: >{cfg['rsi_overbought']} (short), <{cfg['rsi_oversold']} (long)")
        else:
            desc.append("RSI: disabled")
        if cfg["use_adx"]:
            desc.append(f"ADX: <{cfg['adx_threshold']}")
        else:
            desc.append("ADX: disabled")
        print("  " + " | ".join(desc))
        print("-" * 80)

        stats, equity_df, trades = run_backtest_config(df, cfg, verbose=True)
        results[config_name] = stats
        all_equity[config_name] = equity_df

        print(f"\n  Trades: {stats['total_trades']} | Return: {stats['total_return']:+.2f}% | "
              f"Win Rate: {stats['win_rate']:.1f}% | Sharpe: {stats['sharpe']:.2f}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("📊 COMPARISON TABLE")
    print("=" * 80)

    # Header
    print(f"\n{'Metric':<25} {'Config A':<18} {'Config B':<18} {'Config C':<18}")
    print(f"{'':25} {'(Relaxed)':<18} {'(Funding Only)':<18} {'(High Freq)':<18}")
    print("-" * 80)

    # Metrics
    metrics = [
        ("Total Return", "total_return", "{:+.2f}%"),
        ("Sharpe Ratio", "sharpe", "{:.2f}"),
        ("Win Rate", "win_rate", "{:.1f}%"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Max Drawdown", "max_drawdown", "{:.2f}%"),
        ("Total Trades", "total_trades", "{:.0f}"),
        ("Winning Trades", "winning_trades", "{:.0f}"),
        ("Losing Trades", "losing_trades", "{:.0f}"),
        ("Final Capital", "final_capital", "${:,.2f}"),
    ]

    for label, key, fmt in metrics:
        row = f"{label:<25}"
        for cfg_name in ["Config A", "Config B", "Config C"]:
            val = results[cfg_name][key]
            if key == "profit_factor" and val == float("inf"):
                row += f"{'∞':<18}"
            else:
                row += f"{fmt.format(val):<18}"
        print(row)

    # Year by year breakdown
    print("\n" + "=" * 80)
    print("📅 YEAR-BY-YEAR BREAKDOWN")
    print("=" * 80)

    all_years = set()
    for cfg_name in results:
        all_years.update(results[cfg_name]["yearly"].keys())
    all_years = sorted(all_years)

    for cfg_name in ["Config A", "Config B", "Config C"]:
        print(f"\n{cfg_name} ({CONFIGS[cfg_name]['name']}):")
        print(f"  {'Year':<8} {'Trades':<10} {'PnL':<15} {'Win Rate':<10}")
        print("  " + "-" * 45)

        yearly = results[cfg_name]["yearly"]
        for year in all_years:
            if year in yearly:
                y = yearly[year]
                print(f"  {year:<8} {y['trades']:<10} ${y['pnl']:>12,.2f}  {y['win_rate']:>8.1f}%")
            else:
                print(f"  {year:<8} {'0':<10} {'$0.00':>15}  {'-':>8}")

    # Statistical significance note
    print("\n" + "=" * 80)
    print("📈 STATISTICAL SIGNIFICANCE")
    print("=" * 80)
    for cfg_name in ["Config A", "Config B", "Config C"]:
        trades = results[cfg_name]["total_trades"]
        status = "✅ Sufficient" if trades >= 50 else "⚠️  Insufficient"
        print(f"{cfg_name}: {trades} trades - {status} (need ≥50)")

    # Save comparison chart
    save_comparison_chart(all_equity, results)

    print("\n" + "=" * 80)
    print("✅ Comparison complete! Chart saved to data/config_comparison.png")
    print("=" * 80)


def save_comparison_chart(all_equity: Dict, results: Dict):
    """Save comparison equity curves"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = {"Config A": "#00B5AD", "Config B": "#E0592A", "Config C": "#F0AB00"}

    # Equity curves
    ax1 = axes[0]
    for cfg_name, equity_df in all_equity.items():
        if not equity_df.empty:
            ax1.plot(equity_df["timestamp"], equity_df["capital"],
                    label=f"{cfg_name} ({results[cfg_name]['total_return']:+.1f}%)",
                    color=colors[cfg_name], linewidth=1.5)

    ax1.axhline(y=10000, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    ax1.set_title("Equity Curves Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Capital (USDT)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Bar chart comparison
    ax2 = axes[1]
    x = np.arange(4)
    width = 0.25

    metrics_to_plot = [
        ("Total Return %", [results[c]["total_return"] for c in ["Config A", "Config B", "Config C"]]),
        ("Win Rate %", [results[c]["win_rate"] for c in ["Config A", "Config B", "Config C"]]),
        ("Sharpe × 10", [results[c]["sharpe"] * 10 for c in ["Config A", "Config B", "Config C"]]),
        ("Max DD %", [results[c]["max_drawdown"] for c in ["Config A", "Config B", "Config C"]]),
    ]

    for i, cfg_name in enumerate(["Config A", "Config B", "Config C"]):
        values = [m[1][i] for m in metrics_to_plot]
        ax2.bar(x + i * width, values, width, label=cfg_name, color=colors[cfg_name])

    ax2.set_ylabel("Value")
    ax2.set_title("Key Metrics Comparison", fontsize=14, fontweight="bold")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([m[0] for m in metrics_to_plot])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("data/config_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
