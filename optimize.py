"""
Strategy optimization script.
Tests multiple configurations and compares results.
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
import ta


def calculate_indicators(df, ema_period=20, rsi_period=14, volume_ma_period=20,
                         use_trend_filter=False, fast_ema=50, slow_ema=200):
    """Calculate technical indicators with optional trend filter."""
    df = df.copy()

    # EMA
    df['ema'] = ta.trend.ema_indicator(df['close'], window=ema_period)

    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=rsi_period)

    # Volume MA
    df['volume_ma'] = df['volume'].rolling(window=volume_ma_period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Price position relative to EMA
    df['above_ema'] = df['close'] > df['ema']
    prev_above = df['above_ema'].shift(1)
    df['prev_above_ema'] = prev_above.where(prev_above.notna(), False)

    # EMA crossover detection
    df['ema_crossover_up'] = (df['above_ema']) & (~df['prev_above_ema'].astype(bool))

    # Trend filter (50/200 EMA)
    if use_trend_filter:
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=fast_ema)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=slow_ema)
        df['trend_bullish'] = df['ema_fast'] > df['ema_slow']
    else:
        df['trend_bullish'] = True

    return df


def generate_signals(df, rsi_threshold=50, volume_multiplier=1.0):
    """Generate buy/sell signals."""
    # Buy conditions
    buy_condition = (
        (df['above_ema']) &
        (df['rsi'] > rsi_threshold) &
        (df['volume_ratio'] >= volume_multiplier) &
        (df['ema_crossover_up']) &
        (df['trend_bullish'])
    )

    # Sell conditions (price drops below EMA)
    sell_condition = ~df['above_ema'].astype(bool) & df['prev_above_ema'].astype(bool)

    return buy_condition, sell_condition


def run_backtest(df, stop_loss_pct, take_profit_pct, rsi_threshold,
                 use_trend_filter=False, initial_capital=10000, risk_per_trade=0.02):
    """Run backtest with given parameters."""

    # Calculate indicators
    df = calculate_indicators(df, use_trend_filter=use_trend_filter)

    # Generate signals
    entries, exits = generate_signals(df, rsi_threshold=rsi_threshold)

    # Calculate position sizes
    close = df['close']
    size = initial_capital * risk_per_trade / close

    # Run portfolio simulation
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        size=size,
        size_type='amount',
        init_cash=initial_capital,
        fees=0.001,
        sl_stop=stop_loss_pct,
        tp_stop=take_profit_pct,
        freq='1h'
    )

    # Extract metrics
    stats = portfolio.stats()
    returns = portfolio.returns()
    sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0

    trades = portfolio.trades.records_readable
    if len(trades) > 0:
        wins = trades[trades['PnL'] > 0]
        losses = trades[trades['PnL'] <= 0]
        win_rate = len(wins) / len(trades) * 100
        profit_factor = (wins['PnL'].sum() / abs(losses['PnL'].sum())) if len(losses) > 0 and losses['PnL'].sum() != 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0

    return {
        'total_return_pct': float(stats.get('Total Return [%]', 0)),
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': float(stats.get('Max Drawdown [%]', 0)),
        'total_trades': int(stats.get('Total Trades', 0)),
        'start_value': float(stats.get('Start Value', 0)),
        'end_value': float(stats.get('End Value', 0)),
    }


def main():
    print("=" * 80)
    print("STRATEGY OPTIMIZATION - Testing 3 Configurations")
    print("=" * 80)
    print()

    # Load data
    df_1h = pd.read_csv('data/btc_usdt_6m.csv', index_col='timestamp', parse_dates=True)
    df_4h = pd.read_csv('data/btc_usdt_4h_6m.csv', index_col='timestamp', parse_dates=True)

    print(f"1h Data: {len(df_1h)} candles ({df_1h.index[0].date()} to {df_1h.index[-1].date()})")
    print(f"4h Data: {len(df_4h)} candles ({df_4h.index[0].date()} to {df_4h.index[-1].date()})")
    print()

    # Configuration definitions
    configs = [
        {
            'name': 'Config 1',
            'description': 'SL 2.5%, TP 5%, RSI > 60, 1h',
            'df': df_1h,
            'stop_loss': 0.025,
            'take_profit': 0.05,
            'rsi_threshold': 60,
            'use_trend_filter': False,
        },
        {
            'name': 'Config 2',
            'description': 'SL 3%, TP 6%, RSI > 55, 50/200 EMA filter, 1h',
            'df': df_1h,
            'stop_loss': 0.03,
            'take_profit': 0.06,
            'rsi_threshold': 55,
            'use_trend_filter': True,
        },
        {
            'name': 'Config 3',
            'description': 'SL 3%, TP 7%, RSI > 55, 4h timeframe',
            'df': df_4h,
            'stop_loss': 0.03,
            'take_profit': 0.07,
            'rsi_threshold': 55,
            'use_trend_filter': False,
        },
    ]

    # Also run baseline for comparison
    baseline = {
        'name': 'Baseline',
        'description': 'Original: SL 1.5%, TP 3%, RSI > 50, 1h',
        'df': df_1h,
        'stop_loss': 0.015,
        'take_profit': 0.03,
        'rsi_threshold': 50,
        'use_trend_filter': False,
    }
    configs.insert(0, baseline)

    # Run backtests
    results = []
    for config in configs:
        print(f"Testing {config['name']}: {config['description']}...")
        result = run_backtest(
            df=config['df'],
            stop_loss_pct=config['stop_loss'],
            take_profit_pct=config['take_profit'],
            rsi_threshold=config['rsi_threshold'],
            use_trend_filter=config['use_trend_filter'],
        )
        result['name'] = config['name']
        result['description'] = config['description']
        results.append(result)

    print()
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()

    # Print results table
    print(f"{'Config':<12} {'Description':<45} {'Return':>8} {'Sharpe':>8} {'Win %':>7} {'PF':>6} {'MaxDD':>7} {'Trades':>7}")
    print("-" * 105)

    for r in results:
        print(f"{r['name']:<12} {r['description']:<45} {r['total_return_pct']:>7.2f}% {r['sharpe_ratio']:>8.2f} {r['win_rate']:>6.1f}% {r['profit_factor']:>6.2f} {r['max_drawdown_pct']:>6.2f}% {r['total_trades']:>7}")

    print()
    print("=" * 80)

    # Find best config
    best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
    best_return = max(results, key=lambda x: x['total_return_pct'])
    best_pf = max(results, key=lambda x: x['profit_factor'] if x['profit_factor'] != float('inf') else 0)

    print()
    print("BEST PERFORMERS:")
    print(f"  Highest Sharpe Ratio: {best_sharpe['name']} ({best_sharpe['sharpe_ratio']:.2f})")
    print(f"  Highest Return:       {best_return['name']} ({best_return['total_return_pct']:.2f}%)")
    print(f"  Highest Profit Factor: {best_pf['name']} ({best_pf['profit_factor']:.2f})")
    print()

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('data/optimization_results.csv', index=False)
    print("Results saved to data/optimization_results.csv")

    return results


if __name__ == "__main__":
    main()
