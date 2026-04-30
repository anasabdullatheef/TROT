"""
Backtesting module using vectorbt.
Tests the momentum strategy on historical data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import vectorbt as vbt
import config
from strategy import MomentumStrategy
from exchange import ExchangeClient
from logger import logger


class Backtester:
    """Backtesting engine for the momentum strategy."""

    def __init__(self):
        self.strategy = MomentumStrategy()
        self.exchange = ExchangeClient()
        self.results = None
        self.portfolio = None

    def fetch_data(self, days: int = None) -> pd.DataFrame:
        """Fetch historical data for backtesting."""
        days = days or config.BACKTEST_DAYS
        return self.exchange.fetch_historical_data(
            symbol=config.SYMBOL,
            timeframe=config.TIMEFRAME,
            days=days
        )

    def run_backtest(
        self,
        df: pd.DataFrame = None,
        initial_capital: float = None,
        show_plot: bool = False
    ) -> dict:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data (if None, fetches from exchange)
            initial_capital: Starting capital
            show_plot: Whether to display charts

        Returns:
            Dict with backtest results
        """
        # Fetch data if not provided
        if df is None:
            df = self.fetch_data()

        initial_capital = initial_capital or config.INITIAL_CAPITAL

        logger.log_info(f"Running backtest on {len(df)} candles...")
        logger.log_info(f"Period: {df.index[0]} to {df.index[-1]}")
        logger.log_info(f"Initial capital: ${initial_capital:,.2f}")

        # Generate signals using our strategy
        df_signals = self.strategy.generate_signals(df)

        # Create entry and exit signals
        entries = df_signals['signal'] == 1
        exits = df_signals['signal'] == -1

        # Also exit based on stop-loss and take-profit
        # We'll simulate this with vectorbt

        # Run vectorbt portfolio simulation
        close = df['close']

        # Calculate position sizes (2% risk per trade)
        # For simplicity in backtest, use fixed fraction of capital
        size = initial_capital * config.RISK_PER_TRADE / close

        # Create portfolio
        self.portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            size=size,
            size_type='amount',
            init_cash=initial_capital,
            fees=0.001,  # 0.1% trading fee
            sl_stop=config.STOP_LOSS_PCT,
            tp_stop=config.TAKE_PROFIT_PCT,
            freq='1h'
        )

        # Extract results
        self.results = self._extract_results()

        # Print results
        self._print_results()

        if show_plot:
            self._plot_results()

        return self.results

    def _extract_results(self) -> dict:
        """Extract key metrics from portfolio."""
        stats = self.portfolio.stats()

        # Calculate Sharpe ratio manually if not available
        returns = self.portfolio.returns()
        if len(returns) > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Get trade statistics
        trades = self.portfolio.trades.records_readable
        if len(trades) > 0:
            wins = trades[trades['PnL'] > 0]
            losses = trades[trades['PnL'] <= 0]
            win_rate = len(wins) / len(trades) * 100
            avg_win = wins['PnL'].mean() if len(wins) > 0 else 0
            avg_loss = abs(losses['PnL'].mean()) if len(losses) > 0 else 0
            profit_factor = (wins['PnL'].sum() / abs(losses['PnL'].sum())) if len(losses) > 0 and losses['PnL'].sum() != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            'total_return_pct': float(stats.get('Total Return [%]', 0)),
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': float(stats.get('Max Drawdown [%]', 0)),
            'total_trades': int(stats.get('Total Trades', 0)),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'start_value': float(stats.get('Start Value', 0)),
            'end_value': float(stats.get('End Value', 0)),
            'total_pnl': float(stats.get('End Value', 0)) - float(stats.get('Start Value', 0)),
            'benchmark_return': float(stats.get('Benchmark Return [%]', 0)) if 'Benchmark Return [%]' in stats else 0,
            'exposure_time_pct': float(stats.get('Exposure Time [%]', 0)),
        }

    def _print_results(self):
        """Print backtest results in a formatted way."""
        r = self.results

        print("\n" + "="*60)
        print("                    BACKTEST RESULTS")
        print("="*60)
        print(f"\n{'Metric':<30} {'Value':>25}")
        print("-"*60)

        print(f"{'Initial Capital':<30} ${r['start_value']:>20,.2f}")
        print(f"{'Final Capital':<30} ${r['end_value']:>20,.2f}")
        print(f"{'Total PnL':<30} ${r['total_pnl']:>20,.2f}")
        print(f"{'Total Return':<30} {r['total_return_pct']:>20.2f}%")

        print("-"*60)
        print(f"{'Sharpe Ratio':<30} {r['sharpe_ratio']:>25.2f}")
        print(f"{'Max Drawdown':<30} {r['max_drawdown_pct']:>20.2f}%")
        print(f"{'Profit Factor':<30} {r['profit_factor']:>25.2f}")

        print("-"*60)
        print(f"{'Total Trades':<30} {r['total_trades']:>25}")
        print(f"{'Win Rate':<30} {r['win_rate']:>20.2f}%")
        print(f"{'Avg Win':<30} ${r['avg_win']:>20,.2f}")
        print(f"{'Avg Loss':<30} ${r['avg_loss']:>20,.2f}")
        print(f"{'Exposure Time':<30} {r['exposure_time_pct']:>20.2f}%")

        print("="*60 + "\n")

        # Log summary
        logger.log_success(
            f"Backtest complete: {r['total_trades']} trades | "
            f"Sharpe: {r['sharpe_ratio']:.2f} | "
            f"Win Rate: {r['win_rate']:.1f}% | "
            f"Max DD: {r['max_drawdown_pct']:.1f}%"
        )

    def _plot_results(self):
        """Plot backtest results (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(14, 10))

            # Equity curve
            self.portfolio.value().plot(ax=axes[0], title='Portfolio Value')
            axes[0].set_ylabel('Value ($)')

            # Drawdown
            self.portfolio.drawdown().plot(ax=axes[1], title='Drawdown')
            axes[1].set_ylabel('Drawdown')

            # Cumulative returns
            self.portfolio.cumulative_returns().plot(ax=axes[2], title='Cumulative Returns')
            axes[2].set_ylabel('Returns')

            plt.tight_layout()
            plt.savefig('data/backtest_results.png', dpi=150)
            logger.log_info("Backtest chart saved to data/backtest_results.png")
            plt.show()

        except ImportError:
            logger.log_info("Matplotlib not available for plotting")

    def get_trade_log(self) -> pd.DataFrame:
        """Get detailed trade log."""
        if self.portfolio is None:
            return pd.DataFrame()

        trades = self.portfolio.trades.records_readable
        return trades

    def save_results(self, filename: str = 'data/backtest_results.csv'):
        """Save backtest results to CSV."""
        if self.results:
            pd.DataFrame([self.results]).to_csv(filename, index=False)
            logger.log_info(f"Results saved to {filename}")

        # Save trade log
        trades = self.get_trade_log()
        if len(trades) > 0:
            trades.to_csv('data/backtest_trades.csv', index=False)
            logger.log_info("Trade log saved to data/backtest_trades.csv")


def run_backtest():
    """Main function to run backtest."""
    backtester = Backtester()
    results = backtester.run_backtest(show_plot=False)
    backtester.save_results()
    return results


if __name__ == "__main__":
    run_backtest()
