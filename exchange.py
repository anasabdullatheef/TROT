"""
Exchange wrapper module using ccxt.
Handles all Binance API interactions.
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import config
from logger import logger


class ExchangeClient:
    """Wrapper for Binance exchange using ccxt."""

    def __init__(self):
        self.mode = config.TRADING_MODE
        self._initialize_exchange()
        self.positions: Dict[str, dict] = {}  # Track open positions

    def _initialize_exchange(self):
        """Initialize the ccxt exchange client."""
        exchange_config = {
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        }

        if self.mode == 'testnet':
            # Use Binance testnet
            self.exchange = ccxt.binance(exchange_config)
            self.exchange.set_sandbox_mode(True)
            logger.log_info("Initialized Binance TESTNET connection")
        else:
            self.exchange = ccxt.binance(exchange_config)
            logger.log_info("Initialized Binance LIVE connection")

    def fetch_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = None,
        limit: int = 500,
        since: int = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candle data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h')
            limit: Number of candles to fetch
            since: Start timestamp in milliseconds

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        symbol = symbol or config.SYMBOL
        timeframe = timeframe or config.TIMEFRAME

        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.log_error("Failed to fetch OHLCV", str(e))
            raise

    def fetch_historical_data(
        self,
        symbol: str = None,
        timeframe: str = None,
        days: int = 180
    ) -> pd.DataFrame:
        """
        Fetch historical data for backtesting.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days of history

        Returns:
            DataFrame with historical OHLCV data
        """
        symbol = symbol or config.SYMBOL
        timeframe = timeframe or config.TIMEFRAME

        # Calculate start timestamp
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        all_data = []
        current_since = since

        logger.log_info(f"Fetching {days} days of {symbol} {timeframe} data...")

        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    limit=1000,
                    since=current_since
                )

                if not ohlcv:
                    break

                all_data.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1

                # Check if we've reached current time
                if ohlcv[-1][0] >= datetime.now().timestamp() * 1000:
                    break

            except Exception as e:
                logger.log_error("Failed to fetch historical data", str(e))
                break

        if not all_data:
            raise Exception("No historical data retrieved")

        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]

        logger.log_success(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        return df

    def get_ticker(self, symbol: str = None) -> dict:
        """Get current ticker data."""
        symbol = symbol or config.SYMBOL
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.log_error("Failed to fetch ticker", str(e))
            raise

    def get_current_price(self, symbol: str = None) -> float:
        """Get current price."""
        ticker = self.get_ticker(symbol)
        return ticker['last']

    def get_balance(self, currency: str = 'USDT') -> float:
        """Get balance for a currency."""
        try:
            balance = self.exchange.fetch_balance()
            return balance.get(currency, {}).get('free', 0.0)
        except Exception as e:
            logger.log_error("Failed to fetch balance", str(e))
            return 0.0

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Optional[dict]:
        """
        Place a market order.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            quantity: Amount to trade
            stop_loss: Stop-loss price (optional)
            take_profit: Take-profit price (optional)

        Returns:
            Order response dict or None if failed
        """
        try:
            order = self.exchange.create_market_order(
                symbol,
                side,
                quantity
            )

            # Track position if buying
            if side == 'buy':
                self.positions[symbol] = {
                    'entry_price': order.get('average', order.get('price')),
                    'quantity': quantity,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now()
                }
            elif side == 'sell' and symbol in self.positions:
                del self.positions[symbol]

            logger.log_trade(
                action="ORDER_FILLED",
                symbol=symbol,
                side=side,
                price=order.get('average', order.get('price', 0)),
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            return order

        except Exception as e:
            logger.log_error(f"Failed to place {side} order", str(e))
            return None

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Optional[dict]:
        """Place a limit order."""
        try:
            order = self.exchange.create_limit_order(
                symbol,
                side,
                quantity,
                price
            )
            return order
        except Exception as e:
            logger.log_error(f"Failed to place limit order", str(e))
            return None

    def get_open_positions(self) -> Dict[str, dict]:
        """Get all open positions."""
        return self.positions

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)

    def check_stop_loss_take_profit(self, symbol: str = None) -> List[Tuple[str, str, float]]:
        """
        Check if any positions hit stop-loss or take-profit.

        Returns:
            List of (symbol, reason, current_price) tuples for positions to close
        """
        symbol = symbol or config.SYMBOL
        to_close = []

        for sym, position in self.positions.items():
            if symbol and sym != symbol:
                continue

            try:
                current_price = self.get_current_price(sym)
                entry_price = position['entry_price']

                # Check stop-loss
                if position['stop_loss'] and current_price <= position['stop_loss']:
                    to_close.append((sym, 'stop_loss', current_price))

                # Check take-profit
                elif position['take_profit'] and current_price >= position['take_profit']:
                    to_close.append((sym, 'take_profit', current_price))

            except Exception as e:
                logger.log_error(f"Error checking SL/TP for {sym}", str(e))

        return to_close

    def close_position(self, symbol: str, reason: str = "") -> Optional[dict]:
        """Close a position by selling."""
        if symbol not in self.positions:
            logger.log_error(f"No position found for {symbol}")
            return None

        position = self.positions[symbol]
        quantity = position['quantity']

        return self.place_market_order(symbol, 'sell', quantity)


# Global exchange client instance
exchange = ExchangeClient()
