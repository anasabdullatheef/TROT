"""
Exchange Module
Handles order execution via ccxt library for Binance Futures
"""

import ccxt
import os
from typing import Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import config

# Load environment variables
load_dotenv()


class ExchangeClient:
    """
    Wrapper around ccxt for Binance Futures trading
    """

    def __init__(self, testnet: bool = True):
        """
        Initialize exchange client

        Args:
            testnet: If True, use Binance Futures testnet
        """
        self.testnet = testnet

        # Get API credentials from environment
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        # Initialize ccxt Binance client
        self.exchange = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": api_secret,
            "sandbox": testnet,  # Use testnet if True
            "options": {
                "defaultType": "future",
            }
        })

        # Enable rate limiting
        self.exchange.enableRateLimit = True

    def get_balance(self) -> Dict:
        """Get account balance"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get("USDT", {})
            return {
                "total": usdt_balance.get("total", 0),
                "free": usdt_balance.get("free", 0),
                "used": usdt_balance.get("used", 0)
            }
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return {"total": 0, "free": 0, "used": 0}

    def get_position(self, symbol: str = None) -> Optional[Dict]:
        """Get current open position for symbol"""
        if symbol is None:
            symbol = config.SYMBOL

        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                if pos["symbol"] == symbol and pos["contracts"] > 0:
                    return {
                        "symbol": pos["symbol"],
                        "side": pos["side"],  # "long" or "short"
                        "size": pos["contracts"],
                        "entry_price": pos["entryPrice"],
                        "unrealized_pnl": pos["unrealizedPnl"],
                        "leverage": pos["leverage"]
                    }
            return None
        except Exception as e:
            print(f"Error fetching position: {e}")
            return None

    def get_ticker(self, symbol: str = None) -> Dict:
        """Get current ticker/price"""
        if symbol is None:
            symbol = config.SYMBOL

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "symbol": ticker["symbol"],
                "last": ticker["last"],
                "bid": ticker["bid"],
                "ask": ticker["ask"],
                "high": ticker["high"],
                "low": ticker["low"],
                "volume": ticker["baseVolume"]
            }
        except Exception as e:
            print(f"Error fetching ticker: {e}")
            return {}

    def place_market_order(
        self,
        side: str,
        size: float,
        symbol: str = None,
        reduce_only: bool = False
    ) -> Optional[Dict]:
        """
        Place a market order

        Args:
            side: 'buy' or 'sell'
            size: Position size in base currency (BTC)
            symbol: Trading pair
            reduce_only: If True, only reduce position

        Returns:
            Order info dict or None if failed
        """
        if symbol is None:
            symbol = config.SYMBOL

        try:
            params = {}
            if reduce_only:
                params["reduceOnly"] = True

            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=size,
                params=params
            )

            return {
                "id": order["id"],
                "symbol": order["symbol"],
                "side": order["side"],
                "type": order["type"],
                "size": order["amount"],
                "filled": order["filled"],
                "price": order["average"] or order["price"],
                "status": order["status"],
                "timestamp": datetime.fromtimestamp(order["timestamp"] / 1000)
            }
        except Exception as e:
            print(f"Error placing market order: {e}")
            return None

    def place_limit_order(
        self,
        side: str,
        size: float,
        price: float,
        symbol: str = None,
        reduce_only: bool = False
    ) -> Optional[Dict]:
        """Place a limit order"""
        if symbol is None:
            symbol = config.SYMBOL

        try:
            params = {}
            if reduce_only:
                params["reduceOnly"] = True

            order = self.exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=size,
                price=price,
                params=params
            )

            return {
                "id": order["id"],
                "symbol": order["symbol"],
                "side": order["side"],
                "type": order["type"],
                "size": order["amount"],
                "price": order["price"],
                "status": order["status"],
                "timestamp": datetime.fromtimestamp(order["timestamp"] / 1000)
            }
        except Exception as e:
            print(f"Error placing limit order: {e}")
            return None

    def set_stop_loss(
        self,
        side: str,
        size: float,
        stop_price: float,
        symbol: str = None
    ) -> Optional[Dict]:
        """
        Set a stop loss order

        Args:
            side: 'buy' to close short, 'sell' to close long
            size: Position size
            stop_price: Trigger price
        """
        if symbol is None:
            symbol = config.SYMBOL

        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="STOP_MARKET",
                side=side,
                amount=size,
                params={
                    "stopPrice": stop_price,
                    "reduceOnly": True
                }
            )

            return {
                "id": order["id"],
                "type": "stop_loss",
                "stop_price": stop_price,
                "size": size,
                "status": order["status"]
            }
        except Exception as e:
            print(f"Error setting stop loss: {e}")
            return None

    def set_take_profit(
        self,
        side: str,
        size: float,
        take_profit_price: float,
        symbol: str = None
    ) -> Optional[Dict]:
        """
        Set a take profit order

        Args:
            side: 'buy' to close short, 'sell' to close long
            size: Position size
            take_profit_price: Trigger price
        """
        if symbol is None:
            symbol = config.SYMBOL

        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="TAKE_PROFIT_MARKET",
                side=side,
                amount=size,
                params={
                    "stopPrice": take_profit_price,
                    "reduceOnly": True
                }
            )

            return {
                "id": order["id"],
                "type": "take_profit",
                "take_profit_price": take_profit_price,
                "size": size,
                "status": order["status"]
            }
        except Exception as e:
            print(f"Error setting take profit: {e}")
            return None

    def cancel_all_orders(self, symbol: str = None) -> bool:
        """Cancel all open orders for a symbol"""
        if symbol is None:
            symbol = config.SYMBOL

        try:
            self.exchange.cancel_all_orders(symbol)
            return True
        except Exception as e:
            print(f"Error cancelling orders: {e}")
            return False

    def set_leverage(self, leverage: int, symbol: str = None) -> bool:
        """Set leverage for a symbol"""
        if symbol is None:
            symbol = config.SYMBOL

        try:
            self.exchange.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            print(f"Error setting leverage: {e}")
            return False

    def fetch_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = "1h",
        since: int = None,
        limit: int = 1000
    ) -> list:
        """Fetch OHLCV candle data"""
        if symbol is None:
            symbol = config.SYMBOL

        try:
            return self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
        except Exception as e:
            print(f"Error fetching OHLCV: {e}")
            return []


def open_long_position(
    client: ExchangeClient,
    size: float,
    entry_price: float
) -> Optional[Dict]:
    """
    Open a long position with TP/SL

    Returns order info and TP/SL orders
    """
    # Place market buy
    entry_order = client.place_market_order(side="buy", size=size)
    if not entry_order:
        return None

    actual_entry = entry_order["price"] or entry_price

    # Calculate TP/SL prices
    tp_price = actual_entry * (1 + config.TAKE_PROFIT_PCT)
    sl_price = actual_entry * (1 - config.STOP_LOSS_PCT)

    # Set take profit (sell to close long)
    tp_order = client.set_take_profit(
        side="sell",
        size=size,
        take_profit_price=tp_price
    )

    # Set stop loss (sell to close long)
    sl_order = client.set_stop_loss(
        side="sell",
        size=size,
        stop_price=sl_price
    )

    return {
        "entry": entry_order,
        "take_profit": tp_order,
        "stop_loss": sl_order,
        "position_type": "long"
    }


def open_short_position(
    client: ExchangeClient,
    size: float,
    entry_price: float
) -> Optional[Dict]:
    """
    Open a short position with TP/SL

    Returns order info and TP/SL orders
    """
    # Place market sell
    entry_order = client.place_market_order(side="sell", size=size)
    if not entry_order:
        return None

    actual_entry = entry_order["price"] or entry_price

    # Calculate TP/SL prices (reversed for short)
    tp_price = actual_entry * (1 - config.TAKE_PROFIT_PCT)
    sl_price = actual_entry * (1 + config.STOP_LOSS_PCT)

    # Set take profit (buy to close short)
    tp_order = client.set_take_profit(
        side="buy",
        size=size,
        take_profit_price=tp_price
    )

    # Set stop loss (buy to close short)
    sl_order = client.set_stop_loss(
        side="buy",
        size=size,
        stop_price=sl_price
    )

    return {
        "entry": entry_order,
        "take_profit": tp_order,
        "stop_loss": sl_order,
        "position_type": "short"
    }


def close_position(client: ExchangeClient) -> Optional[Dict]:
    """Close any open position at market"""
    position = client.get_position()
    if not position:
        return None

    # Cancel pending orders first
    client.cancel_all_orders()

    # Close position
    side = "sell" if position["side"] == "long" else "buy"
    return client.place_market_order(
        side=side,
        size=position["size"],
        reduce_only=True
    )


if __name__ == "__main__":
    # Test the module (read-only operations only without real API keys)
    print("Testing exchange module...")

    # Initialize client (testnet mode)
    client = ExchangeClient(testnet=True)

    print(f"\nExchange: {client.exchange.id}")
    print(f"Testnet: {client.testnet}")

    # These will work without API keys (public endpoints)
    ticker = client.get_ticker()
    if ticker:
        print(f"\nCurrent BTC/USDT price: ${ticker.get('last', 'N/A')}")

    # Fetch some candles
    candles = client.fetch_ohlcv(limit=5)
    if candles:
        print(f"\nLast 5 candles:")
        for c in candles[-3:]:
            ts = datetime.fromtimestamp(c[0] / 1000)
            print(f"  {ts}: O={c[1]}, H={c[2]}, L={c[3]}, C={c[4]}")

    print("\n(Balance/position operations require valid API keys)")
