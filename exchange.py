"""
Exchange Module
Handles data fetching and order execution via ccxt
Adapted from funding_bot
"""

import ccxt
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import config


class ExchangeClient:
    """Wrapper around ccxt for Binance Futures"""

    def __init__(self, testnet: bool = True):
        self.testnet = testnet

        self.exchange = ccxt.binanceusdm({
            "apiKey": config.BINANCE_API_KEY,
            "secret": config.BINANCE_API_SECRET,
            "sandbox": testnet,
            "options": {"defaultType": "future"}
        })
        self.exchange.enableRateLimit = True

    def get_balance(self) -> Dict:
        """Get account balance"""
        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            return {
                "total": usdt.get("total", 0),
                "free": usdt.get("free", 0),
                "used": usdt.get("used", 0)
            }
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return {"total": 0, "free": 0, "used": 0}

    def get_position(self, symbol: str = None) -> Optional[Dict]:
        """Get current open position"""
        if symbol is None:
            symbol = config.SYMBOL

        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                if pos["symbol"] == symbol and pos["contracts"] > 0:
                    return {
                        "symbol": pos["symbol"],
                        "side": pos["side"],
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
        """Place a market order"""
        if symbol is None:
            symbol = config.SYMBOL

        try:
            params = {"reduceOnly": True} if reduce_only else {}
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
                "size": order["amount"],
                "price": order["average"] or order["price"],
                "status": order["status"],
                "timestamp": datetime.fromtimestamp(order["timestamp"] / 1000)
            }
        except Exception as e:
            print(f"Error placing market order: {e}")
            return None

    def set_stop_loss(self, side: str, size: float, stop_price: float, symbol: str = None) -> Optional[Dict]:
        """Set a stop loss order"""
        if symbol is None:
            symbol = config.SYMBOL

        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="STOP_MARKET",
                side=side,
                amount=size,
                params={"stopPrice": stop_price, "reduceOnly": True}
            )
            return {"id": order["id"], "type": "stop_loss", "stop_price": stop_price}
        except Exception as e:
            print(f"Error setting stop loss: {e}")
            return None

    def set_take_profit(self, side: str, size: float, take_profit_price: float, symbol: str = None) -> Optional[Dict]:
        """Set a take profit order"""
        if symbol is None:
            symbol = config.SYMBOL

        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="TAKE_PROFIT_MARKET",
                side=side,
                amount=size,
                params={"stopPrice": take_profit_price, "reduceOnly": True}
            )
            return {"id": order["id"], "type": "take_profit", "take_profit_price": take_profit_price}
        except Exception as e:
            print(f"Error setting take profit: {e}")
            return None

    def cancel_all_orders(self, symbol: str = None) -> bool:
        """Cancel all open orders"""
        if symbol is None:
            symbol = config.SYMBOL
        try:
            self.exchange.cancel_all_orders(symbol)
            return True
        except Exception as e:
            print(f"Error cancelling orders: {e}")
            return False

    def fetch_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = "1h",
        since: int = None,
        limit: int = 1000
    ) -> List:
        """Fetch OHLCV data"""
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


def fetch_historical_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_date: datetime = None,
    end_date: datetime = None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance Futures public API
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
            "limit": 1500
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_candles.extend(data)
            last_ts = data[-1][0]
            current_start = datetime.fromtimestamp(last_ts / 1000) + timedelta(hours=1)

            print(f"  Fetched {len(data)} candles, total: {len(all_candles)}")
            time.sleep(config.RATE_LIMIT_DELAY)

        except Exception as e:
            print(f"Error fetching OHLCV: {e}")
            break

    if not all_candles:
        return pd.DataFrame()

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


def open_long_position(client: ExchangeClient, size: float, entry_price: float, atr: float) -> Optional[Dict]:
    """Open a long position with dynamic TP/SL based on ATR"""
    entry_order = client.place_market_order(side="buy", size=size)
    if not entry_order:
        return None

    actual_entry = entry_order["price"] or entry_price
    tp_price = actual_entry + (atr * config.ATR_TAKE_PROFIT_MULTIPLIER)
    sl_price = actual_entry - (atr * config.ATR_STOP_LOSS_MULTIPLIER)

    tp_order = client.set_take_profit(side="sell", size=size, take_profit_price=tp_price)
    sl_order = client.set_stop_loss(side="sell", size=size, stop_price=sl_price)

    return {
        "entry": entry_order,
        "take_profit": tp_order,
        "stop_loss": sl_order,
        "position_type": "long",
        "tp_price": tp_price,
        "sl_price": sl_price
    }


def open_short_position(client: ExchangeClient, size: float, entry_price: float, atr: float) -> Optional[Dict]:
    """Open a short position with dynamic TP/SL based on ATR"""
    entry_order = client.place_market_order(side="sell", size=size)
    if not entry_order:
        return None

    actual_entry = entry_order["price"] or entry_price
    tp_price = actual_entry - (atr * config.ATR_TAKE_PROFIT_MULTIPLIER)
    sl_price = actual_entry + (atr * config.ATR_STOP_LOSS_MULTIPLIER)

    tp_order = client.set_take_profit(side="buy", size=size, take_profit_price=tp_price)
    sl_order = client.set_stop_loss(side="buy", size=size, stop_price=sl_price)

    return {
        "entry": entry_order,
        "take_profit": tp_order,
        "stop_loss": sl_order,
        "position_type": "short",
        "tp_price": tp_price,
        "sl_price": sl_price
    }


def close_position(client: ExchangeClient) -> Optional[Dict]:
    """Close any open position at market"""
    position = client.get_position()
    if not position:
        return None

    client.cancel_all_orders()
    side = "sell" if position["side"] == "long" else "buy"
    return client.place_market_order(side=side, size=position["size"], reduce_only=True)


if __name__ == "__main__":
    print("Testing exchange module...")

    # Test public API
    ohlcv = fetch_historical_ohlcv(
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now()
    )
    print(f"Fetched {len(ohlcv)} candles")
    print(ohlcv.tail())
