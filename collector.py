#!/usr/bin/env python3
"""
Data Collector - Runs 24/7 via cron to collect real order flow data
Saves to SQLite for future backtesting with real leading indicators

Run via cron every minute:
* * * * * cd /path/to/atlas_bot && ./venv/bin/python collector.py >> logs/collector.log 2>&1

Data collected:
- Binance order book top 20 levels (bid/ask volumes)
- Binance recent trades (buy vs sell initiated volume)
- Binance liquidations
- Cross-exchange prices (Binance, Bybit, OKX)
"""

import sqlite3
import json
import time
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import ccxt
import os

# Configuration
DB_PATH = "data/orderflow.db"
SYMBOL = "BTCUSDT"
ORDERBOOK_LEVELS = 20
TRADES_LIMIT = 500


def init_db():
    """Initialize SQLite database with tables"""
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Order book snapshots
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orderbook (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            unix_ts INTEGER NOT NULL,
            bid_volume_total REAL,
            ask_volume_total REAL,
            bid_volume_top5 REAL,
            ask_volume_top5 REAL,
            bid_volume_top10 REAL,
            ask_volume_top10 REAL,
            imbalance REAL,
            spread_bps REAL,
            best_bid REAL,
            best_ask REAL,
            raw_data TEXT
        )
    """)

    # Trade flow (aggregated per collection)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tradeflow (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            unix_ts INTEGER NOT NULL,
            buy_volume REAL,
            sell_volume REAL,
            buy_count INTEGER,
            sell_count INTEGER,
            vpin REAL,
            trade_imbalance REAL,
            avg_trade_size REAL,
            large_trade_volume REAL
        )
    """)

    # Liquidations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS liquidations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            unix_ts INTEGER NOT NULL,
            price REAL,
            quantity REAL,
            side TEXT,
            notional_value REAL
        )
    """)

    # Cross-exchange prices
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cross_exchange (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            unix_ts INTEGER NOT NULL,
            binance_price REAL,
            bybit_price REAL,
            okx_price REAL,
            max_divergence REAL,
            binance_vs_bybit REAL,
            binance_vs_okx REAL
        )
    """)

    # Create indices for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_ts ON orderbook(unix_ts)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tradeflow_ts ON tradeflow(unix_ts)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_liquidations_ts ON liquidations(unix_ts)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cross_exchange_ts ON cross_exchange(unix_ts)")

    conn.commit()
    conn.close()


def collect_orderbook():
    """Collect Binance order book snapshot"""
    url = f"https://fapi.binance.com/fapi/v1/depth"
    params = {"symbol": SYMBOL, "limit": ORDERBOOK_LEVELS}

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
        asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]

        bid_volume_total = sum(b[1] for b in bids)
        ask_volume_total = sum(a[1] for a in asks)
        bid_volume_top5 = sum(b[1] for b in bids[:5])
        ask_volume_top5 = sum(a[1] for a in asks[:5])
        bid_volume_top10 = sum(b[1] for b in bids[:10])
        ask_volume_top10 = sum(a[1] for a in asks[:10])

        total = bid_volume_total + ask_volume_total
        imbalance = (bid_volume_total - ask_volume_total) / total if total > 0 else 0

        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread_bps = ((best_ask - best_bid) / best_bid * 10000) if best_bid > 0 else 0

        return {
            "bid_volume_total": bid_volume_total,
            "ask_volume_total": ask_volume_total,
            "bid_volume_top5": bid_volume_top5,
            "ask_volume_top5": ask_volume_top5,
            "bid_volume_top10": bid_volume_top10,
            "ask_volume_top10": ask_volume_top10,
            "imbalance": imbalance,
            "spread_bps": spread_bps,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "raw_data": json.dumps({"bids": bids[:5], "asks": asks[:5]})
        }
    except Exception as e:
        print(f"[{datetime.now()}] Orderbook error: {e}")
        return None


def collect_trades():
    """Collect recent trades and compute VPIN"""
    url = f"https://fapi.binance.com/fapi/v1/aggTrades"
    params = {"symbol": SYMBOL, "limit": TRADES_LIMIT}

    try:
        response = requests.get(url, params=params, timeout=10)
        trades = response.json()

        buy_volume = 0
        sell_volume = 0
        buy_count = 0
        sell_count = 0
        large_trade_volume = 0

        for t in trades:
            qty = float(t["q"])
            price = float(t["p"])
            notional = qty * price

            # is_buyer_maker = True means sell-initiated (taker sold into bid)
            if t["m"]:  # Sell-initiated
                sell_volume += qty
                sell_count += 1
            else:  # Buy-initiated
                buy_volume += qty
                buy_count += 1

            # Large trade threshold: > $100K notional
            if notional > 100000:
                large_trade_volume += qty

        total_volume = buy_volume + sell_volume
        vpin = buy_volume / total_volume if total_volume > 0 else 0.5
        imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        total_count = buy_count + sell_count
        avg_size = total_volume / total_count if total_count > 0 else 0

        return {
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "vpin": vpin,
            "trade_imbalance": imbalance,
            "avg_trade_size": avg_size,
            "large_trade_volume": large_trade_volume
        }
    except Exception as e:
        print(f"[{datetime.now()}] Trades error: {e}")
        return None


def collect_liquidations():
    """Collect recent liquidations from Binance"""
    url = "https://fapi.binance.com/fapi/v1/allForceOrders"
    params = {"symbol": SYMBOL, "limit": 100}

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        liquidations = []
        for liq in data:
            liquidations.append({
                "price": float(liq.get("price", 0)),
                "quantity": float(liq.get("origQty", 0)),
                "side": liq.get("side", ""),
                "notional_value": float(liq.get("price", 0)) * float(liq.get("origQty", 0))
            })

        return liquidations
    except Exception as e:
        # This endpoint may require authentication or have rate limits
        return []


def collect_cross_exchange_prices():
    """Collect prices from multiple exchanges"""
    prices = {}

    def fetch_binance():
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/price"
            params = {"symbol": SYMBOL}
            response = requests.get(url, params=params, timeout=5)
            return "binance", float(response.json()["price"])
        except:
            return "binance", None

    def fetch_bybit():
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            params = {"category": "linear", "symbol": SYMBOL}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return "bybit", float(data["result"]["list"][0]["lastPrice"])
        except:
            return "bybit", None

    def fetch_okx():
        try:
            url = "https://www.okx.com/api/v5/market/ticker"
            params = {"instId": "BTC-USDT-SWAP"}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return "okx", float(data["data"][0]["last"])
        except:
            return "okx", None

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(fetch_binance),
            executor.submit(fetch_bybit),
            executor.submit(fetch_okx)
        ]
        for future in as_completed(futures):
            name, price = future.result()
            if price:
                prices[name] = price

    # Calculate divergences
    binance = prices.get("binance")
    bybit = prices.get("bybit")
    okx = prices.get("okx")

    divergences = []
    binance_vs_bybit = 0
    binance_vs_okx = 0

    if binance and bybit:
        binance_vs_bybit = (binance - bybit) / binance
        divergences.append(abs(binance_vs_bybit))

    if binance and okx:
        binance_vs_okx = (binance - okx) / binance
        divergences.append(abs(binance_vs_okx))

    max_divergence = max(divergences) if divergences else 0

    return {
        "binance_price": binance,
        "bybit_price": bybit,
        "okx_price": okx,
        "max_divergence": max_divergence,
        "binance_vs_bybit": binance_vs_bybit,
        "binance_vs_okx": binance_vs_okx
    }


def save_to_db(orderbook, tradeflow, liquidations, cross_exchange):
    """Save all collected data to SQLite"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.utcnow()
    timestamp = now.isoformat()
    unix_ts = int(now.timestamp())

    # Save orderbook
    if orderbook:
        cursor.execute("""
            INSERT INTO orderbook (
                timestamp, unix_ts, bid_volume_total, ask_volume_total,
                bid_volume_top5, ask_volume_top5, bid_volume_top10, ask_volume_top10,
                imbalance, spread_bps, best_bid, best_ask, raw_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, unix_ts,
            orderbook["bid_volume_total"], orderbook["ask_volume_total"],
            orderbook["bid_volume_top5"], orderbook["ask_volume_top5"],
            orderbook["bid_volume_top10"], orderbook["ask_volume_top10"],
            orderbook["imbalance"], orderbook["spread_bps"],
            orderbook["best_bid"], orderbook["best_ask"],
            orderbook["raw_data"]
        ))

    # Save tradeflow
    if tradeflow:
        cursor.execute("""
            INSERT INTO tradeflow (
                timestamp, unix_ts, buy_volume, sell_volume,
                buy_count, sell_count, vpin, trade_imbalance,
                avg_trade_size, large_trade_volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, unix_ts,
            tradeflow["buy_volume"], tradeflow["sell_volume"],
            tradeflow["buy_count"], tradeflow["sell_count"],
            tradeflow["vpin"], tradeflow["trade_imbalance"],
            tradeflow["avg_trade_size"], tradeflow["large_trade_volume"]
        ))

    # Save liquidations
    for liq in liquidations:
        cursor.execute("""
            INSERT INTO liquidations (
                timestamp, unix_ts, price, quantity, side, notional_value
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            timestamp, unix_ts,
            liq["price"], liq["quantity"], liq["side"], liq["notional_value"]
        ))

    # Save cross-exchange
    if cross_exchange and cross_exchange.get("binance_price"):
        cursor.execute("""
            INSERT INTO cross_exchange (
                timestamp, unix_ts, binance_price, bybit_price, okx_price,
                max_divergence, binance_vs_bybit, binance_vs_okx
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, unix_ts,
            cross_exchange["binance_price"], cross_exchange["bybit_price"],
            cross_exchange["okx_price"], cross_exchange["max_divergence"],
            cross_exchange["binance_vs_bybit"], cross_exchange["binance_vs_okx"]
        ))

    conn.commit()
    conn.close()


def get_db_stats():
    """Get current database statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    stats = {}
    for table in ["orderbook", "tradeflow", "liquidations", "cross_exchange"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM orderbook")
    row = cursor.fetchone()
    stats["date_range"] = (row[0], row[1]) if row[0] else ("N/A", "N/A")

    conn.close()
    return stats


def run_collection():
    """Run a single collection cycle"""
    start = time.time()

    # Collect all data in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        orderbook_future = executor.submit(collect_orderbook)
        trades_future = executor.submit(collect_trades)
        liquidations_future = executor.submit(collect_liquidations)
        prices_future = executor.submit(collect_cross_exchange_prices)

        orderbook = orderbook_future.result()
        tradeflow = trades_future.result()
        liquidations = liquidations_future.result()
        cross_exchange = prices_future.result()

    # Save to database
    save_to_db(orderbook, tradeflow, liquidations, cross_exchange)

    elapsed = time.time() - start

    # Log summary
    now = datetime.utcnow().isoformat()
    ob_imb = orderbook["imbalance"] if orderbook else 0
    vpin = tradeflow["vpin"] if tradeflow else 0
    n_liqs = len(liquidations)
    div = cross_exchange["max_divergence"] * 100 if cross_exchange else 0

    print(f"[{now}] Collected: OB_imb={ob_imb:+.3f} VPIN={vpin:.3f} Liqs={n_liqs} Div={div:.4f}% ({elapsed:.2f}s)")


def main():
    """Main entry point"""
    # Initialize database
    init_db()

    # Run collection
    run_collection()

    # Show stats periodically (every 10th run based on minute)
    minute = datetime.utcnow().minute
    if minute % 10 == 0:
        stats = get_db_stats()
        print(f"    DB Stats: {stats['orderbook']} orderbook, {stats['tradeflow']} trades, "
              f"{stats['liquidations']} liqs, {stats['cross_exchange']} prices")
        print(f"    Range: {stats['date_range'][0]} to {stats['date_range'][1]}")


if __name__ == "__main__":
    main()
